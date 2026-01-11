"""Monte Carlo Simulation Engine for Congressional Voting Dynamics.

This module provides:
- GPU-accelerated Monte Carlo simulation ensemble for voting outcomes
- Vectorized sensitivity analysis for actor and regime parameters
- Trajectory visualization for convergence/chaos detection
- Integration with RNN (ideology evolution) and DQN (voting decisions)
"""

import logging
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Callable

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:  # pragma: no cover - optional dependency
    Parallel = None  # type: ignore[assignment]
    delayed = None  # type: ignore[assignment]
    _HAS_JOBLIB = False

# Optional tqdm for console progress bars
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = None
    _HAS_TQDM = False

import numpy as np

import config
from congress.actors import CongressMan
from congress.law import Law
from gpu_utils import get_array_module, has_cupy, to_cpu_array
from congress.congress import TheCongress
from public_opinion.regime import PublicRegime
from utils import RewardComputer as BaseRewardComputer

logger = logging.getLogger(__name__)

# Optional matplotlib for trajectory visualization
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False


class RewardComputer(BaseRewardComputer):
    """Reward composer specialized for simulation loop.

    Combines party loyalty, IPM alignment, and electoral alignment using
    weights defined in config, and adds an explicit party-alignment bonus
    to make the reward signal clearer for the DQN agent.
    """

    @staticmethod
    def compute_reward(
        vote_action: int,
        party_line_vote: int,
        actor_ideology: np.ndarray,
        law_salience: np.ndarray,
        law_threshold: float,
        district_preference: float,
        loyalty_weight: float | None = None,
        ipm_weight: float | None = None,
        electoral_weight: float | None = None,
    ) -> Tuple[float, Dict[str, float]]:
        loyalty_w = loyalty_weight if loyalty_weight is not None else config.REWARD_LOYALTY_WEIGHT
        ipm_w = ipm_weight if ipm_weight is not None else config.REWARD_IPM_WEIGHT
        electoral_w = electoral_weight if electoral_weight is not None else config.REWARD_ELECTORAL_WEIGHT

        loyalty_reward = BaseRewardComputer.compute_loyalty_reward(vote_action, party_line_vote, loyalty_w)
        ipm_reward = BaseRewardComputer.compute_ipm_reward(actor_ideology, law_salience, law_threshold, vote_action, ipm_w)
        electoral_reward = BaseRewardComputer.compute_electoral_reward(vote_action, district_preference, electoral_w)

        weighted_sum = (loyalty_reward * loyalty_w) + (ipm_reward * ipm_w) + (electoral_reward * electoral_w)
        weight_total = max(loyalty_w + ipm_w + electoral_w, 1e-6)
        composite = weighted_sum / weight_total

        party_bonus = config.REWARD_PARTY_ALIGNMENT_BONUS if vote_action == party_line_vote else -config.REWARD_PARTY_ALIGNMENT_BONUS
        total = float(np.clip(composite + 0.1 * party_bonus, -1.0, 1.0))

        breakdown = {
            "loyalty": float(loyalty_reward),
            "ipm": float(ipm_reward),
            "electoral": float(electoral_reward),
            "party_bonus": float(party_bonus),
            "composite": float(composite),
        }

        return total, breakdown


class CongressEngine:
    """GPU-accelerated Monte Carlo simulation engine for congressional voting outcomes."""

    def __init__(
        self,
        congress_template: TheCongress,
        use_gpu: bool = True,
        ipm_params: Optional[Dict] = None,
        dbn_model: Optional[object] = None,
        use_hmm_state: bool = True
    ) -> None:
        """Initialize engine with a congress template.

        Args:
            congress_template: Template Congress system for cloning.
            use_gpu: Whether to use GPU acceleration if available (default: True).
            ipm_params: IPM voting parameters dict with 'salience' and 'threshold' arrays.
                       If provided, will be used to create realistic laws during simulation.
            dbn_model: Trained DBCongressModel instance for ideology evolution.
                      If provided, will be used to evolve actor ideologies during simulation.
        """
        self.template = congress_template
        self.use_gpu = use_gpu and has_cupy()
        self.xp = get_array_module(self.use_gpu)
        self.ipm_params = ipm_params  # Store IPM parameters for law generation
        self.dbn_model = dbn_model  # Store DBN model for ideology evolution
        self.use_hmm_state = use_hmm_state
        if self.use_gpu:
            logger.info("CongressEngine initialized with GPU acceleration")
        else:
            logger.info("CongressEngine initialized with CPU (NumPy)")
        
        if ipm_params is not None:
            logger.info(f"IPM voting parameters available: salience shape={ipm_params.get('salience', np.array([])).shape}")
        
        if dbn_model is not None:
            logger.info(f"DBN model available for ideology evolution")
        if not use_hmm_state:
            logger.info("HMM inference disabled for regime updates")
    
    def _rand_int(self, low: int, high: int) -> int:
        """Return a random integer draw using the currently selected array module."""
        return int(self.xp.random.randint(low, high))

    def _rand_random(self) -> float:
        """Return a scalar draw in [0,1) from the active random generator."""
        return float(self.xp.random.random())

    def _rand_uniform(self, low: float, high: float) -> float:
        """Return a uniform scalar draw between low and high."""
        return float(self.xp.random.uniform(low, high))

    def _rand_normal(self, shape: Any, scale: float) -> np.ndarray:
        """Draw a normal array on the active backend and move it back to NumPy."""
        arr = self.xp.random.normal(0.0, scale, size=shape, dtype=self.xp.float32)
        return to_cpu_array(arr)

    def init_rnn_for_all_actors(
        self,
        input_dim: int = config.LSTM_INPUT_DIM,
        hidden_dim: int = config.LSTM_HIDDEN_DIM,
        output_dim: int = config.LSTM_OUTPUT_DIM
    ) -> None:
        """Initialize RNN models for all actors in template.
        
        Args:
            input_dim: Ideology dimensionality.
            hidden_dim: LSTM hidden state size.
            output_dim: Output ideology dimensionality.
        """
        for actor in self.template.actors:
            actor.init_rnn_model(input_dim, hidden_dim, output_dim)
        logger.info(f"Initialized RNN models for {len(self.template.actors)} actors")
    
    def init_dqn_for_all_actors(
        self,
        state_dim: int = config.DQN_STATE_DIM,
        hidden_dims: List[int] = None
    ) -> None:
        """Initialize DQN agents for all actors in template.
        
        Args:
            state_dim: State vector dimensionality.
            hidden_dims: Hidden layer dimensions for DQN.
        """
        if hidden_dims is None:
            hidden_dims = config.DQN_HIDDEN_DIMS
        
        for actor in self.template.actors:
            actor.init_dqn_agent(state_dim, hidden_dims)
        logger.info(f"Initialized DQN agents for {len(self.template.actors)} actors")

    def run_monte_carlo(
        self,
        n_simulations: int = 100,
        steps: int = 5,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        show_tqdm: bool = True,
        progress_per_step: bool = True,
        show_small_steps: bool = True,
        show_step_tqdm: bool = True,
        show_component_tqdm: bool = True,
        n_jobs: int = 1,
        joblib_backend: str = "loky",
    ) -> Dict[str, Any]:
        """Run GPU-accelerated Monte Carlo simulation ensemble with IPM-based voting.

        Simulates multiple congressional sessions where:
        1. For each simulation step, a new Law is generated (from IPM params or randomly)
        2. Each actor decides their vote using IPM + DBN influence + regime pressure
        3. Vote outcomes are aggregated to determine law passage

        New args for progress tracking:
            progress_callback: Optional[callable] that receives a dict with progress info.
            show_tqdm: If True and `tqdm` is installed, show a progress bar in console.
            progress_per_step: If True, callback is invoked after each simulation step (more detailed).

        Returns:
            Dictionary containing aggregated results and metrics.
        """
        # Prepare optional progress iterator / callback
        if show_tqdm and not _HAS_TQDM:
            logger.warning("tqdm requested for progress display but package is not installed; continuing without progress bar.")

        total_actors = len(self.template.actors)
        iterator = range(n_simulations)
        if show_tqdm and _HAS_TQDM:
            desc = (
                f"Simulations ({n_simulations} sims, actors={total_actors}, steps={steps}, "
                f"GPU={'yes' if self.use_gpu else 'no'}, scenario={getattr(self.template.regime_engine, 'scenario', 'unknown')})"
            )
            iterator = tqdm(range(n_simulations), desc=desc, unit="sim", leave=True)

        # Notify start
        if progress_callback is not None:
            try:
                progress_callback({"phase": "start", "n_simulations": n_simulations, "steps": steps})
            except Exception as e:
                logger.debug(f"Progress callback start hook failed: {e}")

        effective_jobs = 1 if self.use_gpu else max(1, n_jobs)
        if effective_jobs > 1 and not _HAS_JOBLIB:
            logger.warning("joblib not available; falling back to n_jobs=1")
            effective_jobs = 1

        # Parallel branch (CPU-only) delegates to single-simulation runs to avoid GPU contention
        if effective_jobs > 1:
            try:
                logger.info(
                    f"Parallelizing Monte Carlo across {effective_jobs} CPU workers (GPU disabled for workers)."
                )
                parallel_results = Parallel(n_jobs=effective_jobs, backend=joblib_backend)(
                    delayed(self.run_monte_carlo)(
                        n_simulations=1,
                        steps=steps,
                        progress_callback=None,
                        show_tqdm=False,
                        progress_per_step=False,
                        show_small_steps=False,
                        show_step_tqdm=False,
                        show_component_tqdm=False,
                        n_jobs=1,
                        joblib_backend=joblib_backend,
                    )
                    for _ in range(n_simulations)
                )

                vote_distributions = np.array([
                    res.get("distributions", [0])[0] for res in parallel_results
                ], dtype=np.int32)
                trajectories = np.array([
                    res.get("trajectories", [[0.0] * steps])[0] for res in parallel_results
                ], dtype=float)
                pass_count = int(sum(res.get("probability_of_passing", 0.0) >= 0.5 for res in parallel_results))
                critical_failures = int(sum(res.get("risk_of_flip", 0.0) >= 0.5 for res in parallel_results))

                results: Dict[str, Any] = {
                    "pass_count": pass_count,
                    "vote_distributions": vote_distributions.tolist(),
                    "critical_failures": critical_failures,
                    "trajectories": trajectories.tolist(),
                }

                compiled = self._compile_report(results, n_simulations)

                if progress_callback is not None:
                    try:
                        progress_callback({"phase": "complete", "report": compiled})
                    except Exception as e:
                        logger.debug(f"Progress callback (complete) failed: {e}")

                return compiled
            except Exception as parallel_error:
                logger.debug("Parallel Monte Carlo failed; reverting to sequential: %s", parallel_error)

        # Pre-allocate GPU arrays for batch processing
        vote_distributions = self.xp.zeros(n_simulations, dtype=self.xp.int32)
        trajectories = self.xp.zeros((n_simulations, steps), dtype=self.xp.float32)
        pass_count = 0
        critical_failures = 0
        total_actors = len(self.template.actors)
        half_actors = total_actors / 2

        logger.info(f"Running {n_simulations} simulations with {steps} steps each (GPU: {self.use_gpu})...")
        logger.info(f"IPM-based voting: {'ENABLED' if self.ipm_params else 'DISABLED (using random)'}")

        for sim_idx in iterator:
            # Lightweight clone of template
            template = self.template
            actors_clone = [CongressMan(a.id, a.get_state_vector()) for a in template.actors]
            # Carry over trained RNN/DQN models from template actors where present so per-sim prediction/training works
            for idx, a in enumerate(template.actors):
                if idx < len(actors_clone):
                    if a.rnn_model is not None:
                        actors_clone[idx].rnn_model = a.rnn_model
                        actors_clone[idx].rnn_trainer = a.rnn_trainer
                    if a.dqn_agent is not None:
                        actors_clone[idx].dqn_agent = a.dqn_agent
                        actors_clone[idx].use_dqn = a.use_dqn

            adj_clone = np.array(template.adj_matrix, copy=True)
            sim = TheCongress(actors_clone, adj_clone, template.regime_engine, use_gpu=self.use_gpu)
            # Surface GPU flags and array module to the sim instance for GPU-aware ops
            sim.xp = self.xp
            sim.use_gpu = self.use_gpu
            # Keep last sim instance for test inspection / debugging
            self._last_sim_instance = sim
            # Reset regime state for this independent simulation run
            try:
                sim.regime_engine.reset()
            except Exception:
                pass

            # Determine dimensionality for law generation
            if len(actors_clone) > 0:
                actor_ideology = actors_clone[0].ideology
                if isinstance(actor_ideology, np.ndarray):
                    law_dim = len(actor_ideology)
                else:
                    law_dim = 1
            else:
                law_dim = 3

            # Optionally use a per-simulation step progress bar
            t_iterator = range(steps)
            if show_step_tqdm and _HAS_TQDM:
                desc = f"Sim {sim_idx+1}/{n_simulations} steps (actors={len(sim.actors)})"
                t_iterator = tqdm(range(steps), desc=desc, unit="step", leave=False)

            for t in t_iterator:
                # Step 1: Get current system state
                current_pressure = sim.regime_engine.get_current_pressure()
                
                # Step 2: Generate a law for this round
                if self.ipm_params is not None and 'salience' in self.ipm_params:
                    # Generate law from real IPM parameters
                    salience_matrix = self.ipm_params.get('salience')  # Shape: (n_votes, dim)
                    threshold_vector = self.ipm_params.get('threshold')  # Shape: (n_votes,)
                    
                    if len(salience_matrix) > 0:
                        # Randomly select a vote from IPM parameters
                        vote_idx = self._rand_int(0, len(salience_matrix))
                        law = Law(
                            law_id=t,
                            salience=salience_matrix[vote_idx],
                            threshold=float(threshold_vector[vote_idx]),
                            title=f"Bill_{sim_idx}_{t}"
                        )
                    else:
                        # Fallback to random generation if no votes available
                        law = Law.create_random(law_id=t, dim=law_dim, title=f"Bill_{sim_idx}_{t}")
                else:
                    # Generate random law
                    law = Law.create_random(law_id=t, dim=law_dim, title=f"Bill_{sim_idx}_{t}")
                
                # Step 3: Get contextual influence matrix and vector (use for vote decisions)
                contextual_W = sim.get_contextual_influence_matrix(law, current_pressure)
                # Temporarily replace adj_matrix for influence computation
                old_adj = sim.adj_matrix
                sim.adj_matrix = contextual_W
                influence_vector = sim.get_network_influence_vector()
                # Restore original adj_matrix before reward updates
                sim.adj_matrix = old_adj
                
                # Timing instrumentation for step and components
                step_start_time = time.perf_counter()
                comp_times = {
                    "rnn": 0.0,
                    "dbn": 0.0,
                    "voting": 0.0,
                    "rewards": 0.0,
                    "hmm": 0.0,
                    "other": 0.0,
                }

                z_context = None
                if self.dbn_model is not None:
                    try:
                        z_context = sim.regime_engine.get_context_vector()
                    except Exception as error:
                        logger.debug("Failed to pull regime context for DBN: %s", error)

                # Step 3.5: EVOLVE IDEOLOGIES USING RNN OR DBN (ideology evolution will use contextual W later)
                # Use RNN if available (individual actor memory), fallback to DBN (network-based)
                # Optionally show per-actor progress for evolution
                actor_iter = list(enumerate(sim.actors))
                if show_component_tqdm and _HAS_TQDM:
                    desc = f"Sim {sim_idx+1}/{n_simulations} step {t+1} evol (actors={len(sim.actors)})"
                    actor_iter = list(enumerate(tqdm(sim.actors, desc=desc, leave=False, unit="actor")))

                for i, actor in actor_iter:
                    if actor.rnn_model is not None:
                        # RNN-based evolution (individual actor memory)
                        try:
                            s0 = time.perf_counter()
                            # Get RNN prediction for next ideology
                            next_ideology_rnn = actor.predict_ideology_rnn(current_pressure)
                            s1 = time.perf_counter()
                            comp_times["rnn"] += (s1 - s0)
                            
                            # Apply DBN influence shift if available
                            dbn_shift = influence_vector[i] * 0.1 if self.dbn_model is not None else 0.0
                            
                            # Apply regime volatility
                            regime_noise = self._rand_normal(
                                next_ideology_rnn.shape,
                                sim.regime_engine.volatility_multiplier * 0.01
                            )
                            
                            # Combine: RNN prediction + DBN influence + noise
                            new_ideology = next_ideology_rnn + dbn_shift + regime_noise
                            
                            # Clip to bounds
                            actor.ideology = np.clip(new_ideology, -1.0, 1.0)
                            
                            # Update history for next step
                            actor.update_ideology_history(actor.ideology)
                            
                            # Online training (optional, uses actual outcome as target)
                            if config.LSTM_ONLINE_UPDATE_INTERVAL > 0 and (t + 1) % config.LSTM_ONLINE_UPDATE_INTERVAL == 0:
                                actor.train_rnn_step(actor.ideology, current_pressure)
                            
                            logger.debug(f"RNN evolution for actor {actor.id}: {actor.ideology}")
                        except Exception as e:
                            logger.debug(f"RNN evolution failed for actor {actor.id}: {e}. Falling back to DBN.")
                    elif self.dbn_model is not None:
                        # DBN-based evolution (network influence only)
                        try:
                            s0 = time.perf_counter()
                            # Collect current ideologies from all actors for DBN
                            current_ideologies = np.array([a.ideology for a in sim.actors], dtype=np.float32)
                            # Ensure CPU-backed numpy array for DBN (sklearn-based CPU model)
                            current_ideologies = to_cpu_array(current_ideologies)

                            # Flatten if needed
                            if current_ideologies.ndim > 1:
                                current_ideologies_flat = current_ideologies.flatten()
                            else:
                                current_ideologies_flat = current_ideologies

                            # Predict next state
                            next_ideologies_flat = self.dbn_model.step(current_ideologies_flat, Z_current=z_context)
                            s1 = time.perf_counter()
                            comp_times["dbn"] += (s1 - s0)
                            
                            # Reshape and update
                            if current_ideologies.ndim > 1:
                                next_ideologies = next_ideologies_flat.reshape(current_ideologies.shape)
                            else:
                                next_ideologies = next_ideologies_flat[:len(sim.actors)]
                            
                            # Add noise
                            regime_noise = self._rand_normal(
                                next_ideologies.shape,
                                sim.regime_engine.volatility_multiplier * 0.01
                            )
                            
                            next_ideologies = np.clip(next_ideologies + regime_noise, -1.0, 1.0)
                            
                            # Update all actors
                            for j, actor in enumerate(sim.actors):
                                if current_ideologies.ndim > 1:
                                    actor.ideology = next_ideologies[j]
                                else:
                                    actor.ideology = np.array([next_ideologies[j]])
                                try:
                                    actor.update_ideology_history(actor.ideology)
                                except Exception:
                                    pass
                            
                            logger.debug(f"DBN evolution applied at step {t}")
                        except Exception as e:
                            logger.debug(f"DBN evolution failed: {e}")
                    # else: no evolution, ideology stays constant
                    # D: Bargaining phase – leaders try to buy enough swing votes
                    num_actors = len(sim.actors)
                    if num_actors > 0:
                        base_yes_votes = sum(1 for actor in sim.actors if actor.current_opinion > 0.0)
                        base_acceptance_rate = base_yes_votes / float(num_actors)
                    else:
                        base_acceptance_rate = 0.0

                    leader = sim.get_majority_leader()
                    if (0.45 <= base_acceptance_rate < 0.5 and leader and leader.party
                            and leader.political_capital >= 1.0):
                        logger.info("  Bargaining triggered: Pushing critical bill through.")
                        swing_voters = [
                            actor for actor in sim.actors
                            if (actor.party == leader.party
                                and abs(actor.current_opinion) < 0.4
                                and abs(actor.vote_prob - 0.5) < 0.2)
                        ]
                        cost_per_vote = 1.0
                        for swing in swing_voters:
                            if leader.political_capital < cost_per_vote:
                                break
                            leader.political_capital = max(0.0, leader.political_capital - cost_per_vote)
                            swing.vote_boost = min(0.85, swing.vote_boost + 0.45)
                            swing.current_opinion = 0.95
                            logger.info(f"    Actor {swing.id} flipped to YES (cost: {cost_per_vote:.1f} PC).")
                
                # Step 4: Each actor votes using IPM/DQN + DBN influence + Regime pressure
                yea_votes_for_step = 0  # Reset for each step
                actor_votes = []  # Track votes for learning
                party_line_cache: Dict[Optional[str], int] = {}

                voting_start = time.perf_counter()
                def _get_party_line(party_key: Optional[str]) -> int:
                    if party_key not in party_line_cache:
                        party_line_cache[party_key] = sim.get_party_line(party_key, law)
                    return party_line_cache[party_key]

                def _get_party_line(party_key: Optional[str]) -> int:
                    if party_key not in party_line_cache:
                        party_line_cache[party_key] = sim.get_party_line(party_key, law)
                    return party_line_cache[party_key]

                # Optionally show per-actor progress for voting
                vote_iter = list(enumerate(sim.actors))
                if show_component_tqdm and _HAS_TQDM:
                    desc = f"Sim {sim_idx+1}/{n_simulations} step {t+1} voting (actors={len(sim.actors)})"
                    vote_iter = list(enumerate(tqdm(sim.actors, desc=desc, leave=False, unit="actor")))

                for i, actor in vote_iter:
                    party_line_vote = _get_party_line(actor.party)
                    voted_with_party = False
                    try:
                        # Determine vote using DQN or IPM
                        if actor.use_dqn and actor.dqn_agent is not None:
                            vote_result = actor.decide_vote_dqn(
                                law=law,
                                influence_net=influence_vector[i],
                                regime_pressure=current_pressure,
                                use_exploration=True
                            )
                        else:
                            vote_result = actor.decide_vote(
                                law=law,
                                influence_effect=influence_vector[i],
                                regime_pressure=current_pressure
                            )

                        if vote_result:
                            yea_votes_for_step += 1

                        voted_with_party = (vote_result == party_line_vote)
                        actor_votes.append((i, vote_result))
                        # Record vote history immediately so state updates (EWMA, recent votes) happen before capital/loyalty updates
                        try:
                            actor.record_vote(int(vote_result), party_line_vote)
                        except Exception:
                            pass
                    except Exception as e:
                        logger.debug(f"Vote decision failed for actor {actor.id}: {e}. Using random fallback.")
                        vote_result = self._rand_random() < 0.5
                        if vote_result:
                            yea_votes_for_step += 1
                        voted_with_party = (vote_result == party_line_vote)
                        actor_votes.append((i, vote_result))
                        try:
                            actor.record_vote(int(vote_result), party_line_vote)
                        except Exception:
                            pass
                    # After recording the vote, update political capital and loyalty
                    actor.regenerate_political_capital(voted_with_party)
                
                # Step 4.5: REWARD COMPUTATION AND DQN LEARNING
                rewards_start = time.perf_counter()
                # Compute rewards for each actor based on their vote
                for i, (actor_idx, vote_action) in enumerate(actor_votes):
                    actor = sim.actors[actor_idx]
                    
                    if actor.use_dqn and actor.dqn_agent is not None:
                        try:
                            # Determine party line from Congress heuristic
                            party_line = party_line_cache.get(actor.party)
                            if party_line is None:
                                party_line = sim.get_party_line(actor.party, law)
                            
                            # Compute composite reward
                            reward, reward_breakdown = RewardComputer.compute_reward(
                                vote_action=int(vote_action),
                                party_line_vote=party_line,
                                actor_ideology=actor.ideology,
                                law_salience=law.salience,
                                law_threshold=law.threshold,
                                district_preference=self._rand_uniform(-1, 1),  # Simplified
                                loyalty_weight=config.REWARD_LOYALTY_WEIGHT,
                                ipm_weight=config.REWARD_IPM_WEIGHT,
                                electoral_weight=config.REWARD_ELECTORAL_WEIGHT,
                            )

                            # Store reward for later learning
                            actor.last_reward = reward
                            actor.last_reward_breakdown = reward_breakdown
                            
                            # Build next state (after ideology evolution)
                            next_state = actor._get_dqn_state_vector(law, influence_vector[actor_idx], current_pressure)
                            
                            # Record reward and perform DQN learning step
                            actor.record_dqn_reward(next_state, done=False)
                            
                        except Exception as e:
                            logger.debug(f"Reward computation failed for actor {actor.id}: {e}")
                rewards_end = time.perf_counter()
                comp_times['rewards'] += (rewards_end - rewards_start)
                
                # Step 5: Record outcome for this step
                acceptance_prob = yea_votes_for_step / total_actors
                trajectories[sim_idx, t] = acceptance_prob

                # Per-step progress callback (detailed)
                if progress_callback is not None and progress_per_step:
                    try:
                        progress_callback({
                            "phase": "step_complete",
                            "sim_index": int(sim_idx) + 1,
                            "step_index": int(t) + 1,
                            "n_steps": steps,
                            "acceptance_prob": float(acceptance_prob),
                        })
                    except Exception as e:
                        logger.debug(f"Progress callback (step_complete) failed: {e}")

                # KLUCZ: Wzmocnienie Reżimu (Sprzężenie Zwrotne)
                bill_passed = acceptance_prob > 0.5
                bill_importance = 3.0 if (t % 5 == 0) else 1.0
                try:
                    # Legacy update (rule-based feedback)
                    sim.regime_engine.update_regime_state(bill_passed, bill_importance)
                except Exception:
                    pass

                vote_probs = self.xp.array([a.vote_prob for a in sim.actors], dtype=self.xp.float32)
                vote_probs_cpu = to_cpu_array(vote_probs)
                party_bins = {"Democratic": [], "Republican": []}
                for actor in sim.actors:
                    try:
                        ideological_value = float(actor.ideology[0]) if isinstance(actor.ideology, np.ndarray) else float(actor.ideology)
                    except Exception:
                        ideological_value = 0.0
                    party = getattr(actor, 'party', None)
                    if party in party_bins:
                        party_bins[party].append(ideological_value)

                std_vote_prob = float(np.nanstd(vote_probs_cpu)) if vote_probs_cpu.size else 0.0
                std_dem = float(np.std(party_bins["Democratic"])) if party_bins["Democratic"] else 0.0
                std_rep = float(np.std(party_bins["Republican"])) if party_bins["Republican"] else 0.0
                gap = 0.0
                if party_bins["Democratic"] and party_bins["Republican"]:
                    gap = float(abs(np.mean(party_bins["Democratic"]) - np.mean(party_bins["Republican"])))

                community_cohesion = float(sim.analyze_network_cohesion())

                obs_vec = np.array([
                    acceptance_prob,
                    std_vote_prob,
                    std_dem,
                    std_rep,
                    gap,
                    community_cohesion
                ], dtype=float)

                prev_state = sim.regime_engine.scenario
                if self.use_hmm_state:
                    try:
                        s_h0 = time.perf_counter()
                        sim.regime_engine.update_hmm_with_observation(obs_vec)
                        s_h1 = time.perf_counter()
                        comp_times['hmm'] += (s_h1 - s_h0)

                        predicted_state = sim.regime_engine.get_last_hmm_state()
                        predicted_component = sim.regime_engine.get_last_hmm_component()
                        if predicted_state and predicted_state != prev_state:
                            logger.debug(
                                f"HMM inferred regime state {predicted_state} (component {predicted_component}); "
                                f"switching from {prev_state}."
                            )
                        elif predicted_state:
                            logger.debug(
                                f"HMM reaffirms regime state {predicted_state} (component {predicted_component})."
                            )
                    except Exception:
                        pass

                try:
                    s_h0 = time.perf_counter()
                    sim.regime_engine.update_regime_state_with_observation(acceptance_prob)
                    s_h1 = time.perf_counter()
                    comp_times['hmm'] += (s_h1 - s_h0)
                except Exception:
                    pass

                # Step 6: Update influence weights based on simple reward proxy
                try:
                    # Build simple reward vector: actors who voted with majority get reward
                    rewards = [1.0 if (actor_votes[idx][1] == bill_passed) else 0.0 for idx in range(len(actor_votes))]
                    sim.update_influence_weights(rewards=rewards, learning_rate=0.005)
                except Exception:
                    pass

                # Recompute contextual adjacency after weight updates so evolution uses updated base
                try:
                    contextual_W = sim.get_contextual_influence_matrix(law, current_pressure)
                    sim.adj_matrix = contextual_W
                except Exception:
                    # Fallback to old adjacency if recompute fails
                    sim.adj_matrix = old_adj

                # Step 7: Advance system state (ideology evolution uses contextual adjacency)
                sim.step()

                # Restore base adjacency (which may have been modified by update_influence_weights)
                sim.adj_matrix = old_adj

                # Timing: report slow steps if threshold exceeded
                step_end = time.perf_counter()
                step_total = step_end - step_start_time
                threshold = getattr(config, "SLOW_STEP_THRESHOLD_SECONDS", 0.0)
                timings_report = {k: float(v) for k, v in comp_times.items()}
                timings_report['total'] = float(step_total)

                # Send progress event for slow step (above threshold)
                if step_total > float(threshold):
                    try:
                        if progress_callback is not None:
                            progress_callback({
                                "phase": "slow_step",
                                "sim_index": int(sim_idx) + 1,
                                "step_index": int(t) + 1,
                                "n_steps": steps,
                                "threshold": float(threshold),
                                "timings": timings_report,
                            })
                    except Exception as e:
                        logger.debug(f"Progress callback (slow_step) failed: {e}")

                # If requested, always emit per-step timing events (useful for small-step inspection)
                if show_small_steps and progress_callback is not None:
                    try:
                        progress_callback({
                            "phase": "step_timing",
                            "sim_index": int(sim_idx) + 1,
                            "step_index": int(t) + 1,
                            "n_steps": steps,
                            "timings": timings_report,
                        })
                    except Exception as e:
                        logger.debug(f"Progress callback (step_timing) failed: {e}")
            
            # Final vote count: use votes from last step
            vote_distributions[sim_idx] = int(yea_votes_for_step)

            if yea_votes_for_step > half_actors:
                pass_count += 1

            if abs(yea_votes_for_step - half_actors) < 2:
                critical_failures += 1

            # Per-simulation progress callback
            if progress_callback is not None:
                try:
                    progress_callback({
                        "phase": "simulation_complete",
                        "sim_index": int(sim_idx) + 1,
                        "n_simulations": n_simulations,
                        "pass_count_so_far": pass_count,
                        "last_yea": int(yea_votes_for_step),
                    })
                except Exception as e:
                    logger.debug(f"Progress callback (simulation_complete) failed: {e}")

        # Transfer results back to CPU if using GPU
        if self.use_gpu:
            vote_distributions = to_cpu_array(vote_distributions)
            trajectories = to_cpu_array(trajectories)

        results: Dict[str, Any] = {
            "pass_count": pass_count,
            "vote_distributions": vote_distributions.tolist(),
            "critical_failures": critical_failures,
            "trajectories": trajectories.tolist(),
        }

        compiled = self._compile_report(results, n_simulations)

        if progress_callback is not None:
            try:
                progress_callback({"phase": "complete", "report": compiled})
            except Exception as e:
                logger.debug(f"Progress callback (complete) failed: {e}")

        return compiled

    def _compile_report(self, results: Dict[str, Any], n: int) -> Dict[str, Any]:
        """Compile simulation results into an executive report using vectorized operations."""
        # Convert to GPU arrays for fast computation if needed
        vote_dist = self.xp.array(results["vote_distributions"], dtype=self.xp.float32)
        
        prob = results["pass_count"] / max(1, n)
        avg_votes = float(self.xp.mean(vote_dist))
        std_dev = float(self.xp.std(vote_dist))

        return {
            "probability_of_passing": prob,
            "expected_votes": avg_votes,
            "vote_volatility": std_dev,
            "risk_of_flip": results["critical_failures"] / max(1, n),
            "regime": self.template.regime_engine.scenario,
            "trajectories": results["trajectories"],
            "distributions": results["vote_distributions"],
        }

    def run_sensitivity_analysis(
        self,
        param_name: str,
        param_values: List[float],
        n_simulations: int = 100,
        steps: int = 5,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        show_tqdm: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """Analyze sensitivity of outcomes to parameter variations using vectorized processing.

        Args:
            progress_callback: Optional callback that will receive progress updates for each parameter value.
            show_tqdm: If True and `tqdm` is available, show a progress bar across parameter values.
        
        Supports both actor-level and regime-level parameters:
        - Actor parameters: 'vulnerability_avg', 'volatility_avg', 'ideology_spread'
        - Regime parameters: 'base_pressure', 'volatility_multiplier'
        
        Args:
            param_name: Name of parameter to vary.
            param_values: List of parameter values to test.
            n_simulations: Number of simulations per parameter value.
            steps: Steps per simulation.
            
        Returns:
            Dictionary mapping parameter values to result metrics.
        """
        results: Dict[str, Dict[str, float]] = {}
        original_actors = deepcopy(self.template.actors)
        original_regime = self.template.regime_engine
        
        # Pre-convert param_values to GPU if using GPU
        param_values_gpu = self.xp.array(param_values, dtype=self.xp.float32)
        n_params = len(param_values)

        logger.info(f"Running sensitivity analysis for '{param_name}' across {n_params} values (GPU: {self.use_gpu})...")

        iterator = enumerate(param_values)
        if show_tqdm and _HAS_TQDM:
            iterator = enumerate(tqdm(param_values, desc=f"Sensitivity:{param_name}", unit="val"))

        for idx, value in iterator:
            # Actor-level parameters
            if param_name == "vulnerability_avg":
                for actor in self.template.actors:
                    actor.vulnerability = float(value)
            elif param_name == "volatility_avg":
                for actor in self.template.actors:
                    actor.volatility = float(value)
            elif param_name == "ideology_spread":
                for actor in self.template.actors:
                    actor.ideology = actor.ideology * float(value)
            
            # Regime-level parameters
            elif param_name == "base_pressure":
                self.template.regime_engine = PublicRegime(scenario=original_regime.scenario)
                self.template.regime_engine.base_pressure = float(value)
            elif param_name == "volatility_multiplier":
                self.template.regime_engine = PublicRegime(scenario=original_regime.scenario)
                self.template.regime_engine.volatility_multiplier = float(value)

            # Run simulations
            report = self.run_monte_carlo(n_simulations, steps)
            results[str(value)] = {
                "probability_of_passing": float(report["probability_of_passing"]),
                "expected_votes": float(report["expected_votes"]),
                "vote_volatility": float(report["vote_volatility"]),
                "risk_of_flip": float(report["risk_of_flip"]),
            }
            
            logger.debug(f"Sensitivity {param_name}={value}: P(pass)={results[str(value)]['probability_of_passing']:.3f} ({idx+1}/{n_params})")

            if progress_callback is not None:
                try:
                    progress_callback({
                        "phase": "sensitivity_param_complete",
                        "param_name": param_name,
                        "param_value": value,
                        "index": idx + 1,
                        "n_values": n_params,
                        "metrics": results[str(value)],
                    })
                except Exception as e:
                    logger.debug(f"Progress callback (sensitivity_param_complete) failed: {e}")

        # Restore original state
        self.template.actors = original_actors
        self.template.regime_engine = original_regime
        
        return results

    def get_probability_distribution(self) -> Dict[str, float]:
        """Return distribution statistics over current template actors' vote probabilities using GPU."""
        probs = self.xp.array([a.vote_prob for a in self.template.actors], dtype=self.xp.float32)
        
        if probs.size == 0:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        return {
            "mean": float(self.xp.mean(probs)),
            "median": float(self.xp.median(probs)),
            "std": float(self.xp.std(probs)),
            "min": float(self.xp.min(probs)),
            "max": float(self.xp.max(probs)),
            "percentile_25": float(self.xp.percentile(probs, 25)),
            "percentile_75": float(self.xp.percentile(probs, 75)),
        }

    def visualize_trajectories(self, report: Dict[str, Any], output_path: str = "./results/trajectories.png",
                              dpi: int = 150) -> None:
        """Visualize Monte Carlo trajectories using GPU-accelerated statistics.
        
        Creates a plot showing:
        - Average law acceptance probability over time (ensemble mean)
        - Confidence bands (±1 std) showing prediction uncertainty
        - Individual trajectories (optional, for smaller ensembles)
        
        This helps identify whether the system reaches equilibrium, exhibits
        periodic cycles, or exhibits chaotic behavior.
        
        Args:
            report: Report dictionary from run_monte_carlo.
            output_path: File path to save the visualization.
            dpi: Resolution of output image.
        """
        if not _HAS_MATPLOTLIB:
            logger.warning("Trajectory visualization requires matplotlib. Skipping visualization.")
            return

        trajectories = report.get("trajectories", [])
        if not trajectories:
            logger.warning("No trajectories found in report. Skipping visualization.")
            return

        # Convert to GPU array for fast computation if using GPU
        traj_gpu = self.xp.array(trajectories, dtype=self.xp.float32)
        n_simulations = traj_gpu.shape[0]
        n_steps = traj_gpu.shape[1]

        # Compute statistics on GPU
        mean_traj = self.xp.mean(traj_gpu, axis=0)
        std_traj = self.xp.std(traj_gpu, axis=0)
        min_traj = self.xp.min(traj_gpu, axis=0)
        max_traj = self.xp.max(traj_gpu, axis=0)

        # Transfer back to CPU for plotting
        if self.use_gpu:
            mean_traj = to_cpu_array(mean_traj)
            std_traj = to_cpu_array(std_traj)
            min_traj = to_cpu_array(min_traj)
            max_traj = to_cpu_array(max_traj)

        # Prepare plot
        fig, ax = plt.subplots(figsize=(12, 7), dpi=dpi)
        steps_range = np.arange(n_steps)

        # Plot ensemble mean
        ax.plot(steps_range, mean_traj, 'b-', linewidth=2.5, label='Ensemble Mean (Law Acceptance Probability)', zorder=3)

        # Plot confidence bands (±1 std)
        ax.fill_between(steps_range, mean_traj - std_traj, mean_traj + std_traj, 
                        alpha=0.2, color='blue', label='±1 Std Dev')

        # Plot min/max envelope
        ax.fill_between(steps_range, min_traj, max_traj, alpha=0.1, color='gray', label='Min/Max Range')

        # Plot individual trajectories if ensemble is small (for clarity)
        if n_simulations <= 50:
            for i, traj in enumerate(trajectories):
                ax.plot(steps_range, traj, 'gray', alpha=0.15, linewidth=0.5)

        # Add convergence analysis
        if n_steps > 1:
            final_variance = np.var(mean_traj[-5:]) if n_steps > 5 else 0
            convergence_text = f"Convergence (Final 5-step variance): {final_variance:.6f}\n"
            if final_variance < 0.01:
                convergence_status = "✓ Converged"
            elif final_variance < 0.05:
                convergence_status = "~ Partially Converged"
            else:
                convergence_status = "✗ Chaotic/Oscillating"
            convergence_text += convergence_status
        else:
            convergence_text = "Insufficient steps for convergence analysis"

        # Add labels and formatting
        ax.set_xlabel('Simulation Steps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Law Acceptance Probability', fontsize=12, fontweight='bold')
        
        scenario = report.get("regime", "Unknown").upper()
        ax.set_title(f'Congressional Voting Dynamics Trajectories - {scenario} Scenario\n'
                    f'({n_simulations} simulations, {n_steps} steps) | GPU: {self.use_gpu}',
                    fontsize=13, fontweight='bold')

        # Add convergence info box
        ax.text(0.98, 0.05, convergence_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)

        plt.tight_layout()
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Trajectory visualization saved to {output_path}")
