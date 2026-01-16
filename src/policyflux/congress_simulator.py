"""High-level API for congressional simulations.

Example:
    >>> from policyflux import CongressSimulator
    >>> sim = CongressSimulator(scenario="polarized")
    >>> sim.fit()
    >>> results = sim.simulate(n_simulations=10, steps=5)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

import numpy as np

from policyflux.defaults import SIMULATION, MODELS

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    
    scenario: Literal["stable", "polarized", "crisis"] = "stable"
    n_simulations: int = 100
    steps: int = 10
    seed: int = 42
    use_cache: bool = True
    
    # Network parameters
    cosponsorship_alpha: float = 0.5
    homophily_beta: float = 2.0
    leader_boost: float = 2.0


class CongressSimulator:
    """Main simulator for congressional voting behavior.
    
    Follows a Keras-like compile/fit/simulate workflow:
    - compile(): configure hyperparameters
    - fit(): load data and train models
    - simulate(): run Monte Carlo simulations
    """
    
    def __init__(
        self,
        scenario: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = SimulationConfig(
            scenario=scenario or SIMULATION.SCENARIO,
            seed=seed or SIMULATION.RNG_SEED,
        )
        self.compiled = False
        self.fitted = False
        
        # Will be set during fit()
        self.actors: List[Dict] = []
        self.influence_matrix: Optional[np.ndarray] = None
        self._builder = None
    
    def compile(
        self,
        scenario: Optional[str] = None,
        n_simulations: Optional[int] = None,
        steps: Optional[int] = None,
        **kwargs
    ) -> "CongressSimulator":
        """Configure simulation parameters."""
        if scenario:
            self.config.scenario = scenario
        if n_simulations:
            self.config.n_simulations = n_simulations
        if steps:
            self.config.steps = steps
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.compiled = True
        logger.info(f"Compiled with scenario={self.config.scenario}")
        return self
    
    def fit(self, use_cache: bool = True) -> "CongressSimulator":
        """Load congressional data and train models.
        
        This step:
        1. Fetches voting data from VoteView (or cache)
        2. Trains ideal point model
        3. Computes behavioral features (loyalty, vulnerability)
        4. Builds influence network
        """
        if not self.compiled:
            self.compile()
        
        # Lazy import to avoid TensorFlow at module load
        from policyflux.data import CongressBuilder
        
        logger.info("Building Congress data...")
        self._builder = CongressBuilder(use_cache=use_cache)
        
        # Export actors with trained models
        actors_data, ideal_point_model, ipm_params = self._builder.export_actors_with_model()
        self.actors = actors_data
        self.influence_matrix = getattr(self._builder, "influence_matrix", None)
        
        self.fitted = True
        logger.info(f"Fitted with {len(self.actors)} actors")
        return self
    
    def simulate(
        self,
        n_simulations: Optional[int] = None,
        steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation.
        
        Returns:
            Dictionary with simulation results including:
            - vote_outcomes: voting results per step
            - ideology_trajectories: how ideologies evolve
            - aggregate_stats: summary statistics
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() before simulate()")
        
        n_sims = n_simulations or self.config.n_simulations
        n_steps = steps or self.config.steps
        
        logger.info(f"Running {n_sims} simulations for {n_steps} steps...")
        
        # Simple simulation loop (placeholder for full engine)
        rng = np.random.default_rng(self.config.seed)
        
        results = {
            "n_simulations": n_sims,
            "n_steps": n_steps,
            "scenario": self.config.scenario,
            "n_actors": len(self.actors),
            "vote_outcomes": [],
            "ideology_mean": [],
            "polarization": [],
        }
        
        # Extract ideologies
        ideologies = np.array([a.get("ideology", 0.0) for a in self.actors])
        
        for sim_idx in range(n_sims):
            sim_ideologies = ideologies.copy()
            sim_votes = []
            
            for step in range(n_steps):
                # Simple vote simulation based on ideology
                bill_position = rng.uniform(-1, 1)
                votes = (sim_ideologies * bill_position + rng.normal(0, 0.1, len(sim_ideologies))) > 0
                sim_votes.append(votes.mean())
                
                # Small ideology drift
                sim_ideologies += rng.normal(0, 0.01, len(sim_ideologies))
            
            results["vote_outcomes"].append(sim_votes)
            results["ideology_mean"].append(float(sim_ideologies.mean()))
            results["polarization"].append(float(np.std(sim_ideologies)))
        
        # Aggregate statistics
        results["summary"] = {
            "mean_vote_rate": float(np.mean(results["vote_outcomes"])),
            "mean_polarization": float(np.mean(results["polarization"])),
            "ideology_drift": float(np.std(results["ideology_mean"])),
        }
        
        logger.info(f"Simulation complete. Mean vote rate: {results['summary']['mean_vote_rate']:.3f}")
        return results
    
    def get_actors(self) -> List[Dict]:
        """Return actor data after fitting."""
        if not self.fitted:
            raise RuntimeError("Must call fit() first")
        return self.actors
    
    def get_influence_matrix(self) -> Optional[np.ndarray]:
        """Return influence/adjacency matrix after fitting."""
        if not self.fitted:
            raise RuntimeError("Must call fit() first")
        return self.influence_matrix




