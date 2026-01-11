import numpy as np
import os
from collections import Counter
from typing import List, Optional, TYPE_CHECKING
import logging

from policyflux.congress.actors import CongressMan
from policyflux import config
from policyflux.gpu_utils import get_array_module, to_cpu_array

if TYPE_CHECKING:
    from policyflux.congress.law import Law

# optional networkx for centrality; fallback to degree-based proxy
try:
    import networkx as nx
    _HAS_NETWORKX = True
except Exception:
    nx = None
    _HAS_NETWORKX = False

# optional matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False
class TheCongress:
    def __init__(self, actors: List[CongressMan], adj_matrix: np.ndarray, regime_engine,
                 cosponsorship: Optional[np.ndarray] = None,
                 committee_matrix: Optional[np.ndarray] = None,
                 cospon_alpha: float = 0.5,
                 homophily_beta: float = 2.0,
                 leader_boost: float = 2.0,
                 use_gpu: bool = False):
        """
        System C(t) = {G(t), X(t), Z(t)}
        """
        """Congressional system with network dynamics.

        Args:
            actors: List of CongressMan objects.
            adj_matrix: Adjacency matrix representing influence network.
            regime_engine: Global context/regime configuration.
            cosponsorship: Optional cosponsorship matrix.
            cospon_alpha: Weight for cosponsorship blending [0, 1].
            homophily_beta: Strength of homophily effects.
            leader_boost: Amplification of leader influence.
        """
        self.actors: List[CongressMan] = actors
        self.use_gpu = bool(use_gpu)
        self.xp = get_array_module(self.use_gpu)
        xp = self.xp
        # Przygotuj podstawową macierz wpływów
        adj_init = xp.asarray(adj_matrix, dtype=xp.float32)
        self.regime_engine = regime_engine # Z(t) - Kontekst
        
        # Metryki systemowe (do raportowania)
        self.history = {
            "polarization": [],
            "instability_index": []
        }
        logger = logging.getLogger(__name__)

        # ---- Enhance adjacency with cosponsorship, homophily and leader asymmetry ----
        n = len(self.actors)

        # Validate adjacency matrix shape early to give a clear error message
        if adj_init.shape != (n, n):
            raise ValueError(
                f"Adjacency matrix shape {adj_init.shape} is incompatible with number of actors ({n}). "
                "Ensure the adjacency matrix rows/columns correspond to the provided actors in the same order."
            )

        # Precompute committee memberships for dynamic boosts
        self.actor_committees = [set(getattr(a, "committee_memberships", []) or []) for a in self.actors]

        self.cosponsorship_matrix = self._normalize_matrix(cosponsorship, n)
        self.committee_matrix = self._normalize_matrix(committee_matrix, n)

        C = xp.zeros((n, n), dtype=xp.float32)
        # Try to honor cosponsorship when possible; handle DataFrame alignment by actor id
        if cosponsorship is not None:
            try:
                import pandas as pd
                if isinstance(cosponsorship, pd.DataFrame):
                    actor_ids = [a.id for a in self.actors]
                    df = cosponsorship.reindex(index=actor_ids, columns=actor_ids, fill_value=0)
                    C = xp.asarray(df.values, dtype=xp.float32)
                    self.cosponsorship_matrix = df.values
                elif self.cosponsorship_matrix is not None and self.cosponsorship_matrix.shape == (n, n):
                    C = xp.asarray(self.cosponsorship_matrix, dtype=xp.float32)
                else:
                    logger.warning(
                        "Ignoring cosponsorship matrix with shape %s that does not match actor count %s",
                        getattr(cosponsorship, "shape", None), n
                    )
            except Exception:
                if self.cosponsorship_matrix is not None and self.cosponsorship_matrix.shape == (n, n):
                    C = xp.asarray(self.cosponsorship_matrix, dtype=xp.float32)
                else:
                    logger.warning("Unable to align cosponsorship matrix; proceeding without it.")

        # 2) Blend cosponsorship and provided adjacency
        G = (cospon_alpha * C) + ((1 - cospon_alpha) * adj_init)
        if self.committee_matrix is not None:
            G = G + (config.COMMITTEE_WEIGHT * xp.asarray(self.committee_matrix, dtype=xp.float32))

        # 3) Infer party if missing and build ideology vector (support multi-dimensional)
        # For homophily, use dimension 1 (economic/social) if multi-dimensional
        ide = []
        parties = []
        for a in self.actors:
            if isinstance(a.ideology, np.ndarray) and len(a.ideology) > 0:
                ide_dim1 = float(a.ideology[0])
            else:
                ide_dim1 = float(a.ideology) if isinstance(a.ideology, (int, float)) else 0.0

            ide.append(ide_dim1)

            if getattr(a, 'party', None) is None:
                a.party = 'A' if ide_dim1 < 0 else 'B'
            parties.append(a.party)

        ide = xp.asarray(ide, dtype=xp.float32)

        # 4) Homophily matrix (silniejsze wpływy dla podobnej ideologii / tej samej partii)
        dist = xp.abs(ide.reshape(-1,1) - ide.reshape(1,-1))
        H = xp.exp(-homophily_beta * dist)
        party_arr = xp.asarray([1 if p is not None else 0 for p in parties], dtype=xp.int32)
        party_sign = xp.where(party_arr == 0, -1, party_arr)
        P = xp.where(party_sign.reshape(-1,1) == party_sign.reshape(1,-1), 1.2, 1.0)

        # 5) Apply homophily & party weighting
        G = G * H * P

        G_cpu = to_cpu_array(G)

        # 6) Leader asymmetry: compute centrality via the configured method
        try:
            if _HAS_NETWORKX:
                G_nx = nx.from_numpy_array(G_cpu, create_using=nx.Graph)
                if config.LEADER_CENTRALITY_METHOD == "betweenness":
                    centrality_dict = nx.betweenness_centrality(G_nx, normalized=True)
                else:
                    centrality_dict = nx.degree_centrality(G_nx)
                centrality = np.array([centrality_dict.get(i, 0.0) for i in range(n)], dtype=float)
            else:
                # fallback: use row-sum as proxy centrality
                centrality = np.sum(np.abs(G_cpu), axis=1)
        except Exception:
            centrality = np.sum(np.abs(G_cpu), axis=1)

        # normalize centrality to [0,1] with guard against zero range
        range_span = centrality.max() - centrality.min()
        if range_span > 0:
            centrality = (centrality - centrality.min()) / range_span
        else:
            centrality = np.zeros_like(centrality)

        for actor, cent in zip(self.actors, centrality):
            actor.set_centrality(float(cent))

        multipliers = 1.0 + leader_boost * centrality
        # scale outgoing rows
        G_cpu = (G_cpu.T * multipliers).T

        # 7) remove self-loops
        np.fill_diagonal(G_cpu, 0.0)

        # store enhanced adjacency
        self.adj_matrix = G_cpu
        self.base_adj_matrix = G_cpu.copy()
        self.current_adj_matrix = self.base_adj_matrix.copy()

        # Community detection cache
        self.community_labels = self.detect_communities()

    def detect_communities(self, method: str = "louvain") -> np.ndarray:
        """Detect communities (fragments) using the enriched influence graph."""
        base = np.abs(self.adj_matrix)
        combined = base.copy()
        if self.cosponsorship_matrix is not None:
            combined = combined + 0.6 * np.abs(self.cosponsorship_matrix)
        if self.committee_matrix is not None:
            combined = combined + 0.8 * np.abs(self.committee_matrix)

        combined = np.nan_to_num(combined, nan=0.0)

        try:
            import networkx as nx
            G_nx = nx.from_numpy_array(combined, create_using=nx.Graph)
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G_nx)
                labels = np.array([partition.get(i, 0) for i in range(len(self.actors))], dtype=int)
            except Exception:
                communities = list(nx.community.greedy_modularity_communities(G_nx))
                labels = np.zeros(len(self.actors), dtype=int)
                for idx, comm in enumerate(communities):
                    for node in comm:
                        labels[node] = idx
            self.community_labels = labels
            self._assign_actor_communities(labels)
            return labels
        except Exception:
            labels = np.zeros(len(self.actors), dtype=int)
            self.community_labels = labels
            self._assign_actor_communities(labels)
            return labels

    def _assign_actor_communities(self, labels: np.ndarray) -> None:
        """Assign detected community IDs back to the actor objects."""
        for actor, label in zip(self.actors, labels):
            actor.community_id = int(label)

    def _get_leader_indices(self, top_k: int = 10) -> List[int]:
        """Identify leader-like actors using centrality, roles, and committee power."""
        scores = []
        for idx, actor in enumerate(self.actors):
            score = float(getattr(actor, "centrality", 0.0))
            if getattr(actor, "is_leader", False):
                score += 0.2
            role = getattr(actor, "role", "") or ""
            if isinstance(role, str) and "leader" in role.lower():
                score += 0.3
            score += 0.05 * float(getattr(actor, "committee_power", 0.0))
            scores.append(score)

        if not scores:
            return []

        scores_arr = np.array(scores, dtype=float)
        top_indices = np.argsort(-scores_arr)[:top_k]
        return [int(i) for i in top_indices if scores_arr[int(i)] > 0]

    def _build_committee_mask(self, law: Optional["Law"]) -> Optional[np.ndarray]:
        """Create mask boosting ties among members of committees relevant to a law."""
        if law is None:
            return None
        law_committees = getattr(law, "committee_ids", None)
        if not law_committees:
            return None
        law_committees = set(law_committees)
        n = len(self.actors)
        xp = get_array_module(getattr(self, "use_gpu", False))
        mask = xp.zeros((n, n), dtype=xp.float32)
        for i, comm_i in enumerate(self.actor_committees):
            if not law_committees.intersection(comm_i):
                continue
            for j, comm_j in enumerate(self.actor_committees):
                if law_committees.intersection(comm_j):
                    mask[i, j] = 1.0
        return to_cpu_array(mask) if xp.any(mask) else None

    def _build_dynamic_adj_matrix(self, law: Optional["Law"] = None) -> np.ndarray:
        """Construct G(t) with committee boosts, leader scaling, and regime homophily."""
        xp = get_array_module(getattr(self, 'use_gpu', False))
        adj = xp.asarray(self.base_adj_matrix, dtype=xp.float32).copy()

        # Homophily modulation from regime
        homophily_scale = 1.0
        try:
            if hasattr(self.regime_engine, "get_homophily_scale"):
                homophily_scale = float(self.regime_engine.get_homophily_scale())
        except Exception:
            homophily_scale = 1.0
        adj *= homophily_scale

        # Committee-specific boost for relevant bills
        committee_mask = self._build_committee_mask(law)
        if committee_mask is not None:
            alpha = getattr(config, "COMMITTEE_BOOST_ALPHA", 0.6)
            saliency_factor = 1.0
            try:
                saliency_val = getattr(law, "saliency_score", None)
                if saliency_val is not None:
                    saliency_factor += 0.5 * float(saliency_val)
            except Exception:
                pass
            adj = adj + alpha * saliency_factor * xp.asarray(committee_mask, dtype=xp.float32)

        # Leader/out-degree emphasis using dynamic leader set
        leader_indices = self._get_leader_indices()
        leader_boost = getattr(config, "LEADER_DYNAMIC_BOOST", 0.4)
        for idx in leader_indices:
            cent = float(getattr(self.actors[idx], "centrality", 0.0))
            committee_power = float(getattr(self.actors[idx], "committee_power", 0.0))
            multiplier = 1.0 + leader_boost + 0.5 * cent + 0.05 * committee_power
            adj[idx, :] = adj[idx, :] * multiplier

        xp.fill_diagonal(adj, 0.0)
        return to_cpu_array(xp.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0))

    def update_influence_weights(self, rewards: List[float] = None, learning_rate: float = 0.01) -> None:
        """Update adjacency/influence weights based on simple reciprocity/reward.

        - If rewards provided, increase W_{i,j} when j's actions benefited i in the past.
        - Otherwise, reinforce intra-community ties slightly.
        """
        n = len(self.actors)
        if rewards is not None and len(rewards) == n:
            # Simple rule: actors who received higher reward are more influential
            reward_vec = np.array(rewards, dtype=float)
            if reward_vec.max() > 0:
                reward_norm = reward_vec / (reward_vec.max())
            else:
                reward_norm = reward_vec
            for i in range(n):
                for j in range(n):
                    # increase outgoing weight from j proportional to reward of j
                    self.adj_matrix[i, j] = self.adj_matrix[i, j] * (1.0 + learning_rate * reward_norm[j])
        else:
            # Reinforce intra-community ties
            labels = self.community_labels if self.community_labels is not None else self.detect_communities()
            for i in range(n):
                for j in range(n):
                    if labels[i] == labels[j]:
                        self.adj_matrix[i, j] = self.adj_matrix[i, j] * (1.0 + learning_rate)

        # Keep diagonal zero and clip
        np.fill_diagonal(self.adj_matrix, 0.0)
        self.adj_matrix = np.nan_to_num(self.adj_matrix)

        # Refresh baselines for future dynamic adjustments
        self.base_adj_matrix = self.adj_matrix.copy()
        self.current_adj_matrix = self.base_adj_matrix.copy()

        self.community_labels = self.detect_communities()

    def step(self, law: Optional['Law'] = None):
        """
        One step t -> t+1 in dynamic simulation.
        
        Implements:
        1. Network pressure computation: pressure_i(t) = sum_j(adj_matrix[i,j] * opinion_j(t))
        2. Global regime pressure: Z(t) from PublicRegime
        3. Actor decision probability based on ideology, network, and regime
        4. Ideology evolution: x_i(t+1) = x_i(t) + Δx_i where Δx incorporates:
           - Network influences (adj_matrix weighted ideology diffusion)
           - Global pressure (Z(t))
           - Actor vulnerability/loyalty
        """
        # 1. Get current global context pressure Z(t)
        context_pressure = self.regime_engine.get_current_pressure()
        self.regime_engine.step()  # Advance regime time

        # Build dynamic influence matrix for this step/law
        adj_dynamic = self._build_dynamic_adj_matrix(law)
        self.current_adj_matrix = adj_dynamic
        
        # 2. Extract current actor ideologies (multi-dimensional)
        # Shape: (n_actors, dim) or (n_actors,) if 1D
        current_ideologies = np.array([a.ideology for a in self.actors], dtype=float)
        
        # 3. Compute network pressure for each actor
        # If ideologies are multi-dimensional, use mean for network pressure
        if current_ideologies.ndim == 2:
            ideology_mean = np.mean(current_ideologies, axis=1)  # Shape: (n_actors,)
        else:
            ideology_mean = current_ideologies  # Already 1D
        
        # Network influence: how much network neighbors affect opinion
        xp = get_array_module(getattr(self, "use_gpu", False))
        adj_xp = xp.asarray(adj_dynamic, dtype=xp.float32)
        ide_mean_xp = xp.asarray(ideology_mean, dtype=xp.float32)
        network_pressures = adj_xp.dot(ide_mean_xp)
        network_pressures = xp.tanh(network_pressures)
        network_pressures = to_cpu_array(network_pressures)  # Keep downstream on CPU for actor state updates
        
        # 4. Update actors' opinions and decision probabilities
        votes_probs = []
        ideology_changes = []  # Track ideology deltas
        
        for i, actor in enumerate(self.actors):
            # Compute voting decision probability
            prob = actor.calculate_decision_probability(
                network_pressure=network_pressures[i], 
                global_context=context_pressure
            )
            votes_probs.append(prob)

            # 5. Compute ideology evolution Δx_i(t)
            if current_ideologies.ndim == 2:
                actor_ideology = actor.ideology

                network_shift = 0.05 * np.sum(
                    [adj_dynamic[i, j] * self.actors[j].ideology 
                     for j in range(len(self.actors))],
                    axis=0
                )

                pressure_shift = 0.03 * context_pressure * actor.ideology
                vulnerability_factor = actor.vulnerability
                loyalty_factor = 1.0 - actor.loyalty
                noise = np.random.normal(0, actor.volatility, size=actor.ideology.shape)

                delta_ideology = (
                    vulnerability_factor * network_shift +
                    (context_pressure * loyalty_factor) * pressure_shift +
                    noise
                )

                new_ideology = actor.ideology + delta_ideology
                actor.ideology = np.tanh(new_ideology)
                ideology_changes.append(delta_ideology)
            else:
                network_shift = 0.05 * network_pressures[i]
                pressure_shift = 0.03 * context_pressure * ideology_mean[i]
                noise = np.random.normal(0, actor.volatility)

                delta_ideology = (
                    actor.vulnerability * network_shift +
                    (context_pressure * (1.0 - actor.loyalty)) * pressure_shift +
                    noise
                )

                new_ideology = ideology_mean[i] + delta_ideology
                actor.ideology = np.clip(new_ideology, -1.0, 1.0)
                actor.current_opinion = actor.get_ideological_position()
        
        # 6. Calculate system-level metrics
        self._calculate_system_metrics(votes_probs)
        
        return np.mean(votes_probs)

    def _calculate_system_metrics(self, probs):
        """
        Oblicza wskaźniki stabilności dla dashboardu.
        """
        probs = np.array(probs)
        
        # Polaryzacja: odchylenie standardowe prawdopodobieństw
        # (jeśli wszyscy mają 0.5 -> mała polaryzacja, jeśli 0.0 i 1.0 -> duża)
        polarization = np.std(probs)
        
        # Niestabilność (Instability): miara entropii lub wariancji zmian
        instability = polarization * self.regime_engine.volatility_multiplier
        
        self.history['polarization'].append(polarization)
        self.history['instability_index'].append(instability)

    def get_critical_actors(self):
        """
        Zwraca aktorów o największej 'betweenness' lub wpływie (dla analizy dźwigni).
        W MVP uproszczone: zwraca tych z największą sumą wag wyjściowych w macierzy.
        """
        influence_scores = np.sum(np.abs(self.adj_matrix), axis=1)
        top_indices = np.argsort(-influence_scores)[:5]
        return [self.actors[i] for i in top_indices]

    def get_faction_breakdown(self, threshold=0.5):
        """Dzieli aktorów na frakcje na podstawie ideologii (wymiar 1: ekonomiczny/społeczny)"""
        left = [a for a in self.actors if a.ideology[0] < -threshold]
        center = [a for a in self.actors if -threshold <= a.ideology[0] <= threshold]
        right = [a for a in self.actors if a.ideology[0] > threshold]
        return {'left': left, 'center': center, 'right': right}

    def analyze_network_cohesion(self):
        """Analizuje spójność sieci wśród aktorów"""
        opinions = np.array([a.current_opinion for a in self.actors])
        # średnia korelacja między sąsiadami
        connected_pairs = np.where(self.adj_matrix > 0)
        if len(connected_pairs[0]) == 0:
            return 0.0
        
        correlations = []
        for i, j in zip(connected_pairs[0], connected_pairs[1]):
            correlation = 1 - abs(opinions[i] - opinions[j])
            correlations.append(correlation)
        return np.mean(correlations) if correlations else 0.0

    def get_swing_voters(self, threshold=0.1):
        """Zwraca aktorów z najbardziej niestabilnym głosem (vote_prob blisko 0.5)"""
        swing = [a for a in self.actors if abs(a.vote_prob - 0.5) < threshold]
        return sorted(swing, key=lambda a: abs(a.vote_prob - 0.5))

    def get_majority_party(self) -> Optional[str]:
        """Return the currently largest party by headcount."""
        parties = [actor.party for actor in self.actors if actor.party]
        if not parties:
            return None
        return Counter(parties).most_common(1)[0][0]

    def get_majority_leader(self, party: Optional[str] = None) -> Optional[CongressMan]:
        """Return the actor with the most political capital in the majority party."""
        target_party = party or self.get_majority_party()
        if not target_party:
            return None
        members = [actor for actor in self.actors if actor.party == target_party]
        if not members:
            return None
        return max(members, key=lambda actor: actor.political_capital)

    def get_party_line(self, party: Optional[str], law: 'Law') -> int:
        """Estimate the party's recommended vote for the current law."""
        if party is None:
            return 1
        members = [actor for actor in self.actors if actor.party == party]
        if not members:
            return 1

        try:
            salience_vec = np.asarray(getattr(law, 'salience', [1.0]), dtype=float)
            salience_vec = salience_vec.flatten() if salience_vec.size > 0 else np.array([1.0])
        except Exception:
            salience_vec = np.array([1.0], dtype=float)

        alignments = []
        for actor in members:
            ideology_vec = np.asarray(actor.ideology, dtype=float).flatten()
            if ideology_vec.size == 0:
                alignment = float(actor.current_opinion)
            else:
                if ideology_vec.size != salience_vec.size:
                    if ideology_vec.size < salience_vec.size:
                        padded = np.pad(ideology_vec, (0, salience_vec.size - ideology_vec.size))
                    else:
                        padded = ideology_vec[: salience_vec.size]
                else:
                    padded = ideology_vec
                alignment = float(np.dot(padded, salience_vec[: padded.size]))
            alignments.append(alignment)

        avg_alignment = float(np.mean(alignments)) if alignments else 0.0
        return 1 if avg_alignment >= 0 else 0

    def get_network_influence_vector(self) -> np.ndarray:
        """Calculate network influence effects on each legislator (GPU-aware).

        If the congress instance has a GPU array module attached (via `self.use_gpu` and
        `self.xp`), computations are performed with that array module (CuPy) and then
        transferred back to CPU NumPy arrays before returning.
        """
        xp = get_array_module(getattr(self, 'use_gpu', False))

        num_actors = len(self.actors)
        ideologies = xp.zeros(num_actors, dtype=xp.float32)

        for i, actor in enumerate(self.actors):
            if isinstance(actor.ideology, np.ndarray) and len(actor.ideology) > 0:
                ideologies[i] = float(actor.ideology[0])
            else:
                ideologies[i] = float(actor.ideology) if hasattr(actor, 'ideology') else 0.0

        # Normalize ideologies to [-1, 1]
        ideology_std = float(xp.std(ideologies))
        if ideology_std > 1e-6:
            ideologies_normalized = (ideologies - xp.mean(ideologies)) / ideology_std
            ideologies_normalized = xp.tanh(ideologies_normalized)
        else:
            ideologies_normalized = xp.zeros_like(ideologies)

        # Ensure adjacency is in the same array module
        adj_source = getattr(self, "current_adj_matrix", self.adj_matrix)
        adj_xp = xp.asarray(adj_source)
        influence_vector = adj_xp.dot(ideologies_normalized)

        # Normalize influence to reasonable range
        influence_std = float(xp.std(influence_vector))
        if influence_std > 1e-6:
            influence_normalized = influence_vector / (2.0 * influence_std)
        else:
            influence_normalized = influence_vector

        influence_normalized = xp.nan_to_num(influence_normalized, nan=0.0)
        influence_normalized = xp.clip(influence_normalized, -0.5, 0.5)

        # Community reinforcement uses the original adj matrix (small loop on CPU)
        labels = self.community_labels if self.community_labels is not None else np.zeros(num_actors, dtype=int)
        if labels.size == num_actors:
            labels_cpu = labels if isinstance(labels, np.ndarray) else to_cpu_array(labels)
            for i in range(num_actors):
                same_community = labels_cpu == labels_cpu[i]
                if np.count_nonzero(same_community) > 1:
                    boost = float(np.mean(np.abs(adj_source[i, same_community]))) * config.COMMUNITY_REINFORCEMENT
                    influence_normalized = xp.clip(influence_normalized + boost, -0.5, 0.5)

        # Return a CPU-backed numpy array for downstream code
        return to_cpu_array(influence_normalized)

    def _normalize_matrix(self, matrix: Optional[np.ndarray], size: int) -> Optional[np.ndarray]:
        """Ensure optional cosponsorship/committee matrices match the actor count."""
        if matrix is None:
            return None
        arr = np.array(matrix, dtype=float)
        if arr.shape != (size, size):
            return None
        return arr

    def _get_party_homophily_matrix(self) -> np.ndarray:
        """Return a matrix with 1.0 where actors share the same party, else 0.0."""
        xp = get_array_module(getattr(self, "use_gpu", False))
        parties = [getattr(a, "party", None) for a in self.actors]
        # Map parties to ints for fast equality; None -> -1
        party_ids = xp.asarray([hash(p) if p is not None else -1 for p in parties], dtype=xp.int64)
        matches = party_ids.reshape(-1, 1) == party_ids.reshape(1, -1)
        return to_cpu_array(matches.astype(xp.float32))

    def get_contextual_influence_matrix(self, law: 'Law', regime_pressure: float) -> np.ndarray:
        """Dynamically modify the influence matrix based on context (law and regime).

        This implementation performs heavy matrix math with the appropriate array
        module (NumPy or CuPy) depending on availability, and returns a CPU-backed
        NumPy array for compatibility.
        """
        xp = get_array_module(getattr(self, 'use_gpu', False))

        # 1. Base matrix
        W_base = xp.asarray(self.adj_matrix, dtype=xp.float32).copy()
        W_contextual = W_base.copy()

        # 2. Polarization boost (intra-party strengthening)
        party_matrix = xp.asarray(self._get_party_homophily_matrix(), dtype=xp.float32)
        polarization_boost = (float(regime_pressure) ** 2) * 2.0
        W_contextual = W_contextual + (party_matrix * W_base * polarization_boost)

        # 3. Thematic modulation: actors more aligned with law salience gain outgoing influence
        try:
            # Compute a simple expertise score per actor: abs(dot(ideology, law.salience))
            actor_expertise = xp.array([
                abs(float(np.dot(a.ideology.flatten(), law.salience.flatten()))) if hasattr(a, 'ideology') else 0.0
                for a in self.actors
            ], dtype=xp.float32)
            max_ex = float(xp.max(actor_expertise)) if actor_expertise.size else 0.0
            if max_ex > 0:
                actor_expertise = actor_expertise / max_ex
            else:
                actor_expertise = xp.zeros_like(actor_expertise)

            # Increase outgoing column j by factor depending on expertise
            for j in range(len(self.actors)):
                W_contextual[:, j] = W_contextual[:, j] * (1.0 + 0.5 * float(actor_expertise[j]))
        except Exception:
            # If anything goes wrong, skip thematic modulation
            pass

        # 4. Clip extremes for numerical stability and return CPU array
        W_contextual = xp.nan_to_num(W_contextual, nan=0.0)
        return to_cpu_array(W_contextual)

    def get_system_snapshot(self):
        """Zwraca pełny snapshot stanu systemu"""
        return {
            'timestamp': len(self.history['polarization']),
            'avg_opinion': np.mean([a.current_opinion for a in self.actors]),
            'avg_vote_prob': np.mean([a.vote_prob for a in self.actors]),
            'polarization': self.history['polarization'][-1] if self.history['polarization'] else 0,
            'instability': self.history['instability_index'][-1] if self.history['instability_index'] else 0,
            'cohesion': self.analyze_network_cohesion(),
            'swing_voters_count': len(self.get_swing_voters())
        }

    def render_influence_network(self, output_path: str = "./results/influence_network.png", 
                                dpi: int = 150) -> None:
        """Visualize the influence network G(t) using networkx and matplotlib.
        
        Creates a network graph showing:
        - Nodes: Congressional actors colored by party and sized by centrality
        - Edges: Influence connections weighted by adjacency matrix values
        - Highlights: Leader positions and network structure
        
        Args:
            output_path: File path to save the visualization.
            dpi: Resolution of the output image.
        """
        if not _HAS_NETWORKX or not _HAS_MATPLOTLIB:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("Visualization requires networkx and matplotlib. Skipping network rendering.")
            return

        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Build networkx graph from adjacency matrix with actor metadata
        G = nx.DiGraph()
        
        # Extract ideology and party information
        ideologies = []
        parties = []
        centralities = []
        
        for actor in self.actors:
            G.add_node(actor.id)
            ideology = float(actor.ideology[0]) if isinstance(actor.ideology, np.ndarray) else float(actor.ideology)
            ideologies.append(ideology)
            parties.append(actor.party)
            centralities.append(actor.centrality)

        # Add weighted edges from adjacency matrix
        n = len(self.actors)
        for i in range(n):
            for j in range(n):
                weight = float(self.adj_matrix[i, j])
                if weight > 0:  # Only add positive edges
                    G.add_edge(self.actors[i].id, self.actors[j].id, weight=weight)

        # Prepare visualization
        fig, ax = plt.subplots(figsize=(14, 10), dpi=dpi)
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42, weight='weight')

        # Color nodes by party
        party_colors = {'Republican': '#E81B23', 'Democratic': '#0015BC', 'Independent': '#999999', 'A': '#FF6B6B', 'B': '#4169E1'}
        node_colors = [party_colors.get(parties[i], '#CCCCCC') for i in range(n)]

        # Size nodes by centrality (larger = more influential)
        node_sizes = [300 + 3000 * centralities[i] for i in range(n)]

        # Draw edges with varying width based on weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1.0
        edge_widths = [0.5 + 2.0 * (w / max_weight) for w in weights]

        # Draw network
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.3, 
                              arrowsize=15, arrowstyle='->', width=edge_widths)
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8, edgecolors='black', linewidths=2)

        # Draw labels for high-centrality nodes (leaders)
        leader_threshold = np.percentile(centralities, 75)
        leader_labels = {self.actors[i].id: str(self.actors[i].id) 
                        for i in range(n) if centralities[i] >= leader_threshold}
        nx.draw_networkx_labels(G, pos, labels=leader_labels, ax=ax, font_size=8, font_weight='bold')

        # Add title and legend
        scenario = self.regime_engine.scenario if hasattr(self.regime_engine, 'scenario') else 'Unknown'
        ax.set_title(f'Congressional Influence Network (G(t)) - Scenario: {scenario}', 
                    fontsize=14, fontweight='bold')

        # Create legend for parties
        legend_elements = [
            mpatches.Patch(facecolor='#E81B23', edgecolor='black', label='Republican'),
            mpatches.Patch(facecolor='#0015BC', edgecolor='black', label='Democratic'),
            mpatches.Patch(facecolor='#999999', edgecolor='black', label='Independent'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

        # Add statistics box
        stats_text = (
            f"Network Statistics:\n"
            f"Actors: {n}\n"
            f"Avg Centrality: {np.mean(centralities):.3f}\n"
            f"Network Density: {nx.density(G.to_undirected()):.3f}\n"
            f"Avg Party Homophily: {self._compute_homophily():.3f}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Network visualization saved to {output_path}")

    def _compute_homophily(self) -> float:
        """Compute average party homophily (preference for same-party connections)."""
        n = len(self.actors)
        if n < 2:
            return 0.0
        
        same_party_edges = 0.0
        total_edges = 0.0
        
        for i in range(n):
            for j in range(n):
                if i != j and self.adj_matrix[i, j] > 0:
                    total_edges += self.adj_matrix[i, j]
                    if self.actors[i].party == self.actors[j].party:
                        same_party_edges += self.adj_matrix[i, j]
        
        if total_edges == 0:
            return 0.0
        return float(same_party_edges / total_edges)



