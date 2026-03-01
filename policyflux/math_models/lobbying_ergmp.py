"""
Lobbying Exponential Random Graph Model (ERGM) for network generation.

Models the network of connections between lobbyists and legislators,
where edge probability depends on homophily (similar ideologies/interests)
and network statistics (transitivity, density).
"""

import math
from typing import Any

import policyflux.pfrandom as pfrandom
from policyflux.exceptions import ValidationError


class LobbyingERGMPModel:
    """
    Exponential Random Graph Model for generating lobbying networks.

    This bipartite model generates networks where lobbyists connect to legislators.
    Edge probability depends on:
    - Density: Base probability of connection
    - Transitivity: Tendency for connected components (coalition building)
    - Homophily: Preference for similar ideology/interest alignment
    """

    def __init__(
        self,
        n_lobbyists: int,
        n_legislators: int,
        theta_density: float = -2.0,
        theta_transitivity: float = 0.5,
        theta_homophily: float = 0.5,
    ) -> None:
        """
        Initialize Lobbying ERGM.

        Args:
            n_lobbyists: Number of lobbyist nodes
            n_legislators: Number of legislator nodes
            theta_density: Parameter for edge density (lower = fewer edges)
            theta_transitivity: Parameter for transitivity/clustering
            theta_homophily: Parameter for homophily (higher = similar nodes connect)
        """
        if n_lobbyists < 1:
            raise ValidationError("Must have at least 1 lobbyist")
        if n_legislators < 1:
            raise ValidationError("Must have at least 1 legislator")

        self.n_lobbyists: int = n_lobbyists
        self.n_legislators: int = n_legislators
        self.theta_density: float = theta_density
        self.theta_transitivity: float = theta_transitivity
        self.theta_homophily: float = theta_homophily

        # Bipartite adjacency matrix: lobbyists x legislators
        self.adjacency: list[list[int]] = [[0] * n_legislators for _ in range(n_lobbyists)]

        # Node attributes
        self.lobbyist_attributes: list[dict[str, Any]] = [{} for _ in range(n_lobbyists)]
        self.legislator_attributes: list[dict[str, Any]] = [{} for _ in range(n_legislators)]

    def set_lobbyist_attribute(self, lobbyist_id: int, attribute: str, value: Any) -> None:
        """Set an attribute for a lobbyist (e.g., ideology, interest_type)."""
        if not 0 <= lobbyist_id < self.n_lobbyists:
            raise ValidationError(f"Lobbyist ID {lobbyist_id} out of range [0, {self.n_lobbyists})")
        self.lobbyist_attributes[lobbyist_id][attribute] = value

    def set_legislator_attribute(self, legislator_id: int, attribute: str, value: Any) -> None:
        """Set an attribute for a legislator (e.g., ideology, party)."""
        if not 0 <= legislator_id < self.n_legislators:
            raise ValidationError(
                f"Legislator ID {legislator_id} out of range [0, {self.n_legislators})"
            )
        self.legislator_attributes[legislator_id][attribute] = value

    def get_lobbyist_attribute(self, lobbyist_id: int, attribute: str) -> Any:
        """Get an attribute for a lobbyist."""
        if not 0 <= lobbyist_id < self.n_lobbyists:
            raise ValidationError(f"Lobbyist ID {lobbyist_id} out of range [0, {self.n_lobbyists})")
        return self.lobbyist_attributes[lobbyist_id].get(attribute)

    def get_legislator_attribute(self, legislator_id: int, attribute: str) -> Any:
        """Get an attribute for a legislator."""
        if not 0 <= legislator_id < self.n_legislators:
            raise ValidationError(
                f"Legislator ID {legislator_id} out of range [0, {self.n_legislators})"
            )
        return self.legislator_attributes[legislator_id].get(attribute)

    def _edge_probability(self, lobbyist_id: int, legislator_id: int) -> float:
        """
        Calculate the probability of edge (lobbyist, legislator).

        Uses exponential family formulation with sufficient statistics.
        """
        # Density contribution
        prob = math.exp(self.theta_density)

        # Transitivity contribution - count common connections
        if self.theta_transitivity != 0:
            common_connections = sum(
                self.adjacency[lobbyist_id][k] + self.adjacency[other_lobbyist][legislator_id]
                for k in range(self.n_legislators)
                for other_lobbyist in range(self.n_lobbyists)
                if k != legislator_id and other_lobbyist != lobbyist_id
            )
            transitivity_stat = min(common_connections, 10)  # Cap to prevent overflow
            prob *= math.exp(self.theta_transitivity * transitivity_stat)

        # Homophily contribution (similarity of attributes)
        if self.theta_homophily != 0:
            homophily_stat = self._compute_homophily(lobbyist_id, legislator_id)
            prob *= math.exp(self.theta_homophily * homophily_stat)

        # Normalize to [0, 1]
        return prob / (1.0 + prob)

    def _compute_homophily(self, lobbyist_id: int, legislator_id: int) -> float:
        """
        Compute homophily statistic between lobbyist and legislator.

        Returns 1.0 if they share attributes, 0.0 otherwise.
        """
        lobbyist_attrs = self.lobbyist_attributes[lobbyist_id]
        legislator_attrs = self.legislator_attributes[legislator_id]

        if not lobbyist_attrs or not legislator_attrs:
            return 0.0

        shared_attrs = sum(
            1.0
            for attr in lobbyist_attrs
            if attr in legislator_attrs and lobbyist_attrs[attr] == legislator_attrs[attr]
        )

        total_attrs = len(lobbyist_attrs)
        return shared_attrs / total_attrs if total_attrs > 0 else 0.0

    def generate(self, seed: int | None = None) -> list[list[int]]:
        """
        Generate a lobbying network using ERGM.

        Args:
            seed: Optional random seed for reproducibility

        Returns:
            Bipartite adjacency matrix (lobbyists x legislators)
        """
        if seed is not None:
            pfrandom.set_seed(seed)

        # Start with empty network
        self.adjacency = [[0] * self.n_legislators for _ in range(self.n_lobbyists)]

        # Generate edges
        for i in range(self.n_lobbyists):
            for j in range(self.n_legislators):
                edge_prob = self._edge_probability(i, j)
                if pfrandom.random() < edge_prob:
                    self.adjacency[i][j] = 1

        return self.adjacency

    def get_adjacency(self) -> list[list[int]]:
        """Return a copy of the current adjacency matrix."""
        return [row[:] for row in self.adjacency]

    def get_lobbyist_reach(self, lobbyist_id: int) -> list[int]:
        """Get list of legislator IDs that a lobbyist connects to."""
        if not 0 <= lobbyist_id < self.n_lobbyists:
            raise ValidationError(f"Lobbyist ID {lobbyist_id} out of range [0, {self.n_lobbyists})")
        return [j for j in range(self.n_legislators) if self.adjacency[lobbyist_id][j]]

    def get_legislator_exposure(self, legislator_id: int) -> list[int]:
        """Get list of lobbyist IDs that connect to a legislator."""
        if not 0 <= legislator_id < self.n_legislators:
            raise ValidationError(
                f"Legislator ID {legislator_id} out of range [0, {self.n_legislators})"
            )
        return [i for i in range(self.n_lobbyists) if self.adjacency[i][legislator_id]]

    def get_degree(self, is_lobbyist: bool, node_id: int) -> int:
        """
        Get the degree (number of connections) of a node.

        Args:
            is_lobbyist: True for lobbyist, False for legislator
            node_id: ID of the node
        """
        if is_lobbyist:
            if not 0 <= node_id < self.n_lobbyists:
                raise ValidationError(f"Lobbyist ID {node_id} out of range [0, {self.n_lobbyists})")
            return sum(self.adjacency[node_id])
        else:
            if not 0 <= node_id < self.n_legislators:
                raise ValidationError(
                    f"Legislator ID {node_id} out of range [0, {self.n_legislators})"
                )
            return sum(row[node_id] for row in self.adjacency)

    def get_density(self) -> float:
        """Calculate network density (proportion of possible edges present)."""
        if self.n_lobbyists == 0 or self.n_legislators == 0:
            return 0.0
        total_edges = sum(sum(row) for row in self.adjacency)
        max_edges = self.n_lobbyists * self.n_legislators
        return total_edges / max_edges

    def get_connected_lobbyists(self) -> int:
        """Count lobbyists with at least one connection."""
        return sum(1 for i in range(self.n_lobbyists) if self.get_degree(True, i) > 0)

    def get_connected_legislators(self) -> int:
        """Count legislators with at least one connection."""
        return sum(1 for j in range(self.n_legislators) if self.get_degree(False, j) > 0)

    def get_average_lobbyist_reach(self) -> float:
        """Get average number of legislators per lobbyist."""
        if self.n_lobbyists == 0:
            return 0.0
        return sum(self.get_degree(True, i) for i in range(self.n_lobbyists)) / self.n_lobbyists

    def get_average_legislator_exposure(self) -> float:
        """Get average number of lobbyists per legislator."""
        if self.n_legislators == 0:
            return 0.0
        return (
            sum(self.get_degree(False, j) for j in range(self.n_legislators)) / self.n_legislators
        )
