"""
Exponential Random Graph Model (ERGM) for network generation.

ERGM is used to model and generate random networks of congressional relationships,
such as coalition formations, co-sponsorships, or party alliances.
"""

import math
from typing import Any

import policyflux.pfrandom as pfrandom
from policyflux.exceptions import ValidationError


class ExponentialRandomGraphModel:
    """
    Exponential Random Graph Model for generating networks of legislative relationships.

    This model generates networks where the probability of edges depends on
    network statistics (density, transitivity, homophily, etc.).
    """

    def __init__(
        self,
        n_nodes: int,
        theta_density: float = -1.5,
        theta_transitivity: float = 0.5,
        theta_homophily: float = 0.3,
    ) -> None:
        """
        Initialize ERGM.

        Args:
            n_nodes: Number of nodes (legislators or other entities) in the network
            theta_density: Parameter for edge density (lower = fewer edges)
            theta_transitivity: Parameter for transitivity/clustering (higher = more triangles)
            theta_homophily: Parameter for homophily (higher = similar nodes connect)
        """
        if n_nodes < 2:
            raise ValidationError("Network must have at least 2 nodes")

        self.n_nodes: int = n_nodes
        self.theta_density: float = theta_density
        self.theta_transitivity: float = theta_transitivity
        self.theta_homophily: float = theta_homophily

        # Network adjacency matrix (undirected)
        self.adjacency: list[list[int]] = [[0] * n_nodes for _ in range(n_nodes)]
        self.node_attributes: list[dict[str, Any]] = [{} for _ in range(n_nodes)]

    def set_node_attribute(self, node_id: int, attribute: str, value: Any) -> None:
        """Set an attribute for a node (e.g., party, ideology)."""
        if not 0 <= node_id < self.n_nodes:
            raise ValidationError(f"Node ID {node_id} out of range [0, {self.n_nodes})")
        self.node_attributes[node_id][attribute] = value

    def get_node_attribute(self, node_id: int, attribute: str) -> Any:
        """Get an attribute for a node."""
        if not 0 <= node_id < self.n_nodes:
            raise ValidationError(f"Node ID {node_id} out of range [0, {self.n_nodes})")
        return self.node_attributes[node_id].get(attribute)

    def _edge_probability(self, i: int, j: int, current_adjacency: list[list[int]]) -> float:
        """
        Calculate the probability of edge (i, j) given current network state.

        Uses exponential family formulation with sufficient statistics.
        """
        # Density contribution
        density_stat = 1.0
        prob = math.exp(self.theta_density * density_stat)

        # Transitivity contribution (count triangles involving i-j edge)
        if self.theta_transitivity != 0:
            triangles = sum(
                current_adjacency[i][k] * current_adjacency[j][k]
                for k in range(self.n_nodes)
                if k != i and k != j
            )
            transitivity_stat = triangles
            prob *= math.exp(self.theta_transitivity * transitivity_stat)

        # Homophily contribution (similarity of node attributes)
        if self.theta_homophily != 0:
            homophily_stat = self._compute_homophily(i, j)
            prob *= math.exp(self.theta_homophily * homophily_stat)

        # Normalize to [0, 1]
        return prob / (1.0 + prob)

    def _compute_homophily(self, i: int, j: int) -> float:
        """
        Compute homophily statistic between nodes i and j.

        Returns 1.0 if nodes share attributes, 0.0 otherwise.
        """
        if not self.node_attributes[i] or not self.node_attributes[j]:
            return 0.0

        shared_attrs = sum(
            1.0
            for attr in self.node_attributes[i]
            if attr in self.node_attributes[j]
            and self.node_attributes[i][attr] == self.node_attributes[j][attr]
        )

        total_attrs = len(self.node_attributes[i])
        return shared_attrs / total_attrs if total_attrs > 0 else 0.0

    def generate(self, seed: int | None = None) -> list[list[int]]:
        """
        Generate a network using ERGM.

        Args:
            seed: Optional random seed for reproducibility

        Returns:
            Adjacency matrix of the generated network
        """
        if seed is not None:
            pfrandom.set_seed(seed)

        # Start with empty network
        self.adjacency = [[0] * self.n_nodes for _ in range(self.n_nodes)]

        # Generate edges
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                edge_prob = self._edge_probability(i, j, self.adjacency)
                if pfrandom.random() < edge_prob:
                    self.adjacency[i][j] = 1
                    self.adjacency[j][i] = 1  # Undirected

        return self.adjacency

    def get_adjacency(self) -> list[list[int]]:
        """Return the current adjacency matrix."""
        return [row[:] for row in self.adjacency]  # Return copy

    def get_degree(self, node_id: int) -> int:
        """Get the degree (number of connections) of a node."""
        if not 0 <= node_id < self.n_nodes:
            raise ValidationError(f"Node ID {node_id} out of range [0, {self.n_nodes})")
        return sum(self.adjacency[node_id])

    def get_density(self) -> float:
        """Calculate network density (proportion of possible edges present)."""
        if self.n_nodes < 2:
            return 0.0
        max_edges = (self.n_nodes * (self.n_nodes - 1)) / 2
        actual_edges = sum(sum(row) for row in self.adjacency) / 2
        return actual_edges / max_edges

    def get_clustering_coefficient(self, node_id: int) -> float:
        """
        Get local clustering coefficient for a node.

        Measures the proportion of triangles among the node's neighbors.
        """
        if not 0 <= node_id < self.n_nodes:
            raise ValidationError(f"Node ID {node_id} out of range [0, {self.n_nodes})")

        neighbors = [i for i in range(self.n_nodes) if self.adjacency[node_id][i]]
        if len(neighbors) < 2:
            return 0.0

        triangles = sum(
            self.adjacency[neighbors[i]][neighbors[j]]
            for i in range(len(neighbors))
            for j in range(i + 1, len(neighbors))
        )

        max_triangles = len(neighbors) * (len(neighbors) - 1) / 2
        return triangles / max_triangles if max_triangles > 0 else 0.0

    def get_average_clustering_coefficient(self) -> float:
        """Get the average clustering coefficient across all nodes."""
        if self.n_nodes == 0:
            return 0.0
        return sum(self.get_clustering_coefficient(i) for i in range(self.n_nodes)) / self.n_nodes

    def get_connected_component_sizes(self) -> list[int]:
        """
        Find connected components in the network.

        Returns a list of component sizes.
        """
        visited = [False] * self.n_nodes
        components = []

        def dfs(node: int) -> int:
            visited[node] = True
            size = 1
            for neighbor in range(self.n_nodes):
                if self.adjacency[node][neighbor] and not visited[neighbor]:
                    size += dfs(neighbor)
            return size

        for i in range(self.n_nodes):
            if not visited[i]:
                components.append(dfs(i))

        return sorted(components, reverse=True)
