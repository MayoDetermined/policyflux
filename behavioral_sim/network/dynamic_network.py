from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from behavioral_sim.network.influence import InfluenceFunction


@dataclass
class DynamicNetwork:
    """Applies a chain of influence functions to produce G(t)."""

    base_adj: torch.Tensor
    influence_functions: List[InfluenceFunction] = field(default_factory=list)
    device: torch.device = torch.device("cpu")

    def compute(self, X: torch.Tensor, Z: Optional[torch.Tensor] = None) -> torch.Tensor:
        G = self.base_adj.to(self.device)
        features = X.to(self.device)
        context = Z.to(self.device) if Z is not None else None
        for fn in self.influence_functions:
            G = fn(G, features, context)
        # zero out self-loops for safety
        G = G.clone()
        G.fill_diagonal_(0.0)
        return G
