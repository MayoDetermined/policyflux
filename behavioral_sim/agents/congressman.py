from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from behavioral_sim.agents.base import AbstractAgent, BehavioralParameters
from congress.actors import CongressMan as LegacyCongressMan


class CongressAgent(LegacyCongressMan, AbstractAgent):
    """Adapter that wraps the legacy CongressMan into the new agent protocol.

    The adapter keeps backward-compatible behavior while exposing the
    configurable :class:`BehavioralParameters` object and simple utility
    hooks expected by the new engine.
    """

    def __init__(
        self,
        id: int,
        data: dict[str, Any],
        ideal_point_model: Optional[Any] = None,
        params: Optional[BehavioralParameters] = None,
    ) -> None:
        super().__init__(id=id, data=data, ideal_point_model=ideal_point_model)
        self.params = params or BehavioralParameters(
            vulnerability=float(data.get("vulnerability", 0.1)),
            loyalty=float(data.get("loyalty", 0.5)),
            volatility=float(data.get("volatility", 0.1)),
        )

    # --- AbstractAgent API -------------------------------------------------
    def compute_utility(self, law: Any, context: Any, network_view: Any) -> float:
        """Score a proposed vote using law salience, context pressure, and network effects.

        Heurystyka:
        - IPM-proxy: projekcja ideologii na kierunek salience ustawy minus próg.
        - Karanie presją kontekstu Z(t) ważone podatnością (vulnerability).
        - Wzmocnienie średnim wpływem sieci (network_view) i dopasowaniem komisji.
        """

        # Ideology vector
        ideol = np.array(self.ideology, dtype=np.float32).reshape(-1)

        # Law features
        salience = None
        threshold = 0.0
        committee_bonus = 0.0
        if law is not None:
            salience = getattr(law, "salience", None)
            threshold = float(getattr(law, "threshold", 0.0) or 0.0)
            law_committees = set(getattr(law, "committee_ids", []) or [])
            if law_committees and getattr(self, "committee_memberships", None):
                overlap = law_committees.intersection(set(self.committee_memberships))
                committee_bonus = 0.05 * len(overlap)

        # IPM-like projection utility
        base_util = float(np.tanh(self.get_ideological_position()))
        if salience is not None:
            sal = np.array(salience, dtype=np.float32).reshape(-1)
            if sal.shape[0] == ideol.shape[0]:
                norm = float(np.linalg.norm(sal) + 1e-6)
                direction = sal / norm
                projection = float(np.dot(ideol, direction))
                base_util = (projection - threshold) * norm

        # Contextual pressure
        pressure = 0.0
        if context is not None:
            try:
                pressure = float(getattr(context, "get_current_pressure", lambda: getattr(context, "base_pressure", 0.0))())
            except Exception:
                pressure = float(getattr(context, "base_pressure", 0.0) or 0.0)

        # Network influence (mean of incoming weights)
        network_term = 0.0
        if network_view is not None:
            try:
                if isinstance(network_view, torch.Tensor):
                    network_term = float(torch.mean(network_view).item())
                else:
                    network_term = float(np.mean(network_view))
            except Exception:
                network_term = 0.0

        utility = base_util + network_term + committee_bonus - self.params.vulnerability * pressure
        return float(utility)

    def make_decision(self, utility: float) -> int:
        return int(utility >= 0.0)

    def update_state(self, outcome: Any, context: Any) -> None:
        party_line = 1 if getattr(outcome, "party_line", 1) else 0 if isinstance(outcome, object) else outcome
        vote = int(getattr(outcome, "vote", getattr(outcome, "action", 1))) if hasattr(outcome, "vote") else party_line
        try:
            self.record_vote(vote=vote, party_line_vote=party_line)
        except Exception:
            pass

    # Convenience hook for tensor device transfers used by runner
    def to_tensor(self, device: torch.device) -> torch.Tensor:
        arr = torch.as_tensor(self.ideology, dtype=torch.float32, device=device)
        return arr.view(1, -1)
