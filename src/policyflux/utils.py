"""Reward functions for congressional RL training.

Defines how to compute rewards for voting decisions based on:
- Loyalty (party alignment)
- Ideal Point Model (ideology alignment)
- Electoral outcomes (public opinion)
"""

from typing import Optional
import numpy as np


class RewardComputer:
    """Computes rewards for congressional voting decisions."""
    
    @staticmethod
    def compute_loyalty_reward(
        vote_action: int,
        party_line_vote: int,
        loyalty_weight: float = 1.0
    ) -> float:
        """Compute reward component for party loyalty.
        
        R_loyalty = +loyalty_weight if vote matches party line
                  = -loyalty_weight if vote opposes party line
        
        Args:
            vote_action: Action taken (0=against, 1=for).
            party_line_vote: Party's recommended vote (0 or 1).
            loyalty_weight: Weight for loyalty reward (default: 1.0).
        
        Returns:
            Reward in [-loyalty_weight, +loyalty_weight].
        """
        if vote_action == party_line_vote:
            return loyalty_weight
        else:
            return -loyalty_weight
    
    @staticmethod
    def compute_ipm_reward(
        actor_ideology: np.ndarray,
        law_salience: np.ndarray,
        law_threshold: float,
        vote_action: int,
        ipm_weight: float = 1.0
    ) -> float:
        """Compute reward component based on Ideal Point Model alignment.
        
        Reward is higher when vote aligns with actor's ideological preference
        based on the IPM formula.
        
        Args:
            actor_ideology: Actor's ideal point vector.
            law_salience: Law's salience vector.
            law_threshold: Law's threshold parameter.
            vote_action: Vote taken (0=against, 1=for).
            ipm_weight: Weight for IPM reward (default: 1.0).
        
        Returns:
            Reward based on IPM alignment.
        """
        # Compute IPM preference
        linear_pred = np.dot(law_salience, actor_ideology) - law_threshold
        ipm_preference = 1.0 / (1.0 + np.exp(-np.clip(linear_pred, -10, 10)))  # Sigmoid
        
        # Reward matches vote to IPM preference
        if vote_action == 1:  # Voting FOR
            reward = ipm_weight * (2.0 * ipm_preference - 1.0)  # In [-weight, +weight]
        else:  # Voting AGAINST
            reward = ipm_weight * (2.0 * (1.0 - ipm_preference) - 1.0)
        
        return float(reward)
    
    @staticmethod
    def compute_electoral_reward(
        vote_action: int,
        district_preference: float,
        electoral_weight: float = 1.0
    ) -> float:
        """Compute reward component based on electoral/constituency alignment.
        
        Reward is positive when vote aligns with district preference.
        
        Args:
            vote_action: Vote taken (0=against, 1=for).
            district_preference: District's preference in [-1, 1] where:
                                -1 = strong preference against
                                +1 = strong preference for
                                 0 = neutral
            electoral_weight: Weight for electoral reward (default: 1.0).
        
        Returns:
            Reward in [-electoral_weight, +electoral_weight].
        """
        if vote_action == 1:  # Voting FOR
            return electoral_weight * district_preference
        else:  # Voting AGAINST
            return -electoral_weight * district_preference
    
    @staticmethod
    def compute_composite_reward(
        vote_action: int,
        party_line_vote: int,
        actor_ideology: np.ndarray,
        law_salience: np.ndarray,
        law_threshold: float,
        district_preference: Optional[float] = None,
        loyalty_weight: float = 1.0,
        ipm_weight: float = 1.0,
        electoral_weight: float = 0.5
    ) -> float:
        """Compute composite reward combining all components.
        
        Total reward combines:
        - Loyalty to party (40% weight)
        - IPM alignment (40% weight)
        - Electoral alignment (20% weight)
        
        Args:
            vote_action: Vote taken (0=against, 1=for).
            party_line_vote: Party's recommended vote (0 or 1).
            actor_ideology: Actor's ideal point.
            law_salience: Law's salience vector.
            law_threshold: Law's threshold.
            district_preference: District preference in [-1, 1]. If None, assumed 0.
            loyalty_weight: Base loyalty weight.
            ipm_weight: Base IPM weight.
            electoral_weight: Base electoral weight.
        
        Returns:
            Composite reward (typically normalized to [-1, 1]).
        """
        # Compute individual components
        loyalty_reward = RewardComputer.compute_loyalty_reward(
            vote_action, party_line_vote, loyalty_weight
        )
        
        ipm_reward = RewardComputer.compute_ipm_reward(
            actor_ideology, law_salience, law_threshold, vote_action, ipm_weight
        )
        
        electoral_reward = 0.0
        if district_preference is not None:
            electoral_reward = RewardComputer.compute_electoral_reward(
                vote_action, district_preference, electoral_weight
            )
        
        # Weighted combination
        total_weight = loyalty_weight + ipm_weight + (electoral_weight if district_preference is not None else 0)
        
        if total_weight > 0:
            composite = (loyalty_reward + ipm_reward + electoral_reward) / total_weight
        else:
            composite = 0.0
        
        return float(np.clip(composite, -1.0, 1.0))
    
    @staticmethod
    def compute_political_capital_reward(
        political_capital: float,
        political_capital_max: float,
        action_against_party: bool,
        capital_weight: float = 0.5
    ) -> float:
        """Compute reward based on political capital state.
        
        Reward is higher when actor has more political capital to spend,
        and negative if spending capital against party line.
        
        Args:
            political_capital: Current political capital.
            political_capital_max: Maximum political capital.
            action_against_party: Whether action goes against party.
            capital_weight: Weight for capital reward.
        
        Returns:
            Capital-based reward component.
        """
        capital_ratio = political_capital / max(political_capital_max, 1e-6)
        
        if action_against_party:
            # Cost for using capital against party
            return -capital_weight * (1.0 - capital_ratio)
        else:
            # Benefit for using capital with party
            return capital_weight * capital_ratio




