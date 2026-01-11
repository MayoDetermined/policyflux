"""Law/Bill representation with IPM parameters.

This module defines the Law class representing legislative bills/laws
with their positions in ideological space and voting salience parameters.
"""

from typing import Optional, List
import numpy as np


class Law:
    """Represents a legislative bill with ideological position and voting parameters.
    
    A law has:
    - Multi-dimensional ideological position (salience vector a_j)
    - Difficulty/threshold parameter (b_j)
    - Metadata (id, title, session)
    
    The voting probability for legislator i follows the ideal point model:
    P(vote_i = 1) = sigmoid(||a_j|| * (x_i · (a_j / ||a_j||) - b_j))
    
    Where:
    - x_i: legislator's ideal point (multi-dimensional)
    - a_j: law's salience vector (how much does this law distinguish legislators?)
    - b_j: law's threshold parameter (difficulty of passing)
    """
    
    def __init__(
        self,
        law_id: int,
        salience: np.ndarray,
        threshold: float,
        title: str = "",
        session: int = 118,
        vote_ids: Optional[List[int]] = None,
        policy_domain: Optional[str] = None,
        committee_ids: Optional[List[str]] = None,
        saliency_score: Optional[float] = None,
        complexity: Optional[float] = None,
    ):
        """Initialize a Law object.
        
        Args:
            law_id: Unique identifier for the law.
            salience: Salience vector a_j (shape: (dim,) where dim is ideological dimensions).
                     Can be extracted from IPM or generated randomly.
            threshold: Difficulty parameter b_j (scalar). Lower values = easier to pass.
            title: Descriptive title of the law.
            session: Congressional session (e.g., 118 for 118th Congress).
            vote_ids: Optional list of roll-call IDs associated with this law.
        """
        self.law_id: int = int(law_id)
        self.salience: np.ndarray = np.array(salience, dtype=np.float32)
        self.threshold: float = float(threshold)
        self.title: str = str(title)
        self.session: int = int(session)
        self.vote_ids: List[int] = vote_ids if vote_ids is not None else []
        self.policy_domain: Optional[str] = policy_domain
        self.committee_ids: List[str] = committee_ids if committee_ids is not None else []
        self.saliency_score: Optional[float] = saliency_score
        self.complexity: Optional[float] = complexity
        
        # Validate dimensions
        if self.salience.ndim != 1:
            raise ValueError(f"Salience must be 1D array, got shape {self.salience.shape}")
        
        self.dim = len(self.salience)
    
    def get_voting_probability(self, legislator_ideal_point: np.ndarray) -> float:
        """Compute voting probability using ideal point model.
        
        P(vote=1) = sigmoid(salience_magnitude * (x_i · normalized_salience - threshold))
        
        Args:
            legislator_ideal_point: Legislator's ideal point x_i (shape: (dim,))
        
        Returns:
            Probability of voting yes (value in [0, 1])
        """
        if len(legislator_ideal_point) != self.dim:
            raise ValueError(
                f"Legislator ideal point dimension {len(legislator_ideal_point)} "
                f"does not match law dimension {self.dim}"
            )
        
        # Normalize salience to unit vector and compute magnitude
        salience_magnitude = np.linalg.norm(self.salience, ord=2)
        
        if salience_magnitude < 1e-8:
            # If salience is near-zero, return neutral probability
            return 0.5
        
        salience_normalized = self.salience / salience_magnitude
        
        # Dot product: legislator position projected onto law's ideological direction
        projection = float(np.dot(legislator_ideal_point, salience_normalized))
        
        # Utility: how far is legislator from threshold?
        utility = salience_magnitude * (projection - self.threshold)
        
        # Sigmoid activation
        probability = float(1.0 / (1.0 + np.exp(-utility)))
        
        return probability
    
    def get_state_vector(self) -> dict:
        """Return serializable state dictionary for the law."""
        return {
            "law_id": self.law_id,
            "salience": self.salience.tolist(),
            "threshold": self.threshold,
            "title": self.title,
            "session": self.session,
            "dim": self.dim,
            "vote_ids": self.vote_ids,
            "policy_domain": self.policy_domain,
            "committee_ids": self.committee_ids,
            "saliency_score": self.saliency_score,
            "complexity": self.complexity,
        }
    
    @classmethod
    def create_random(cls, law_id: int, dim: int = 3, **kwargs) -> "Law":
        """Create a law with random salience and threshold.
        
        Args:
            law_id: Unique identifier for the law.
            dim: Number of ideological dimensions (default: 3).
            **kwargs: Additional arguments passed to __init__.
        
        Returns:
            New Law instance with random parameters.
        """
        salience = np.random.randn(dim).astype(np.float32)
        threshold = float(np.random.uniform(-1.0, 1.0))
        return cls(law_id, salience, threshold, **kwargs)
    
    @classmethod
    def create_from_ipm_parameters(
        cls,
        law_id: int,
        vote_id: int,
        ipm_voting_params: dict,
        title: str = "",
        **kwargs
    ) -> "Law":
        """Create a law from trained IPM voting parameters.
        
        Args:
            law_id: Unique identifier for the law.
            vote_id: Index of the vote in IPM (used to extract a_j and b_j).
            ipm_voting_params: Dictionary with keys 'salience' and 'threshold'.
                             From export_actors_with_model()[2].
            title: Descriptive title of the law.
            **kwargs: Additional arguments passed to __init__.
        
        Returns:
            New Law instance initialized from IPM parameters.
        
        Raises:
            ValueError: If vote_id is out of range or voting_params is invalid.
        """
        if 'salience' not in ipm_voting_params or 'threshold' not in ipm_voting_params:
            raise ValueError("ipm_voting_params must have 'salience' and 'threshold' keys")
        
        salience_matrix = ipm_voting_params['salience']  # Shape: (n_votes, dim)
        threshold_vector = ipm_voting_params['threshold']  # Shape: (n_votes,)
        
        if vote_id >= len(salience_matrix) or vote_id < 0:
            raise ValueError(f"vote_id {vote_id} out of range [0, {len(salience_matrix)-1}]")
        
        salience = salience_matrix[vote_id]
        threshold = float(threshold_vector[vote_id])
        
        return cls(law_id, salience, threshold, title=title, vote_ids=[vote_id], **kwargs)
