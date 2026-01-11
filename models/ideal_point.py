import torch.nn as nn
import torch

class IdealPointModel(nn.Module):
    """
    Multi-dimensional logistic ideal point model:
    P(vote=1) = σ( ||a_j|| * (x_i - b_j) )
    
    Wymiary:
    - Dimension 1: Pryncypia (lewicowość vs. prawicowość)
    - Dimension 2: Ekonomicznie (interwencjonizm vs. wolny rynek)
    - Dimension 3: Emotions axis (populizm vs. establishment)
    
    x_i: legislator's position in multi-dimensional space (ideal points)
    a_j: discriminant/salience of vote j (how much does this vote distinguish?)
    b_j: difficulty parameter (legislative threshold)
    """

    def __init__(self, 
                 n_legislators: int, 
                 n_votes: int, 
                 dim=3
                 ) -> None:
        
        super().__init__()

        self.dim = dim
        self.x = nn.Parameter(torch.randn(n_legislators, dim) * 0.1)  # Legislator ideal points
        self.a = nn.Parameter(torch.ones(n_votes, dim))  # Discriminant weights
        self.b = nn.Parameter(torch.zeros(n_votes))  # Difficulty parameters

    def forward(self, leg_ids, vote_ids):
        """
        Args:
            leg_ids: tensor of legislator indices
            vote_ids: tensor of vote indices
        
        Returns:
            Probability of voting YES for each (legislator, vote) pair
        """
        # Get legislator ideal points and vote parameters
        xi = self.x[leg_ids]  # Shape: (batch_size, dim)
        a = self.a[vote_ids]  # Shape: (batch_size, dim)
        b = self.b[vote_ids]  # Shape: (batch_size,)
        
        # Compute euclidean distance between legislator position and vote threshold
        # For multi-dim: distance = ||xi - b|| where b is broadcasted to all dimensions
        if self.dim > 1:
            # Broadcast b (scalar) to match all dimensions
            b_expanded = b.unsqueeze(-1)  # Shape: (batch_size, 1)
            # Compute L2 distance for each sample in the batch
            distance = torch.norm(xi - b_expanded, dim=1, p=2)  # Shape: (batch_size,)
        else:
            distance = torch.abs(xi.squeeze() - b)
        
        # Probability: sigmoid(salience * distance)
        # a represents how much this vote distinguishes legislators
        salience = torch.norm(a, dim=1, p=2) if self.dim > 1 else a.squeeze()
        return torch.sigmoid(salience * distance)

