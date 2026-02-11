from math import exp
from typing import Optional, List, Union, Any

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..core.layer_template import Layer
from ..core.id_generator import get_id_generator
from ..core.types import PolicyPosition, PolicySpace, UtilitySpace

class IdealPointLayer(Layer, PolicySpace):
    def __init__(self, 
                id: Optional[int] = None,
                input_dim: int = 2,
                output_dim: int = 2,
                space: Optional[PolicySpace] = None,
                status_quo: Optional[PolicySpace] = None,
                name: str = "IdealPoint") -> None:
        if id is None:
            id = get_id_generator().generate_layer_id()
        super().__init__(id, name, input_dim, output_dim)
        PolicySpace.__init__(self, input_dim)
        self.space: PolicySpace = space if space is not None else PolicySpace(input_dim)
        self.status_quo: PolicySpace = status_quo if status_quo is not None else PolicySpace(input_dim)

    def compile(self) -> None:
        pass

    def _sq_distance(self, a: Union[PolicySpace, List[float]], b: Union[PolicySpace, List[float]]) -> float:
        # Handle both PolicySpace objects and lists
        if isinstance(a, PolicySpace):
            a_pos = a.get_position()
            a_dim = a.dimensions
        else:
            a_pos = a
            a_dim = len(a)

        if isinstance(b, PolicySpace):
            b_pos = b.get_position()
            b_dim = b.dimensions
        else:
            b_pos = b
            b_dim = len(b)

        if a_dim != b_dim:
            raise ValueError(f"Dimension mismatch: {a_dim} != {b_dim}")
        return sum((x - y) ** 2 for x, y in zip(a_pos, b_pos))

    def _delta_utility(self, bill_space: Union[PolicySpace, List[float]]) -> float:
        return (
            self._sq_distance(self.space, self.status_quo)
            - self._sq_distance(self.space, bill_space)
        )

    def _sigmoid(self, t: float) -> float:
        return 1 / (1 + exp(-t))

    def call(self, bill_space: UtilitySpace, **kwargs) -> float:
        delta_u = self._delta_utility(bill_space)
        return self._sigmoid(delta_u)

class IdealPointEncoderDF:
    def __init__(self, output_dim: int, dataset: pd.DataFrame) -> None:
        if not HAS_TORCH:
            raise ImportError("torch is required for IdealPointEncoderDF")
        input_dim: int = dataset.shape[1]
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        if HAS_TORCH and nn:
            self.linear: Any = nn.Linear(input_dim, output_dim)
        else:
            self.linear = None

    def forward(self, x: Any) -> Any:
        if not HAS_TORCH:
            raise ImportError("torch is required for forward pass")
        return torch.sigmoid(self.linear(x))
    
    def encode(self, df: pd.DataFrame) -> Any:
        """Encode any DataFrame to output_dim dimensional space.
        
        Args:
            df: DataFrame with the same number of columns as the training dataset
            
        Returns:
            Tensor of shape (n_rows, output_dim) representing the encoded space
            
        Raises:
            ValueError: If DataFrame dimensions don't match input_dim
        """
        if not HAS_TORCH:
            raise ImportError("torch is required for encode method")
        if df.shape[1] != self.input_dim:
            raise ValueError(
                f"DataFrame has {df.shape[1]} columns, but encoder expects {self.input_dim}"
            )
        
        # Convert DataFrame to tensor
        x = torch.tensor(df.values, dtype=torch.float32)
        
        # Encode using forward pass
        with torch.no_grad():
            encoded = self.forward(x)
        
        return encoded

class IdealPointTextEncoder:
    def __init__(self, output_dim: int, corpus: List[str], max_features: int = 1000) -> None:
        """Text encoder that maps text documents to output_dim dimensional space.
        
        Args:
            output_dim: Dimensionality of the output space
            corpus: List of text documents to fit the vectorizer
            max_features: Maximum number of TF-IDF features to extract
        """
        if not HAS_TORCH:
            raise ImportError("torch is required for IdealPointTextEncoder")
        self.output_dim: int = output_dim
        self.max_features: int = max_features
        
        # Initialize and fit TF-IDF vectorizer
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(max_features=max_features)
        self.vectorizer.fit(corpus)
        
        # Linear layer to map from TF-IDF space to output_dim
        if HAS_TORCH and nn:
            self.linear: Any = nn.Linear(len(self.vectorizer.get_feature_names_out()), output_dim)
        else:
            self.linear = None
        
    def forward(self, x: Any) -> Any:
        """Forward pass through the linear layer.
        
        Args:
            x: Tensor of shape (batch_size, tfidf_features)
            
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        if not HAS_TORCH:
            raise ImportError("torch is required for forward pass")
        return torch.sigmoid(self.linear(x))
    
    def encode(self, texts: Union[List[str], str]) -> Any:
        """Encode text(s) to output_dim dimensional space.
        
        Args:
            texts: Single text string or list of text strings to encode
            
        Returns:
            Tensor of shape (n_texts, output_dim) representing the encoded space
        """
        if not HAS_TORCH:
            raise ImportError("torch is required for encode method")
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        # Transform texts to TF-IDF features
        tfidf_matrix = self.vectorizer.transform(texts)
        
        # Convert sparse matrix to dense tensor
        x = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float32)
        
        # Encode using forward pass
        with torch.no_grad():
            encoded = self.forward(x)
        
        return encoded