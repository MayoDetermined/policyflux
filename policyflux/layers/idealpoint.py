from math import exp
from typing import Optional, List, Union

import torch
from torch.nn import nn

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..core.layer_template import Layer
from ..core.id_generator import get_id_generator
from ..core.types import UtilitySpace

## TO DO: Train functionality to be implemented

class IdealPointLayer(Layer):
    def __init__(self, 
                id: Optional[int] = None,
                input_dim: int = 2,
                output_dim: int = 2,
                space: Optional[UtilitySpace] = None,
                status_quo: Optional[UtilitySpace] = None,
                name: str = "IdealPoint") -> None:
        if id is None:
            id = get_id_generator().generate_layer_id()
        super().__init__(id, name, input_dim, output_dim)
        self.space = space if space is not None else []
        self.status_quo = status_quo if status_quo is not None else []

    # def train(self, data: Optional[List[UtilitySpace]] = None) -> None:
    #     """Train the encoder by creating the `space` from provided data.

    #     The `space` is set to the centroid (element-wise mean) of the
    #     list of utility-space vectors in `data`. If `status_quo` is empty,
    #     it will be initialized to the same centroid.

    #     Args:
    #         data: A list of utility-space vectors (each a sequence of numbers).

    #     Raises:
    #         ValueError: If no data is provided or input dimensions mismatch.
    #     """
    #     if not data:
    #         raise ValueError("No data provided to train IdealPointEncoder")

    #     # Validate consistent dimensions
    #     first_len = len(data[0])
    #     if any(len(d) != first_len for d in data):
    #         raise ValueError("All samples in data must have the same dimensionality")

    #     # Compute centroid (element-wise mean)
    #     centroid = [sum(values) / len(data) for values in zip(*data)]

    #     # Assign the learned space and (optionally) initialize status quo
    #     self.space = centroid
    #     if not self.status_quo:
    #         self.status_quo = centroid.copy()

    def compile(self) -> None:
        pass

    def _sq_distance(self, a: UtilitySpace, b: UtilitySpace) -> float:
        if len(a) != len(b):
            raise ValueError(f"Dimension mismatch: {len(a)} != {len(b)}")
        return sum((x - y) ** 2 for x, y in zip(a, b))

    def _delta_utility(self, bill_space: UtilitySpace) -> float:
        return (
            self._sq_distance(self.space, self.status_quo)
            - self._sq_distance(self.space, bill_space)
        )

    def _sigmoid(self, t: float) -> float:
        return 1 / (1 + exp(-t))

    def call(self, bill_space: UtilitySpace, **kwargs) -> float:
        delta_u = self._delta_utility(bill_space)
        return self._sigmoid(delta_u)
    
class IdealPointEncoderDF(nn.Module):
    def __init__(self, output_dim: int, dataset: pd.DataFrame) -> None:
        super().__init__()
        input_dim = dataset.shape[1]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))
    
    def encode(self, df: pd.DataFrame) -> torch.Tensor:
        """Encode any DataFrame to output_dim dimensional space.
        
        Args:
            df: DataFrame with the same number of columns as the training dataset
            
        Returns:
            Tensor of shape (n_rows, output_dim) representing the encoded space
            
        Raises:
            ValueError: If DataFrame dimensions don't match input_dim
        """
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

class IdealPointTextEncoder(nn.Module):
    def __init__(self, output_dim: int, corpus: List[str], max_features: int = 1000) -> None:
        """Text encoder that maps text documents to output_dim dimensional space.
        
        Args:
            output_dim: Dimensionality of the output space
            corpus: List of text documents to fit the vectorizer
            max_features: Maximum number of TF-IDF features to extract
        """
        super().__init__()
        self.output_dim = output_dim
        self.max_features = max_features
        
        # Initialize and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.vectorizer.fit(corpus)
        
        # Linear layer to map from TF-IDF space to output_dim
        self.linear = nn.Linear(len(self.vectorizer.get_feature_names_out()), output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear layer.
        
        Args:
            x: Tensor of shape (batch_size, tfidf_features)
            
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        return torch.sigmoid(self.linear(x))
    
    def encode(self, texts: Union[List[str], str]) -> torch.Tensor:
        """Encode text(s) to output_dim dimensional space.
        
        Args:
            texts: Single text string or list of text strings to encode
            
        Returns:
            Tensor of shape (n_texts, output_dim) representing the encoded space
        """
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