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

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

from ..core.layer_template import Layer
from ..core.id_generator import get_id_generator
from ..core.types import PolicyPosition, PolicySpace, UtilitySpace

from .data_layer_processor_template import LayerDataProcessor

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

class IdealPointEncoderDF(LayerDataProcessor):
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

class IdealPointTextEncoder(LayerDataProcessor):
    def __init__(self,
                 output_dim: int,
                 corpus: List[str],
                 max_features: int = 1000,
                 use_embeddings: bool = True,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 ngram_range: tuple = (1, 2),
                 hidden_dims: Optional[List[int]] = None) -> None:
        """Hybrid text encoder that maps text documents to output_dim dimensional space.

        Combines TF-IDF (for syntactic features) with sentence embeddings (for semantic features)
        and uses trainable neural network layers to map to the ideal point space.

        Args:
            output_dim: Dimensionality of the output ideal point space
            corpus: List of text documents to fit the vectorizer and embeddings
            max_features: Maximum number of TF-IDF features to extract
            use_embeddings: Whether to use sentence embeddings (semantic features)
            embedding_model: Name of the sentence-transformers model to use
            ngram_range: Range of n-grams for TF-IDF (captures syntactic patterns)
            hidden_dims: List of hidden layer dimensions for the neural network.
                        If None, uses [256, 128] as default
        """
        if not HAS_TORCH:
            raise ImportError("torch is required for IdealPointTextEncoder")

        self.output_dim: int = output_dim
        self.max_features: int = max_features
        self.use_embeddings: bool = use_embeddings
        self.ngram_range: tuple = ngram_range

        # Initialize TF-IDF vectorizer with n-grams for syntactic features
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )
        self.vectorizer.fit(corpus)
        tfidf_dim = len(self.vectorizer.get_feature_names_out())

        # Initialize sentence transformer for semantic features
        if use_embeddings:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers is required when use_embeddings=True. "
                    "Install with: pip install sentence-transformers"
                )
            self.embedding_model: Optional[Any] = SentenceTransformer(embedding_model)
            # Get embedding dimension
            test_embedding = self.embedding_model.encode(["test"], convert_to_tensor=False)
            embedding_dim = len(test_embedding[0])
            input_dim = tfidf_dim + embedding_dim
        else:
            self.embedding_model = None
            input_dim = tfidf_dim

        # Build trainable neural network
        if hidden_dims is None:
            hidden_dims = [256, 128]

        if HAS_TORCH and nn:
            layers: List[Any] = []
            prev_dim = input_dim

            # Hidden layers with ReLU activation
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
                prev_dim = hidden_dim

            # Output layer with sigmoid activation
            layers.append(nn.Linear(prev_dim, output_dim))
            layers.append(nn.Sigmoid())

            self.network: Any = nn.Sequential(*layers)
        else:
            self.network = None
        
    def forward(self, x: Any) -> Any:
        """Forward pass through the neural network.

        Args:
            x: Tensor of shape (batch_size, input_features)

        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        if not HAS_TORCH:
            raise ImportError("torch is required for forward pass")
        return self.network(x)

    def _extract_features(self, texts: List[str]) -> Any:
        """Extract hybrid features from texts (TF-IDF + embeddings).

        Args:
            texts: List of text strings

        Returns:
            Tensor with concatenated TF-IDF and embedding features
        """
        if not HAS_TORCH:
            raise ImportError("torch is required for feature extraction")

        # Extract TF-IDF features (syntactic)
        tfidf_matrix = self.vectorizer.transform(texts)
        tfidf_features = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float32)

        # Extract semantic embeddings if enabled
        if self.use_embeddings and self.embedding_model is not None:
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            # Move to CPU if needed and ensure correct dtype
            if embeddings.device.type != 'cpu':
                embeddings = embeddings.cpu()
            embeddings = embeddings.to(dtype=torch.float32)

            # Concatenate TF-IDF and embeddings
            features = torch.cat([tfidf_features, embeddings], dim=1)
        else:
            features = tfidf_features

        return features

    def encode(self, texts: Union[List[str], str]) -> Any:
        """Encode text(s) to output_dim dimensional ideal point space.

        Args:
            texts: Single text string or list of text strings to encode

        Returns:
            Tensor of shape (n_texts, output_dim) representing the encoded ideal points
        """
        if not HAS_TORCH:
            raise ImportError("torch is required for encode method")

        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        # Extract hybrid features
        features = self._extract_features(texts)

        # Encode using forward pass
        with torch.no_grad():
            encoded = self.forward(features)

        return encoded

    def encode_df(self, df: pd.DataFrame, text_column: str) -> Any:
        """Encode DataFrame of texts to ideal point space.

        Args:
            df: DataFrame containing text data
            text_column: Name of the column containing text

        Returns:
            Tensor of shape (n_rows, output_dim) representing encoded ideal points

        Raises:
            ValueError: If text_column is not in the DataFrame
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        texts = df[text_column].tolist()
        return self.encode(texts)

    def train_step(self, texts: List[str], targets: Any, optimizer: Any, criterion: Any) -> float:
        """Perform a single training step.

        Args:
            texts: List of text strings
            targets: Target ideal points, shape (batch_size, output_dim)
            optimizer: PyTorch optimizer
            criterion: Loss function

        Returns:
            Loss value for this step
        """
        if not HAS_TORCH:
            raise ImportError("torch is required for training")

        optimizer.zero_grad()

        # Extract features and forward pass
        features = self._extract_features(texts)
        predictions = self.forward(features)

        # Compute loss and backpropagate
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        return loss.item()