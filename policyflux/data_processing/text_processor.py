from __future__ import annotations

import re
from collections import Counter
from typing import Any

from policyflux.exceptions import OptionalDependencyError

try:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    HAS_TORCH = True
except ImportError:
    torch = None
    pad_sequence = None
    HAS_TORCH = False

from .abstract_data_processor import DataProcessor


class SimpleTextVectorizer(DataProcessor):
    def __init__(self, texts_to_process: list[str], tokenizer_name: str = "basic_english"):
        if not HAS_TORCH:
            raise OptionalDependencyError(
                "SimpleTextVectorizer requires torch. Install optional dependency with: "
                "pip install policyflux[torch]"
            )
        super().__init__(name="SimpleTextVectorizer")
        self.texts = texts_to_process
        self.tokenizer_name = tokenizer_name
        self.vocab = None
        self.word_to_idx: dict[str, int] = {}
        self.idx_to_word: dict[int, str] = {}

    def tokenize(self, text: str) -> list[str]:
        """Simple tokenizer that splits on whitespace and punctuation"""
        text = text.lower()
        # Split on whitespace and punctuation
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def build_vocab(self, min_freq: int = 1) -> SimpleTextVectorizer:
        """Build vocabulary from all texts"""
        # Collect all tokens
        all_tokens = []
        for text in self.texts:
            all_tokens.extend(self.tokenize(text))

        # Count token frequencies
        token_counts = Counter(all_tokens)

        # Build vocab with special tokens
        self.word_to_idx = {"<pad>": 0, "<unk>": 1}
        idx = 2

        for word, count in token_counts.items():
            if count >= min_freq:
                self.word_to_idx[word] = idx
                idx += 1

        # Create reverse mapping
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        return self

    def text_pipeline(self, text: str) -> list[int]:
        """Convert text to list of indices"""
        tokens = self.tokenize(text)
        return [self.word_to_idx.get(token, 1) for token in tokens]  # 1 is <unk>

    def collect_batch(self, batch: list[str]) -> torch.Tensor:
        """Convert batch of texts to padded tensor"""
        if not HAS_TORCH or pad_sequence is None:
            raise OptionalDependencyError("torch is required for collect_batch")
        tokenized_batch = [
            torch.tensor(self.text_pipeline(text), dtype=torch.long) for text in batch
        ]
        padded_batch = pad_sequence(tokenized_batch, batch_first=True, padding_value=0)
        return padded_batch

    def fit(self, data: list[Any]) -> None:
        """Fit the processor to the data (build vocab)."""
        self.texts = data
        self.build_vocab()

    def process(self, data: list[Any]) -> list[Any]:
        """Process all texts and return vectorized representations."""
        self.fit(data)
        return [self.vectorize(text) for text in data]

    def vectorize(self, text: str) -> torch.Tensor:
        """Vectorize a single text"""
        if not HAS_TORCH:
            raise OptionalDependencyError("torch is required for vectorize")
        if not self.word_to_idx:
            self.build_vocab()
        return torch.tensor(self.text_pipeline(text), dtype=torch.long)
