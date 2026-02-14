import torch
from torch.nn.utils.rnn import pad_sequence
import re
from collections import Counter

from data_processor_template import DataProcessor

class SimpleTextVectorizer(DataProcessor):
    def __init__(self, texts_to_process: list[str], tokenizer_name: str = 'basic_english'):
        super().__init__(name="SimpleTextVectorizer")
        self.texts = texts_to_process
        self.tokenizer_name = tokenizer_name
        self.vocab = None
        self.word_to_idx = {}
        self.idx_to_word = {}

    def tokenize(self, text: str) -> list[str]:
        """Simple tokenizer that splits on whitespace and punctuation"""
        text = text.lower()
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def build_vocab(self, min_freq: int = 1):
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
        tokenized_batch = [torch.tensor(self.text_pipeline(text), dtype=torch.long) for text in batch]
        padded_batch = pad_sequence(tokenized_batch, batch_first=True, padding_value=0)
        return padded_batch
    
    def fit(self, data):
        # TO DO: Implement fit method if needed. For text processing, this might involve building the vocabulary.
        """Easily fit the processor to the data (build vocab)"""
        pass
    
    def process(self) -> torch.Tensor:
        """Process all texts and return as padded tensor"""
        if self.vocab is None:
            self.build_vocab()
        return self.collect_batch(self.texts)

    def vectorize(self, text: str) -> torch.Tensor:
        """Vectorize a single text"""
        if not self.word_to_idx:
            self.build_vocab()
        return torch.tensor(self.text_pipeline(text), dtype=torch.long)
    