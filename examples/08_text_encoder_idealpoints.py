"""
Example: Text Encoder for Ideal Points

This example demonstrates how to use the hybrid IdealPointTextEncoder
to map policy texts to ideal point space, capturing both semantic
and syntactic differences.

The encoder combines:
- TF-IDF features (syntactic patterns, n-grams)
- Sentence embeddings (semantic meaning)
- Trainable neural network (custom mapping to ideal points)
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from policyflux.layers.idealpoint import IdealPointTextEncoder

# Example policy texts for different political positions
corpus = [
    "We need to increase taxes on the wealthy to fund social programs",
    "Lower taxes will stimulate economic growth and create jobs",
    "Healthcare is a human right and should be universally provided",
    "Free market competition improves healthcare quality and reduces costs",
    "Climate change requires immediate government intervention",
    "Market solutions are the best way to address environmental concerns",
    "We must strengthen workers' rights and increase minimum wage",
    "Reducing regulations will help businesses thrive and hire more workers",
    "Education funding should be increased to ensure equal opportunity",
    "School choice and voucher programs improve educational outcomes",
]

# Create encoder that maps texts to 2D ideal point space
# use_embeddings=True enables hybrid mode (TF-IDF + sentence embeddings)
encoder = IdealPointTextEncoder(
    output_dim=2,  # 2D ideal point space (e.g., economic left-right, social liberal-conservative)
    corpus=corpus,
    max_features=100,  # TF-IDF features
    use_embeddings=True,  # Enable semantic embeddings
    embedding_model="all-MiniLM-L6-v2",  # Fast, lightweight model
    ngram_range=(1, 2),  # Capture unigrams and bigrams for syntax
    hidden_dims=[128, 64]  # Neural network architecture
)

print("=== Basic Encoding ===")
# Encode single text
text = "We need progressive taxation to reduce inequality"
encoded = encoder.encode(text)
print(f"Text: {text}")
print(f"Ideal point: {encoded.numpy()}")
print()

# Encode multiple texts
texts = [
    "Strong labor unions protect worker rights",
    "Deregulation promotes business growth"
]
encoded_batch = encoder.encode(texts)
print("Batch encoding:")
for i, (text, point) in enumerate(zip(texts, encoded_batch)):
    print(f"{i+1}. {text}")
    print(f"   Ideal point: {point.numpy()}")
print()

print("=== DataFrame Encoding ===")
# Create DataFrame with policy texts
df = pd.DataFrame({
    'policy_id': range(len(corpus)),
    'policy_text': corpus,
    'party': ['D', 'R', 'D', 'R', 'D', 'R', 'D', 'R', 'D', 'R']
})

# Encode entire DataFrame
encoded_df = encoder.encode_df(df, text_column='policy_text')
print(f"Encoded {len(df)} policies to {encoder.output_dim}D ideal point space")
print(f"Shape: {encoded_df.shape}")
print(f"Sample ideal points:\n{encoded_df[:3].numpy()}")
print()

print("=== Training the Encoder ===")
# Example: Train encoder to learn specific ideal point mappings
# In practice, you would have labeled data with known ideal points

# Synthetic training data: labeled ideal points
# Assume left-leaning texts should map to (-1, 0) and right-leaning to (1, 0)
training_texts = [
    "Progressive taxation and social welfare",
    "Tax cuts and limited government intervention",
    "Universal healthcare coverage",
    "Private healthcare markets",
]
training_targets = torch.tensor([
    [-0.8, 0.2],   # Left-leaning position
    [0.8, 0.2],    # Right-leaning position
    [-0.7, 0.3],   # Left-leaning on social issues
    [0.7, -0.1],   # Right-leaning economic
], dtype=torch.float32)

# Setup training
optimizer = optim.Adam(encoder.network.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train for a few epochs
n_epochs = 50
print(f"Training for {n_epochs} epochs...")
for epoch in range(n_epochs):
    loss = encoder.train_step(training_texts, training_targets, optimizer, criterion)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss:.4f}")

print("\nAfter training:")
trained_encoded = encoder.encode(training_texts)
for i, (text, pred, target) in enumerate(zip(training_texts, trained_encoded, training_targets)):
    print(f"{i+1}. {text[:50]}...")
    print(f"   Predicted: {pred.numpy()}, Target: {target.numpy()}")
print()

print("=== Semantic and Syntactic Differences ===")
# Test how encoder captures semantic similarity vs syntactic differences
similar_texts = [
    "We must increase taxes on the rich",  # Similar semantics
    "Wealthy individuals should pay higher taxes",  # Same meaning, different syntax
    "Tax cuts for the wealthy are necessary",  # Opposite semantics
]

encoded_similar = encoder.encode(similar_texts)
print("Comparing semantically similar/different texts:")
for i, (text, point) in enumerate(zip(similar_texts, encoded_similar)):
    print(f"{i+1}. {text}")
    print(f"   Ideal point: {point.numpy()}")

# Calculate pairwise distances
def euclidean_distance(p1, p2):
    return torch.sqrt(torch.sum((p1 - p2) ** 2)).item()

dist_1_2 = euclidean_distance(encoded_similar[0], encoded_similar[1])
dist_1_3 = euclidean_distance(encoded_similar[0], encoded_similar[2])
print(f"\nDistance between texts 1 and 2 (similar semantics): {dist_1_2:.4f}")
print(f"Distance between texts 1 and 3 (opposite semantics): {dist_1_3:.4f}")
print()

print("=== Using Without Embeddings (TF-IDF only) ===")
# Create encoder with only TF-IDF (syntactic features)
encoder_tfidf = IdealPointTextEncoder(
    output_dim=2,
    corpus=corpus,
    use_embeddings=False,  # Disable embeddings
    ngram_range=(1, 3)  # Use more n-grams for syntax
)

encoded_tfidf = encoder_tfidf.encode(similar_texts)
print("Same texts with TF-IDF only:")
for i, (text, point) in enumerate(zip(similar_texts, encoded_tfidf)):
    print(f"{i+1}. {text}")
    print(f"   Ideal point: {point.numpy()}")
