# Qodo Embeddings

This repository demonstrates how to use the Qodo-Embed-1-1.5B model for generating embeddings and calculating similarities between text samples.

## Installation

```bash
pip install sentence-transformers
```

## Usage

The example below shows how to use the Qodo-Embed-1-1.5B model to generate embeddings for code snippets and calculate similarity scores between them.

```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("Qodo/Qodo-Embed-1-1.5B")

# Run inference
sentences = [
    'accumulator = sum(item.value for item in collection)',  
    'result = reduce(lambda acc, curr: acc + curr.amount, data, 0)',  
    'matrix = [[i*j for j in range(n)] for i in range(n)]',
    'a = sum(items)',
    'def add(list): s=0; for i in list: s+=i;  return s',
]

# Generate embeddings
embeddings = model.encode(sentences)
print(embeddings)
# Output shape: [5, 1536]

# Calculate similarity scores between embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# Output shape: [5, 5]
```

## Features

- Generate embeddings for code snippets
- Calculate similarity scores between code samples
- 1536-dimensional embedding vectors
- Based on the SentenceTransformer framework

## Model Information

The Qodo-Embed-1-1.5B model is hosted on the Hugging Face Hub and specializes in generating embeddings for code snippets. It can be used for various applications such as code similarity detection, code search, and code clustering.
