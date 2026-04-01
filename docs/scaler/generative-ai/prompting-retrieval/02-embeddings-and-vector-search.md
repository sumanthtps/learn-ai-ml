---
title: Embeddings and Vector Search
sidebar_position: 2
description: Embeddings, similarity metrics, approximate nearest neighbor search, and semantic retrieval foundations.
---

# Embeddings and Vector Search

Embeddings are the bridge between unstructured language and searchable mathematical representation.

## Why embeddings matter

- semantic search
- retrieval
- clustering
- recommendation
- classification and routing

## Visual intuition

![Embedding projection example](https://commons.wikimedia.org/wiki/Special:Redirect/file/2016_02_mini_embedding.png)

Image source: [Wikimedia Commons - 2016 02 mini embedding](https://commons.wikimedia.org/wiki/File:2016_02_mini_embedding.png)

This kind of visualization helps build the intuition that semantically related items tend to appear near one another in vector space.

## What an embedding is

An embedding is a dense numeric vector representing meaning or structure.

Two texts that mean similar things should ideally have vectors that point in similar directions.

## Cosine similarity example

```python
import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)
```

### Code explanation

Cosine similarity measures angle similarity rather than raw magnitude.

- `dot` measures alignment
- `norm_a` and `norm_b` compute vector lengths
- dividing by both lengths normalizes the score

This is widely used because we often care more about semantic direction than vector size.

## Exact vs approximate search

In small datasets, exact nearest-neighbor search is fine.

In large datasets, approximate nearest-neighbor methods are preferred because they are much faster.

Terms to know:

- ANN
- HNSW
- IVF
- product quantization

## Good retrieval depends on more than embeddings

Also important:

- document chunking
- metadata
- filters
- reranking
- query rewriting

## Important interview questions

- What is an embedding?
- Why is cosine similarity common?
- Why use approximate nearest-neighbor search?
- What are the limits of vector search alone?

## Quick revision

- embeddings turn meaning into vectors
- similarity search retrieves semantically related content
- retrieval quality depends on both embedding quality and indexing strategy
