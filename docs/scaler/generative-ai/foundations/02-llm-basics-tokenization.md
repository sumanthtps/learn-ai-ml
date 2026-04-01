---
title: LLM Basics and Tokenization
sidebar_position: 2
description: Beginner-to-intermediate notes on tokenization, context windows, embeddings intuition, and how LLM inputs are represented.
---

# LLM Basics and Tokenization

Before learning transformers deeply, you should understand what actually flows into a language model. The model does not read plain English the way a human does. It receives sequences of tokens mapped to IDs, then vectors.

## Why this topic matters

- token limits affect product design
- prompt cost depends on tokens
- retrieval systems often fail because of poor chunk-token tradeoffs
- tokenization explains why some words split strangely

## Core subtopics

- tokens
- tokenizers
- subword splitting
- token IDs
- context windows
- embeddings intuition

## What is a token

A token is a small unit of text processed by the model.

Examples:

- a full word
- part of a word
- punctuation
- whitespace-related segment

This is why "token count" and "word count" are not the same thing.

## Why tokenization exists

If models treated every possible word as a separate item:

- vocabulary would explode
- rare words would be hard to handle
- misspellings and morphology would be messy

Subword tokenization solves this by splitting text into reusable pieces.

## Context window

The context window is the amount of token history the model can use at once.

Why it matters:

- longer prompts cost more
- retrieved context competes for space
- multi-turn systems need memory strategies

## Toy tokenizer example

```python
def whitespace_tokenize(text: str) -> list[str]:
    return text.strip().split()


sample = "Generative AI changes software workflows."
print(whitespace_tokenize(sample))
```

### Code explanation

This is not how production LLM tokenizers work, but it helps beginners understand the idea.

- `strip()` removes extra whitespace at the edges
- `split()` separates on spaces
- the output list is a sequence of units

Real tokenizers are more advanced because they:

- split unknown words into smaller parts
- support punctuation and Unicode robustly
- optimize vocabulary efficiency

## Turning tokens into IDs

Models do not process raw strings directly. They first map tokens to integers.

Example:

```text
"hello" -> 314
"world" -> 995
```

Those IDs are then converted into embeddings.

## Sliding context example

```python
def windows(tokens: list[str], size: int) -> list[list[str]]:
    out = []
    for i in range(len(tokens) - size + 1):
        out.append(tokens[i:i + size])
    return out
```

### Code explanation

This toy function demonstrates how sequence windows are formed.

- each slice is a short local context
- sequence models are trained to predict what comes next from such contexts

In real training, this idea appears at much larger scale with token IDs rather than plain strings.

## Common beginner confusions

- token count is not word count
- context length is not "memory forever"
- embeddings are not the same as retrieved vectors from a vector database, even though both are vectors

## Important interview questions

- Why do LLMs use tokenization instead of raw words?
- Why are token counts important in GenAI products?
- What is a context window?
- Why can a single word become multiple tokens?

## Quick revision

- models consume token sequences, not raw text
- tokenization affects cost, latency, and context budgeting
- token IDs become embeddings before deeper model computation begins
