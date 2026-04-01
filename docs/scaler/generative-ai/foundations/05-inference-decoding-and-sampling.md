---
title: Inference, Decoding, and Sampling
sidebar_position: 5
description: How models generate outputs at runtime, including logits, softmax, greedy decoding, temperature, top-k, top-p, and latency concerns.
---

# Inference, Decoding, and Sampling

Once a model is trained, the next question is how it generates outputs in practice.

## What happens during inference

At each generation step:

1. model reads current token context
2. model produces scores for possible next tokens
3. decoding logic chooses the next token
4. chosen token is appended
5. process repeats

## Why decoding matters

The same model can behave very differently depending on decoding strategy.

- greedy decoding can be stable but repetitive
- temperature can increase creativity
- top-k and top-p can reduce low-quality randomness

## Toy softmax sampler

```python
import math
import random


def softmax(values: list[float]) -> list[float]:
    exps = [math.exp(v) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def sample_token(logits: list[float]) -> int:
    probs = softmax(logits)
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return i
    return len(probs) - 1
```

### Code explanation

This is a toy demonstration of token sampling.

- `softmax` converts arbitrary scores into probabilities that sum to `1`
- a random number is sampled in `[0, 1)`
- the code walks cumulative probability mass until that random threshold is crossed
- the chosen index represents the sampled next token

Real systems add:

- temperature scaling
- top-k or top-p filtering
- repetition penalties
- special stop-token handling

## Decoding strategies to know

- greedy decoding
- temperature sampling
- top-k sampling
- top-p or nucleus sampling
- beam search for some seq2seq tasks

## Runtime engineering concepts

- batching
- streaming
- KV cache
- latency per generated token
- throughput under concurrency

## Important interview questions

- What is the difference between logits and probabilities?
- How does temperature affect generation?
- What is the difference between top-k and top-p?
- Why does token-by-token decoding create latency?
- What is KV cache used for?

## Quick revision

- inference is token-by-token generation
- decoding strategy strongly shapes model behavior
- runtime quality is a combination of model quality and inference engineering
