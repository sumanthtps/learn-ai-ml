---
title: LLM Training, Scaling, and Alignment
sidebar_position: 4
description: Pretraining, supervised fine-tuning, preference alignment, data curation, and scaling concepts for large language models.
---

# LLM Training, Scaling, and Alignment

To master GenAI, you need to understand not just how to use models, but how they are created and improved.

## Big picture

Modern LLM development often has several stages:

1. pretraining
2. supervised fine-tuning
3. preference alignment
4. deployment and continual evaluation

## Pretraining

Pretraining usually teaches next-token prediction over huge text corpora.

The model repeatedly sees:

- context tokens
- correct next token

and learns to assign higher probability to the correct continuation.

## Why pretraining is powerful

- it teaches broad language patterns
- it captures code, reasoning traces, style, and world patterns from data
- it creates a reusable base model

## Simple training-window example

```python
def make_next_token_examples(tokens: list[int], window: int):
    examples = []
    for i in range(len(tokens) - window):
        x = tokens[i:i + window]
        y = tokens[i + window]
        examples.append((x, y))
    return examples
```

### Code explanation

This toy function shows how next-token training data can be constructed.

- `x` is the context window
- `y` is the token the model should predict next
- many such examples train the model to estimate next-token probabilities

Real systems do this at massive scale with batching, masking, distributed training, and much more complex infrastructure.

## Scaling

Model quality often improves with:

- more parameters
- more high-quality data
- more compute

But scaling is not free. It also increases:

- training cost
- inference cost
- latency
- deployment complexity

## Supervised fine-tuning

After pretraining, supervised fine-tuning teaches the model to follow task or chat-style instructions more reliably.

Examples:

- customer support tone
- coding assistant behavior
- structured extraction format

## Preference alignment

The goal of alignment is to move the model toward preferred behavior.

Common high-level methods to know:

- RLHF
- DPO
- constitutional approaches

You do not need to derive the full math to discuss them well in interviews, but you should know what problem they solve.

## Data quality matters more than many beginners expect

Poor data causes:

- bad style
- hallucination reinforcement
- unsafe behavior
- brittle task performance

## Important interview questions

- What does pretraining teach a model?
- What is the difference between pretraining and supervised fine-tuning?
- Why is alignment needed after base-model training?
- What tradeoffs come with scaling model size?

## Quick revision

- pretraining learns broad next-token behavior
- instruction tuning improves task following
- alignment improves preference behavior and safety
- data quality is central at every stage
