---
title: Fine-tuning, Evaluation, and Guardrails
sidebar_position: 1
description: Fine-tuning concepts, evaluation methods, and safety guardrails for GenAI systems.
---

# Fine-tuning, Evaluation, and Guardrails

After prompting and retrieval, the next question is whether the system is accurate, stable, and safe.

## Introduction

This topic sits at the point where GenAI stops being a demo and starts becoming an engineering discipline. Many teams can prototype a chatbot. Far fewer can answer:

- when should we fine-tune
- how do we measure improvement
- how do we prevent unsafe output
- how do we know the system is actually getting better over time

## Prompting vs fine-tuning vs RAG

Prompting:

- easiest to start
- no model training required

RAG:

- best when knowledge must come from external documents

Fine-tuning:

- useful when behavior, style, or narrow task adaptation must improve consistently

## Fine-tuning intuition

Fine-tuning updates model parameters using task-specific data.

Use it when:

- output style must be consistent
- classification behavior needs improvement
- domain-specific patterns matter repeatedly

Do not fine-tune just to inject changing knowledge that RAG can supply.

## Types of adaptation to know

- prompt engineering
- retrieval augmentation
- supervised fine-tuning
- preference optimization
- parameter-efficient fine-tuning such as LoRA

## Example training-data format

```python
training_example = {
    "messages": [
        {"role": "system", "content": "You are a support assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "Open settings, choose security, then click reset password."}
    ]
}
```

### Code explanation

This is a simple chat-style supervised fine-tuning example.

- the model sees the input conversation
- the target behavior is the assistant message
- many such examples teach consistent behavior or style

High-quality fine-tuning data matters more than simply having a large amount of messy data.

## Evaluation

You need both offline and online evaluation.

Examples:

- exact match
- factuality checks
- retrieval precision
- answer relevance
- human review

## Evaluation layers to master

- model-level evaluation
- prompt-level evaluation
- retrieval-level evaluation
- end-to-end task evaluation
- online business evaluation

Good teams evaluate both:

- correctness
- usefulness

because a technically correct answer can still be unhelpful.

## What to evaluate in RAG systems

- retrieval quality
- faithfulness to sources
- completeness
- latency
- refusal correctness

## Guardrails

Guardrails are controls around model behavior.

Examples:

- input filtering
- output moderation
- tool permission limits
- source-grounded answering
- policy checks

## Practical guardrail layers

- input moderation
- content classification
- tool permission boundaries
- prompt injection detection
- output moderation
- policy refusal templates
- human escalation

## Common mistakes

- measuring only "looks good to me"
- skipping retrieval evaluation
- using fine-tuning to solve a retrieval problem
- forgetting prompt injection defenses

## Important interview questions

- When should you fine-tune instead of using RAG?
- What does LoRA or PEFT solve?
- How would you evaluate factuality?
- What are the layers of guardrails in a production assistant?
- Why is human review still important?

## Quick revision

- evaluation is mandatory, not optional
- fine-tuning changes model behavior
- guardrails reduce misuse and unsafe output
