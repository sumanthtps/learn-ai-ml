---
id: mlp-notation
title: "MLP notation: inputs, weights, biases, layers, and shapes"
sidebar_label: "08 · MLP Notation"
sidebar_position: 8
slug: /theory/dnn/mlp-notation-inputs-weights-biases-layers-and-shapes
description: "The notation of multi-layer perceptrons: vectors, matrices, layer dimensions, batch dimension, and shape reasoning."
tags: [mlp, notation, tensors, shapes, matrices, ann]
---

# MLP notation: inputs, weights, biases, layers, and shapes

Once hidden layers are introduced ([Note 9](09-multi-layer-perceptron-intuition.md)), the next problem is notation. Without clear notation, forward propagation and backpropagation become confusing very quickly.

**Goal of this note**: Standardize the mathematical language so that when we write equations, we're all on the same page about what vectors, matrices, and dimensions mean.

## Continuity guide

**From**: [Note 7 — Perceptron Failure](07-why-a-single-perceptron-fails-on-nonlinear-problems.md) (why we need hidden layers)

**In this note**: **Notation and tensor shapes** — how to think about dimensions when data flows through layers

**Next**: [Note 9 — MLP Intuition](09-multi-layer-perceptron-intuition.md) (why hidden layers work) → [Note 10 — Forward Propagation](10-forward-propagation-and-how-a-neural-network-predicts.md) (step-by-step execution)

**Why this matters**: Forward pass → backpropagation both depend on getting dimensions right. A single mistake in shapes breaks the entire gradient computation.

## Single example notation

Let the input feature vector be:

$$
x \in \mathbb{R}^{d}
$$

For one hidden layer:

$$
z^{(1)} = W^{(1)}x + b^{(1)}
$$

$$
a^{(1)} = \phi(z^{(1)})
$$

$$
z^{(2)} = W^{(2)}a^{(1)} + b^{(2)}
$$

$$
\hat{y} = g(z^{(2)})
$$

where:

- $W^{(1)}$ maps input to hidden layer
- $W^{(2)}$ maps hidden layer to output
- $\phi$ is hidden-layer activation
- $g$ is output activation or identity, depending on the task

## Shape reasoning

Suppose:

- input dimension = 3
- hidden units = 4
- output units = 2

Then:

$$
x \in \mathbb{R}^{3 \times 1}
$$

$$
W^{(1)} \in \mathbb{R}^{4 \times 3}, \quad b^{(1)} \in \mathbb{R}^{4 \times 1}
$$

$$
a^{(1)} \in \mathbb{R}^{4 \times 1}
$$

$$
W^{(2)} \in \mathbb{R}^{2 \times 4}, \quad b^{(2)} \in \mathbb{R}^{2 \times 1}
$$

$$
\hat{y} \in \mathbb{R}^{2 \times 1}
$$

The general rule is:

$$
W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}
$$

where $n_l$ is the number of units in layer $l$.

## Batch dimension

In real training, we usually process many examples together.

If batch size is $m$, then:

$$
X \in \mathbb{R}^{m \times d}
$$

For a PyTorch linear layer with batch-first layout:

$$
Z^{(1)} = X(W^{(1)})^T + b^{(1)}
$$

So in practical code, shape conventions may differ slightly from handwritten notes, but the meaning is the same.

## Why notation matters for continuity

This note is the bridge between:

- the intuition of hidden layers
- the mechanics of forward propagation

If shapes are unclear here, later topics like backpropagation, CNNs, and transformers become harder because all of them rely on tensor-shape reasoning.

## A small PyTorch example

```python
import torch
import torch.nn as nn

X = torch.randn(5, 3)  # batch of 5, each with 3 features

layer1 = nn.Linear(3, 4)
layer2 = nn.Linear(4, 2)

h = layer1(X)
out = layer2(torch.relu(h))

print("X:", X.shape)
print("h:", h.shape)
print("out:", out.shape)
```

## Reading the shapes

- input: `(5, 3)`
- after first linear layer: `(5, 4)`
- after output layer: `(5, 2)`

The first dimension is batch size. The second dimension is feature size at that stage of the network.

## Interview questions

<details>
<summary>Why is the weight matrix shape <code>(out_features, in_features)</code> in PyTorch?</summary>

Because each output unit has one row of weights over all input features.
</details>

<details>
<summary>Why do we need bias vectors?</summary>

Bias shifts each unit independently and increases flexibility.
</details>

<details>
<summary>Why does this note come before forward propagation?</summary>

Because forward propagation is just repeated application of these matrix operations.
</details>

<details>
<summary>What is the difference between a single example and a batch in notation?</summary>

A single example is usually written as a vector, while a batch is a matrix whose first dimension represents the number of examples.
</details>

<details>
<summary>Why are tensor shapes so important in deep learning?</summary>

Because most implementation bugs come from shape mismatch, incorrect matrix multiplication, or misunderstanding what each dimension represents.
</details>

<details>
<summary>What does the weight-matrix shape rule mean?</summary>

It means the weight matrix for layer <code>l</code> maps from the previous layer with <code>n_(l-1)</code> units to the current layer with <code>n_l</code> units, so its shape is "current layer size by previous layer size".
</details>

## Final takeaway

MLP notation is the grammar of neural networks. Once you can read shapes and layer mappings, the rest of the theory becomes much easier to follow.

## References

- CampusX YouTube: MLP Notation
