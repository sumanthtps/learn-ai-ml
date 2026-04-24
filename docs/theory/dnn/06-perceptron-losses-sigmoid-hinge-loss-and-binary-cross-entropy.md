---
id: perceptron-losses
title: "Perceptron losses, sigmoid, hinge loss, and binary cross-entropy"
sidebar_label: "06 · Perceptron Losses"
sidebar_position: 6
slug: /theory/dnn/perceptron-losses-sigmoid-hinge-loss-and-binary-cross-entropy
description: "Why hard threshold training is limiting, and how sigmoid, hinge loss, and binary cross-entropy lead toward modern neural network training."
tags: [loss-functions, sigmoid, hinge-loss, binary-cross-entropy, perceptron]
---

# Perceptron losses, sigmoid, hinge loss, and binary cross-entropy

The perceptron update rule ([Note 5](05-perceptron-training-and-the-perceptron-trick.md)) is intuitive, but it has a limitation: it does not give a smooth notion of "how wrong" a prediction is. That is why loss functions matter.

**The problem**: Perceptron training uses hard 0/1 labels. The gradient is 0 if correct, undefined if incorrect. Optimization is jerky and slow.

**The solution**: Use smooth loss functions that penalize "confidence" in wrong predictions. Enter sigmoid, hinge loss, and binary cross-entropy.

## Continuity guide

**From**: [Note 5 — Perceptron Training](05-perceptron-training-and-the-perceptron-trick.md) (how the perceptron trick updates weights)

**In this note**: Why hard ±1 training is limited; introduce sigmoid activation and **smooth loss functions** (BCE, hinge)

**Next**: [Note 7 — Why Perceptron Fails](07-why-a-single-perceptron-fails-on-nonlinear-problems.md) (show XOR problem)

**Then**: [Note 8 — MLP Notation](08-mlp-notation-inputs-weights-biases-layers-and-shapes.md) (formalize multi-layer network notation)

**Later**: [Note 14 — Loss Functions](14-loss-functions-in-deep-learning.md) revisits all loss functions with full mathematical rigor

## Why we need a loss

Training becomes easier when we can assign a numerical penalty to bad predictions.

Instead of only saying:

- correct
- incorrect

we want to say:

- slightly wrong
- very wrong

This leads to differentiable optimization later.

## Hard threshold problem

A step function produces a hard output:

$$
\hat{y} =
\begin{cases}
1 & \text{if } z \ge 0 \\
0 & \text{if } z < 0
\end{cases}
$$

The issue is that this function is not useful for gradient-based training because it is not smoothly differentiable.

That motivates sigmoid.

## Sigmoid function

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Properties:

- output lies in $(0,1)$
- can be interpreted as a probability in binary classification
- differentiable

Derivative:

$$
\sigma'(z) = \sigma(z)(1-\sigma(z))
$$

This formula appears again during backpropagation.

## Hinge loss

For binary labels:

$$
y \in \{-1, +1\}
$$

and score:

$$
f(x)=w^T x + b
$$

hinge loss is:

$$
\mathcal{L}_{\text{hinge}} = \max(0, 1 - y f(x))
$$

Interpretation:

- if the point is correctly classified with enough margin, loss is zero
- if not, the model is penalized

This is closely connected to support vector machines.

## Binary cross-entropy

If:

$$
p = \sigma(z)
$$

then binary cross-entropy is:

$$
\mathcal{L}_{\text{BCE}} = -\left[y \log p + (1-y)\log(1-p)\right]
$$

for:

$$
y \in \{0,1\}
$$

### Why BCE is important

It strongly penalizes confident wrong predictions.

Examples:

- true label `1`, predicted probability `0.9` -> small loss
- true label `1`, predicted probability `0.01` -> very large loss

This makes it a strong fit for probabilistic binary classification.

## Continuity with the next notes

These ideas connect directly:

- perceptron training gave a mistake-driven update
- sigmoid gives a smooth output
- BCE gives a smooth objective
- backpropagation later uses these smooth functions to compute gradients

This is the transition from classical perceptron learning to modern neural-network training.

## Comparison table

| Concept | Output type | Typical labels | Main use |
| --- | --- | --- | --- |
| Step threshold | hard class | `{0,1}` or `{-1,+1}` | classical perceptron |
| Sigmoid | probability-like | `{0,1}` | binary classification |
| Hinge loss | margin penalty | `{-1,+1}` | SVM-style training |
| BCE | probabilistic loss | `{0,1}` | modern binary classification |

## PyTorch example

```python
import torch
import torch.nn.functional as F

z = torch.tensor([-2.0, 0.0, 2.0])
p = torch.sigmoid(z)
print("sigmoid:", p)

target = torch.tensor([0.0, 1.0, 1.0])
bce = F.binary_cross_entropy(p, target)
print("BCE:", float(bce))

y_pm = torch.tensor([-1.0, 1.0, 1.0])
hinge = torch.clamp(1 - y_pm * z, min=0).mean()
print("Hinge:", float(hinge))
```

## Interview questions

<details>
<summary>Why not train directly with the hard threshold?</summary>

Because it does not provide a useful smooth gradient for optimization.
</details>

<details>
<summary>Why is sigmoid paired with BCE so often?</summary>

Because the mathematics becomes clean and the loss has a probabilistic interpretation.
</details>

<details>
<summary>Is hinge loss used in neural networks?</summary>

It can be, but BCE and cross-entropy are more common in standard deep learning.
</details>

<details>
<summary>What is the practical difference between hinge loss and BCE?</summary>

Hinge loss focuses on margin-based separation, while BCE treats the output probabilistically and penalizes confident wrong predictions strongly.
</details>

<details>
<summary>Why is sigmoid important historically in deep learning?</summary>

Because it replaces the hard threshold with a smooth differentiable output, making gradient-based learning possible.
</details>

<details>
<summary>Why does BCE become very large for confident wrong predictions?</summary>

Because the logarithm heavily penalizes predicted probabilities near 0 when the true label is 1, or near 1 when the true label is 0.
</details>

## Final takeaway

Loss functions turn classification from a yes/no decision rule into an optimization problem. That shift is essential for the rest of deep learning.

## References

- CampusX YouTube: Perceptron Loss Function | Hinge Loss | Binary Cross Entropy | Sigmoid Function
