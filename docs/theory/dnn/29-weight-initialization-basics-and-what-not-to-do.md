---
id: weight-initialization-basics
title: "Weight initialization basics and what not to do"
sidebar_label: "29 · Weight Init Basics"
sidebar_position: 29
slug: /theory/dnn/weight-initialization-basics-and-what-not-to-do
description: "Why all-zeros initialization breaks symmetry, why large random values cause exploding activations, and the principle of variance preservation that motivates proper initialization."
tags: [initialization, symmetry-breaking, deep-learning, training]
---

# Weight initialization basics and what not to do

Before a neural network trains, its weights must be set to some initial values. The choice is consequential: bad initialization causes the forward pass to produce useless activations before training even begins, making learning impossible or extremely slow. This note explains the failure modes of naive initialization strategies and builds the motivation for Xavier and He initialization.

## One-line definition

Weight initialization is the setting of initial parameter values before training begins. Bad initialization causes vanishing activations, exploding activations, or symmetry — all of which prevent effective learning.

## Why this topic matters

With a good optimizer, learning rate, and architecture, a network with bad initialization may still fail to converge. The weight values in the first forward pass determine the scale of the gradients in the first backward pass — if that scale is wildly wrong, training may never recover. Proper initialization is the first line of defense against vanishing and exploding gradients.

## Failure mode 1: All-zeros initialization

Setting all weights to zero:

$$
W^{(l)} = 0 \quad \text{for all } l
$$

**Why this fails — the symmetry problem:**

All neurons in a given layer compute exactly the same output because they receive the same input and have the same weights:

![Weight initialization comparison — all-zeros (broken symmetry), random (proper variance), and bad variance scales showing how activations explode or vanish](https://commons.wikimedia.org/wiki/Special:Redirect/file/Artificial_neural_network.svg)
*Source: [Wikimedia Commons — Artificial Neural Network](https://commons.wikimedia.org/wiki/File:Artificial_neural_network.svg) (CC BY-SA 4.0)*

$$
z_i^{(l)} = \sum_j W_{ij}^{(l)} a_j^{(l-1)} = 0 \cdot a_j^{(l-1)} = 0 \text{ for all } i
$$

During backpropagation, all neurons receive the same gradient and update identically. After the update, all weights in the layer are still equal — just different from zero. All neurons in every hidden layer remain synchronized forever: they learn the same features, which is identical to having one neuron per layer. The network has essentially no hidden capacity.

**The symmetry must be broken** for neurons to learn different features.

## Failure mode 2: All-same nonzero value

Setting all weights to the same constant $c \neq 0$:

$$
W^{(l)}_{ij} = c
$$

Same symmetry problem as all-zeros. The weights and gradients remain identical across all neurons in each layer throughout training. The "hidden" layer is effectively a single neuron.

## Failure mode 3: Very small random values

Initializing from a distribution with very small variance, e.g.:

$$
W^{(l)} \sim \mathcal{N}(0, 0.001^2)
$$

In the forward pass, each pre-activation is a weighted sum:

$$
z^{(l)} = W^{(l)} a^{(l-1)}
$$

With $n_{in}$ input units and weights of variance $0.001^2$, the variance of the pre-activation is:

$$
\text{Var}(z^{(l)}) = n_{in} \cdot 0.001^2 \cdot \text{Var}(a^{(l-1)})
$$

For $n_{in} = 100$: $\text{Var}(z^{(l)}) = 100 \times 10^{-6} \times \text{Var}(a^{(l-1)}) = 10^{-4} \cdot \text{Var}(a^{(l-1)})$

After 10 layers, the activation variance is:

$$
\text{Var}(a^{(10)}) \approx (10^{-4})^{10} \cdot \text{Var}(a^{(0)}) = 10^{-40}
$$

Activations collapse to zero. Gradients also collapse to zero (vanishing gradients before any training occurs).

## Failure mode 4: Very large random values

Initializing with too-large variance, e.g.:

$$
W^{(l)} \sim \mathcal{N}(0, 1^2)
$$

With $n_{in} = 100$ and weight variance $1$:

$$
\text{Var}(z^{(l)}) = 100 \cdot 1 \cdot \text{Var}(a^{(l-1)}) = 100 \cdot \text{Var}(a^{(l-1)})
$$

After 10 layers: $\text{Var}(a^{(10)}) \approx 100^{10} = 10^{20}$

Activations explode to infinity. Gradients also explode. Training produces NaN loss.

## The Goldilocks problem

| Initialization | Activation variance | Result |
|---|---|---|
| All-zeros | 0 at every layer | Symmetry — no learning |
| Too small variance | Collapses to 0 | Vanishing activations and gradients |
| Too large variance | Grows to ∞ | Exploding activations and gradients |
| Just right | Stays ~constant | Gradients flow, training works |

The "just right" initialization is the variance that keeps activation variance stable across layers. This is exactly what Xavier and He initialization provide.

## The key insight for correct initialization

For a linear layer $z = Wx + b$ (ignoring bias):

$$
\text{Var}(z_j) = n_{in} \cdot \text{Var}(W) \cdot \text{Var}(x)
$$

To keep $\text{Var}(z) = \text{Var}(x)$ (stable variance across layers):

$$
n_{in} \cdot \text{Var}(W) = 1 \quad \Rightarrow \quad \text{Var}(W) = \frac{1}{n_{in}}
$$

This is the **LeCun initialization** for linear activations. Xavier adds a correction for the backward pass; He adds a correction for ReLU's zeroing of half the units.

## PyTorch example: observing the variance collapse

```python
import torch
import torch.nn as nn

def check_activations(weight_std, n_layers=10, n_units=256):
    """Measure activation variance through a deep network."""
    x = torch.randn(1, n_units)
    layers = []
    for _ in range(n_layers):
        W = torch.randn(n_units, n_units) * weight_std
        x = torch.relu(x @ W)
    return x.var().item()

# Too small
print(f"std=0.001: var = {check_activations(0.001):.2e}")  # ~0

# Too large
print(f"std=1.000: var = {check_activations(1.000):.2e}")  # ~1e+20 or inf

# He initialization: std = sqrt(2/n_in) = sqrt(2/256) ≈ 0.088
import math
he_std = math.sqrt(2.0 / 256)
print(f"He init:   var = {check_activations(he_std):.4f}")  # ~1.0
```

## Interview questions

<details>
<summary>Why does all-zeros initialization fail for neural networks?</summary>

All-zeros initialization creates perfect symmetry: all neurons in each layer compute the same pre-activation (zero), receive the same gradient, and update by the same amount. They remain identical throughout training. The network behaves as if each layer has one neuron — all hidden capacity is wasted. Random initialization breaks this symmetry so neurons can specialize to learn different features.
</details>

<details>
<summary>Why does initializing with very small variance cause problems?</summary>

With small weights, each layer multiplies the activation variance by n_in · Var(W) ≪ 1. After L layers, the variance is (n_in · Var(W))^L ≈ 0. Activations collapse to zero, and because the backward pass needs activation values, the gradients also vanish. The network is stuck before any learning occurs.
</details>

<details>
<summary>What is the correct principle for setting initial weight variance?</summary>

The weight variance should be set so that activation variance stays approximately constant across layers: Var(z^(l)) ≈ Var(a^(l-1)). From the forward pass equation Var(z) = n_in · Var(W) · Var(x), we need Var(W) = 1/n_in. This is the LeCun initialization for linear activations. Xavier adjusts for the backward pass by averaging n_in and n_out; He adjusts for ReLU's variance halving.
</details>

## Common mistakes

- Using `torch.zeros()` for weight initialization and wondering why the model does not learn — all-zeros is the most common beginner mistake.
- Copying a default initialization from one architecture to another without checking the activation function — He init for ReLU, Xavier for tanh/sigmoid.
- Not initializing biases — PyTorch initializes biases to a uniform distribution by default. Using `nn.init.zeros_(bias)` is often better for hidden layers.

## Advanced perspective

The initialization problem can be viewed as ensuring the network starts in a "trainable region" of the loss landscape. The mean-field theory of deep networks formalizes this: there is an ordered/chaotic phase transition. Networks in the ordered phase (small weights) have vanishing gradients; networks in the chaotic phase (large weights) have exploding gradients; networks at the critical point (correct initialization) have gradients that propagate well. Xavier and He initialization are principled ways to set the critical point.

## Final takeaway

The three failure modes — symmetry (all-zeros), vanishing (too small), exploding (too large) — each prevent training in a different way. The solution is to initialize weights randomly (breaking symmetry) from a distribution whose variance is calibrated to keep activation scale stable across layers. Xavier and He initialization do exactly this.

## References

- LeCun, Y., Bottou, L., Orr, G., & Müller, K.-R. (1998). Efficient BackProp.
- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.
