---
id: xavier-he-initialization
title: "Xavier/Glorot and He initialization"
sidebar_label: "30 · Xavier & He Init"
sidebar_position: 30
slug: /theory/dnn/xavier-glorot-and-he-initialization
description: "Xavier initialization preserves gradient variance for tanh/sigmoid networks; He initialization corrects this for ReLU networks — both derived from the principle of maintaining signal variance across layers."
tags: [initialization, xavier, glorot, he, deep-learning]
---

# Xavier/Glorot and He initialization

Note 29 explained why naive initialization fails: all-zeros causes symmetry, too-large values explode, too-small values vanish. Xavier and He initialization are the principled solutions — they set the weight variance so that the signal neither grows nor shrinks as it passes through the layers.

## One-line definition

Xavier initialization sets weight variance to $2/(n_{in} + n_{out})$ to preserve activation variance through symmetric activations (sigmoid, tanh); He initialization sets it to $2/n_{in}$ to account for the fact that ReLU kills half of all pre-activations.

## Why this topic matters

The right weight initialization enables training of networks with tens or hundreds of layers. The wrong initialization causes vanishing or exploding activations in the first forward pass, before any training has occurred. Xavier and He initialization are the defaults in essentially every modern deep learning framework — understanding their derivation tells you when to switch from one to the other.

## The core principle: preserve variance

The goal is that after layer $l$, the variance of activations should be approximately the same as before. If it grows: exploding activations. If it shrinks: vanishing activations.

![Xavier vs He initialization — Xavier preserves variance through symmetric activation functions; He compensates for ReLU's zero-killing property](https://commons.wikimedia.org/wiki/Special:Redirect/file/Artificial_neural_network.svg)
*Source: [Wikimedia Commons — Artificial Neural Network](https://commons.wikimedia.org/wiki/File:Artificial_neural_network.svg) (CC BY-SA 4.0)*

For a linear layer $z = Wx + b$ (ignoring bias):

$$
\text{Var}(z_j) = n_{in} \cdot \text{Var}(W) \cdot \text{Var}(x)
$$

This comes from the fact that $z_j = \sum_{i=1}^{n_{in}} W_{ji} x_i$ and, assuming $W$ and $x$ are independent with zero mean:

$$
\text{Var}(z_j) = \sum_{i=1}^{n_{in}} \text{Var}(W_{ji}) \cdot \text{Var}(x_i) = n_{in} \cdot \text{Var}(W) \cdot \text{Var}(x)
$$

To keep $\text{Var}(z) = \text{Var}(x)$, we need:

$$
\text{Var}(W) = \frac{1}{n_{in}}
$$

A symmetric condition (preserve variance in both forward and backward passes) gives:

$$
\text{Var}(W) = \frac{2}{n_{in} + n_{out}}
$$

## Xavier/Glorot initialization

Derived by Glorot & Bengio (2010) for activations that are linear near zero (sigmoid, tanh):

$$
W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}},\ \sqrt{\frac{6}{n_{in} + n_{out}}}\right)
$$

or equivalently, Gaussian form:

$$
W \sim \mathcal{N}\left(0,\ \frac{2}{n_{in} + n_{out}}\right)
$$

The uniform form is the standard (used in PyTorch as `xavier_uniform_`). The constant 6 in the uniform version comes from the fact that $\text{Var}(\mathcal{U}(-a, a)) = a^2/3$.

**Derivation intuition**: With tanh activations, $\tanh'(0) = 1$, so the activation is approximately linear near zero and the variance analysis applies directly. Setting $\text{Var}(W) = 2/(n_{in} + n_{out})$ balances the forward and backward signal variance simultaneously.

## He initialization

ReLU sets negative pre-activations to zero. This kills exactly half the neurons (assuming symmetric input distribution), which halves the effective variance.

To compensate for this halving:

$$
\text{Var}(W) = \frac{2}{n_{in}}
$$

The factor of 2 corrects for the ReLU half-zeroing. Distribution forms:

$$
W \sim \mathcal{N}\left(0,\ \sqrt{\frac{2}{n_{in}}}\right) \quad \text{(standard: He normal)}
$$

$$
W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}}},\ \sqrt{\frac{6}{n_{in}}}\right) \quad \text{(He uniform)}
$$

**When to use He**: Any layer followed by ReLU or Leaky ReLU. If you use ReLU throughout, He normal is the default.

## Which initialization to use

| Activation | Recommended init |
|---|---|
| Sigmoid | Xavier (Glorot) |
| Tanh | Xavier (Glorot) |
| ReLU | He normal |
| Leaky ReLU | He normal (with modified factor) |
| SELU | LeCun normal ($1/n_{in}$) |
| Linear (output) | Xavier |

## PyTorch example

```python
import torch
import torch.nn as nn

# PyTorch default initialization for Linear layers uses:
# weights ~ Kaiming uniform (He uniform), biases ~ Uniform[-1/sqrt(fan_in), 1/sqrt(fan_in)]

# Manual initialization examples
layer = nn.Linear(256, 128)

# Xavier uniform (for tanh/sigmoid)
nn.init.xavier_uniform_(layer.weight)
nn.init.zeros_(layer.bias)

# Xavier normal
nn.init.xavier_normal_(layer.weight)

# He normal (for ReLU)
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# He uniform
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')

# Initializing a full model
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

model = nn.Sequential(
    nn.Linear(64, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10)
)
model.apply(init_weights)
```

## Empirical validation: what happens without proper init

A simple experiment shows the importance:

```python
import torch
import torch.nn as nn

def measure_activation_variance(init_std):
    layers = [nn.Linear(256, 256) for _ in range(10)]
    for layer in layers:
        nn.init.normal_(layer.weight, std=init_std)
    
    x = torch.randn(32, 256)
    for layer in layers:
        x = torch.relu(layer(x))
    return x.var().item()

print(f"std=0.01 (too small): variance = {measure_activation_variance(0.01):.6f}")  # near 0
print(f"std=1.00 (too large): variance = {measure_activation_variance(1.00):.2f}")   # huge
print(f"He init:              variance = {measure_activation_variance(0.0627):.4f}") # ~1
# std = sqrt(2/256) ≈ 0.0884 for He; in practice depends on fan_in
```

## Interview questions

<details>
<summary>Why does Xavier initialization use n_in + n_out in the denominator?</summary>

The denominator n_in + n_out is the harmonic compromise between two conditions: (1) preserving variance in the forward pass requires Var(W) = 1/n_in, and (2) preserving variance in the backward pass requires Var(W) = 1/n_out. Xavier takes the average of these two conditions: Var(W) = 2/(n_in + n_out). This keeps the signal stable in both directions during the first epoch.
</details>

<details>
<summary>Why does He initialization use only n_in (not n_in + n_out)?</summary>

He initialization corrects for ReLU zeroing approximately half of all pre-activations. The factor 2 in 2/n_in compensates for this halving of variance. The backward pass correction (1/n_out) is omitted because the gradient through a ReLU layer also has its variance affected by the same factor, making the forward-pass correction approximately sufficient for both directions.
</details>

<details>
<summary>What is the difference between fan_in and fan_out mode in He initialization?</summary>

fan_in mode (Var = 2/n_in) is designed to preserve variance in the forward pass — useful for networks used for prediction. fan_out mode (Var = 2/n_out) is designed to preserve variance in the backward pass — sometimes preferred for networks used as discriminators. The default and most common choice is fan_in.
</details>

<details>
<summary>What initialization does PyTorch use by default for Linear and Conv2d layers?</summary>

PyTorch uses Kaiming uniform (He uniform) as the default for both Linear and Conv2d layers: weights ~ Uniform(-sqrt(6/fan_in), sqrt(6/fan_in)), biases ~ Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)). This is suitable for ReLU networks. For tanh/sigmoid networks, you should explicitly apply xavier_uniform_ or xavier_normal_.
</details>

## Common mistakes

- Using Xavier initialization with ReLU — the correction factor of 2 for the ReLU half-zeroing is missing, causing vanishing activations in deep ReLU networks.
- Using He initialization with sigmoid/tanh — the factor 2 overcorrects for activations that do not zero half the units.
- Not resetting initialization after changing the activation function — the init is tightly coupled to the activation.
- Forgetting that PyTorch's default is Kaiming uniform (He), which is correct for ReLU but may not be what you want for sigmoid/tanh output layers.

## Advanced perspective

Xavier and He initialization can be derived more rigorously using the theory of mean field propagation in neural networks. The "ordered/chaotic phase transition" framework shows that networks initialized at the boundary between ordered and chaotic behavior (the "edge of chaos") have the deepest gradient flow. Xavier and He initialization correspond to initializing at this boundary for their respective activation functions. This perspective also motivates the LeCun (1/n_in) initialization for SELU and the specialized initialization schemes for attention layers in transformers.

## Final takeaway

Xavier for sigmoid/tanh networks, He for ReLU networks. Both are derived from the same principle: set weight variance so that the signal variance neither grows nor shrinks as it passes through a layer. Getting initialization right is a prerequisite for training deep networks, and PyTorch's defaults (He) are correct for ReLU but require manual override for other activation choices.

## References

- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. AISTATS.
- He, K., et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. ICCV.
