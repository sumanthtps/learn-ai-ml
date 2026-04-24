---
id: vanishing-exploding-gradients
title: "Vanishing gradients and exploding gradients"
sidebar_label: "18 · Vanishing & Exploding Gradients"
sidebar_position: 18
slug: /theory/dnn/vanishing-gradients-and-exploding-gradients
description: "How gradients shrink to zero or grow unboundedly during backpropagation in deep networks, why this prevents learning, and the solutions that made training deep networks practical."
tags: [vanishing-gradients, exploding-gradients, backpropagation, relu, deep-learning]
---

# Vanishing gradients and exploding gradients

Backpropagation computes gradients by multiplying local derivatives layer by layer, from output to input. In deep networks, this chain of multiplications can go catastrophically wrong: gradients either shrink exponentially toward zero (vanishing) or grow exponentially toward infinity (exploding). Both prevent effective learning.

## One-line definition

Vanishing gradients occur when repeated multiplication of small derivatives during backpropagation makes the gradient at early layers approach zero. Exploding gradients occur when repeated multiplication of large derivatives makes the gradient grow unboundedly.

![Simple RNN module — a single tanh layer creates the vanishing gradient problem; every backprop step multiplies by tanh′ ∈ (0,1) making gradients shrink exponentially through time](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png)
*Source: [Colah's Blog — Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (CC BY 4.0)*

## Why this topic matters

Vanishing and exploding gradients are the primary reason deep networks (more than ~4 layers) were nearly untrainable before 2010. They are also why RNNs struggle with long sequences. Understanding these problems explains the motivation for ReLU activations, careful weight initialization, gradient clipping, batch normalization, and residual connections — all of which are direct responses to this instability.

## How gradients flow through layers

For an $L$-layer network with activations $a^{(l)} = \phi(z^{(l)})$, the gradient of the loss w.r.t. the weights in layer $l$ requires multiplying local derivatives from layer $L$ back to layer $l$:

$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(L)} \cdot \prod_{k=l+1}^{L} \left( (W^{(k)})^T \cdot \text{diag}(\phi'(z^{(k-1)})) \right) \cdot a^{(l-1)T}
$$

The product in the middle is a chain of $L - l$ matrix multiplications.

## The vanishing gradient mechanism

The sigmoid activation and its derivative are:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

The maximum value of $\sigma'(z)$ is $0.25$ (at $z = 0$). For inputs that are not near zero, $\sigma'(z)$ is much smaller.

At each layer, the gradient is multiplied by $\sigma'(z^{(l)})$. For a 10-layer network with sigmoid activations:

$$
\prod_{l=1}^{10} \sigma'(z^{(l)}) \leq 0.25^{10} = 9.5 \times 10^{-7}
$$

The gradient at the first layer is effectively zero. Parameters in early layers receive no meaningful learning signal.

**Tanh** is centered at zero (unlike sigmoid) but still saturates: $\text{tanh}'(z) \leq 1$, and for large $|z|$, $\text{tanh}'(z) \to 0$. It suffers the same vanishing problem, though to a lesser degree.

## The exploding gradient mechanism

If weight matrices have large eigenvalues, the reverse happens. Each multiplication amplifies the gradient:

$$
\prod_{l=1}^{L} \|W^{(l)}\| \gg 1 \Longrightarrow \text{gradient grows exponentially}
$$

For a 10-layer network with weights whose spectral norm is 1.5 per layer:

$$
1.5^{10} \approx 57
$$

The gradient at the first layer is 57 times larger than at the output. This causes large parameter updates, numerical instability, and NaN loss.

Exploding gradients are more common in RNNs (where the same weight matrix $W_h$ is multiplied $T$ times for a sequence of length $T$) than in feedforward networks.

## The signal-to-noise analogy

Think of the gradient as a message passing backward through the network. At each layer, the message is multiplied by a local gain factor:

$$
\text{gain} = \|W^{(l)}\| \cdot |\phi'(z^{(l)})|
$$

- If the average gain $< 1$: signal decays exponentially — vanishing
- If the average gain $> 1$: signal grows exponentially — exploding
- If the average gain $\approx 1$: signal propagates well

Good initialization, activation choice, and normalization all aim to keep this gain near 1.

## Solutions

### 1. ReLU and its variants

ReLU has a derivative of 1 for positive inputs and 0 for negative inputs:

$$
\text{ReLU}(z) = \max(0, z), \qquad \frac{d}{dz}\text{ReLU}(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}
$$

For positive activations, the gradient passes through without shrinking. This makes deep networks with ReLU much easier to train than those with sigmoid. The downside is the **dying ReLU problem**: neurons with negative pre-activations get zero gradient and stop learning permanently.

### 2. Careful weight initialization

Xavier/Glorot initialization sets the weight variance to:

$$
\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

He initialization (for ReLU) sets it to:

$$
\text{Var}(W) = \frac{2}{n_{\text{in}}}
$$

Both aim to preserve the variance of activations and gradients across layers, keeping the average gain near 1.

### 3. Gradient clipping (for exploding)

Cap the gradient norm at a threshold $\tau$ before the parameter update:

$$
\text{if } \|\nabla \mathcal{L}\| > \tau: \quad \nabla \mathcal{L} \leftarrow \tau \cdot \frac{\nabla \mathcal{L}}{\|\nabla \mathcal{L}\|}
$$

This prevents exploding gradients without eliminating the gradient direction. It is the standard tool for RNN and transformer training.

```python
# PyTorch gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4. Batch normalization

Normalizes activations at each layer to have zero mean and unit variance:

$$
\hat{z}^{(l)} = \frac{z^{(l)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

By keeping activations in a stable range, batch normalization prevents the saturation that causes vanishing gradients with sigmoid/tanh.

### 5. Residual connections (ResNets)

Add a skip connection that bypasses each block:

$$
h^{(l+1)} = F(h^{(l)}) + h^{(l)}
$$

The gradient of the loss w.r.t. $h^{(l)}$ is:

$$
\frac{\partial \mathcal{L}}{\partial h^{(l)}} = \frac{\partial \mathcal{L}}{\partial h^{(l+1)}} \cdot \left(F'(h^{(l)}) + 1\right)
$$

The $+1$ term provides a direct gradient highway that does not shrink, even if $F'$ is small. This is what makes 50, 100, and even 1000-layer networks trainable.

## PyTorch diagnostic

```python
import torch
import torch.nn as nn

# Build a deep sigmoid network to observe vanishing gradients
model = nn.Sequential(*[
    nn.Sequential(nn.Linear(64, 64), nn.Sigmoid())
    for _ in range(10)
])

x = torch.randn(32, 64)
y = torch.randn(32, 64)
loss = nn.MSELoss()(model(x), y)
loss.backward()

# Check gradient norms layer by layer
for i, layer in enumerate(model):
    grad_norm = layer[0].weight.grad.norm().item()
    print(f"Layer {i} gradient norm: {grad_norm:.6f}")
# Observe: norms shrink toward zero in early layers
```

## Interview questions

<details>
<summary>Why does sigmoid cause vanishing gradients but ReLU largely avoids it?</summary>

The sigmoid derivative is bounded above by $0.25$. Multiplying this through 10 layers gives a factor of at most $0.25^{10} \approx 10^{-6}$. The gradient at the first layer is essentially zero. ReLU's derivative is exactly $1$ for positive activations, so the gradient passes through without attenuation. This one property made deep networks trainable.
</details>

<details>
<summary>Why are exploding gradients especially problematic in RNNs?</summary>

In an RNN, the same weight matrix $W_h$ is applied $T$ times for a sequence of length $T$. Backpropagation through time multiplies the gradient by this recurrent transition repeatedly. If the spectral norm of $W_h$ is greater than $1$, the gradient grows exponentially in $T$. For sequences of length $100$, a matrix with norm $1.1$ gives roughly $1.1^{100} \approx 13{,}780$, which is catastrophically large. This is why gradient clipping is essential for RNN training.
</details>

<details>
<summary>What does gradient clipping do, and why does it not harm learning?</summary>

Gradient clipping rescales the gradient vector to have norm at most τ when the norm exceeds τ. It preserves the gradient direction — only the magnitude is changed. This prevents the parameter update from taking an enormous step due to an exploding gradient, without eliminating the useful directional information. The threshold τ = 1.0 is a common default.
</details>

<details>
<summary>How do residual connections solve the vanishing gradient problem?</summary>

A residual connection adds the input directly to the output:

$$
h_{l+1} = F(h_l) + h_l
$$

The gradient through this block becomes:

$$
\frac{\partial L}{\partial h_l} = \frac{\partial L}{\partial h_{l+1}} \cdot \left(F'(h_l) + 1\right)
$$

The $+1$ term ensures there is always a gradient path that does not shrink, even if $F'(h_l)$ is very small. This highway for gradients allows networks with hundreds of layers to be trained.
</details>

## Common mistakes

- Assuming vanishing gradients only affect the first layer — they affect all layers before the last, with earlier layers worse.
- Using sigmoid in hidden layers of deep networks — modern practice uses ReLU or its variants.
- Applying gradient clipping to the entire model when only one layer is exploding — diagnose which layer before applying global clipping.
- Confusing gradient clipping (rescaling the gradient vector) with gradient truncation (zeroing gradient values above a threshold).

## Advanced perspective

The spectral radius of the Jacobian of each layer determines the long-run behavior of gradient flow. If the product of spectral radii is $< 1$, gradients vanish; if $> 1$, they explode. Batch normalization effectively constrains the spectral structure of activations. Careful initialization (Xavier, He) sets the initial spectral radii to have product close to 1. Residual connections bypass the Jacobian entirely for the skip path. Each of these solutions targets the same root cause: controlling the eigenvalue spectrum of the backward pass.

## Final takeaway

Vanishing and exploding gradients are the same problem from two sides: the product of local gradient factors along the backward path diverges from 1. Every major training stabilization technique — ReLU, initialization, batch normalization, residual connections, gradient clipping — is a solution to this one root cause.

## References

- Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult.
- He, K., et al. (2016). Deep Residual Learning for Image Recognition.
