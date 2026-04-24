---
id: regularization-l1-l2
title: "Regularization, weight decay, L1, and L2 in neural networks"
sidebar_label: "26 · L1 / L2 Regularization"
sidebar_position: 26
slug: /theory/dnn/regularization-weight-decay-l1-and-l2-in-neural-networks
description: "L1 and L2 regularization add penalty terms to the loss function to discourage large weights, with L2 producing weight decay and L1 producing sparsity."
tags: [regularization, l1, l2, weight-decay, overfitting, deep-learning]
---

# Regularization, weight decay, L1, and L2 in neural networks

When a neural network memorizes training data rather than learning generalizable patterns, it overfits. Regularization is the family of techniques that discourage overfitting by constraining the model's complexity. L1 and L2 regularization do this by adding a penalty on weight magnitude directly into the loss function.

## One-line definition

L1 and L2 regularization add penalty terms proportional to weight magnitude ($\sum |w|$ for L1, $\sum w^2$ for L2) to the training loss, discouraging large weights and reducing overfitting.

![Neural network with weights — L1/L2 regularization penalizes all weights in the network; L2 shrinks weights toward zero while L1 drives many weights to exactly zero (sparsity)](https://commons.wikimedia.org/wiki/Special:Redirect/file/MultiLayerPerceptron.svg)
*Source: [Wikimedia Commons — MultiLayerPerceptron](https://commons.wikimedia.org/wiki/File:MultiLayerPerceptron.svg) (CC BY-SA 4.0)*

## Why this topic matters

A model with unconstrained weights can perfectly fit training data by making weights arbitrarily large for the few features that happen to correlate with labels in the training set. Regularization penalizes this — the model pays a cost for large weights, so it only uses large weights when they produce a large reduction in the data loss. This encourages simpler, more generalizable representations.

## The regularized loss function

General form:

$$
\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{data}}(\hat{y}, y) + \lambda \cdot \Omega(W)
$$

where:
- $\mathcal{L}_{\text{data}}$: original task loss (BCE, cross-entropy, MSE, etc.)
- $\Omega(W)$: regularization penalty on weights
- $\lambda$: regularization strength (hyperparameter)

## L2 regularization (Ridge / Weight Decay)

$$
\Omega(W) = \|W\|_2^2 = \sum_{i,j} W_{ij}^2
$$

The gradient of the L2 penalty with respect to $W$:

$$
\frac{\partial}{\partial W} \lambda \|W\|_2^2 = 2\lambda W
$$

Adding this to the weight gradient changes the update rule:

$$
W \leftarrow W - \eta \frac{\partial \mathcal{L}_{\text{data}}}{\partial W} - 2\eta\lambda W
$$

$$
W \leftarrow W(1 - 2\eta\lambda) - \eta \frac{\partial \mathcal{L}_{\text{data}}}{\partial W}
$$

The factor $(1 - 2\eta\lambda)$ multiplies the weight before each update — this is **weight decay**: weights are pulled toward zero at every step, regardless of the gradient.

**Properties of L2:**
- Differentiable everywhere — smooth optimization
- Penalizes large weights quadratically — small weights are barely penalized, large weights are strongly penalized
- Does NOT produce sparse solutions — all weights are shrunk but none are exactly zeroed
- Equivalent to a Gaussian prior on weights (Bayesian perspective)

## L1 regularization (Lasso)

$$
\Omega(W) = \|W\|_1 = \sum_{i,j} |W_{ij}|
$$

The gradient of the L1 penalty (subgradient):

$$
\frac{\partial}{\partial W} \lambda \|W\|_1 = \lambda \cdot \text{sign}(W)
$$

Update rule:

$$
W \leftarrow W - \eta \frac{\partial \mathcal{L}_{\text{data}}}{\partial W} - \eta\lambda \cdot \text{sign}(W)
$$

**Properties of L1:**
- Pushes weights toward zero by a constant amount ($\lambda$) rather than proportionally
- **Produces sparse solutions**: weights near zero get pushed exactly to zero (soft thresholding effect)
- Not differentiable at $W = 0$ — requires subgradient
- Equivalent to a Laplace prior on weights (Bayesian perspective)

## Why L1 produces sparsity but L2 does not

The geometric intuition:

$$\text{L1 penalty ball} = \{W : \|W\|_1 \leq C\} \quad \text{(diamond/rhombus shape)}$$

$$\text{L2 penalty ball} = \{W : \|W\|_2^2 \leq C\} \quad \text{(sphere shape)}$$

When the loss function's level curves intersect the constraint region, they tend to touch the L1 diamond at a corner (where some weights are exactly 0), but touch the L2 sphere at a smooth point (where all weights are nonzero but small). This is why L1 produces sparsity and L2 does not.

## Comparison

| | L1 | L2 |
|---|---|---|
| Penalty | $\lambda \sum \lvert W_{ij} \rvert$ | $\lambda \sum W_{ij}^2$ |
| Gradient | $\lambda \cdot \text{sign}(W)$ | $2\lambda W$ |
| Sparsity | Yes — drives weights to 0 | No — shrinks weights uniformly |
| Differentiability | No (at $W=0$) | Yes everywhere |
| Effect on small weights | Same as large weights | Small weights barely penalized |
| Use case | Feature selection, sparse models | General overfitting control |

## PyTorch example

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(64, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10)
)

# L2 regularization via optimizer weight_decay
# PyTorch's weight_decay IS L2 regularization (adds λ||W||² to the loss)
optimizer_l2 = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4   # λ for L2
)

# L1 regularization — must be added manually to the loss
def l1_loss(model, lambda_l1=1e-4):
    l1 = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1

criterion = nn.CrossEntropyLoss()
x = torch.randn(32, 64)
y = torch.randint(0, 10, (32,))

optimizer_l2.zero_grad()
loss = criterion(model(x), y) + l1_loss(model)  # L1 added manually
loss.backward()
optimizer_l2.step()
```

**Note on `weight_decay` in Adam vs AdamW**: In `torch.optim.Adam`, `weight_decay` couples the L2 penalty with the adaptive scaling (incorrect). In `torch.optim.AdamW`, weight decay is applied directly after the adaptive update (correct). Always use `AdamW` when applying L2 regularization with Adam-based optimizers.

## Choosing $\lambda$

- Start with $\lambda = 10^{-4}$ as a default
- Increase $\lambda$ if validation loss is much higher than training loss (overfitting)
- Decrease $\lambda$ if training loss is high (underfitting)
- $\lambda$ is tuned on the validation set, not the test set

## Interview questions

<details>
<summary>What is the mathematical effect of L2 regularization on the weight update?</summary>

L2 regularization adds a penalty term proportional to the sum of squared weights, so its gradient contribution is $2 \lambda W$. This modifies the update rule to:

$$
W \leftarrow W(1 - 2\eta\lambda) - \eta \frac{\partial \mathcal{L}_{\text{data}}}{\partial W}
$$

The factor $(1 - 2\eta\lambda)$ multiplies the weight before each update, causing all weights to decay toward zero at every step. That is why L2 regularization is also called weight decay. The learning rate $\eta$ and regularization strength $\lambda$ together determine how fast weights shrink.
</details>

<details>
<summary>Why does L1 regularization produce sparse weights but L2 does not?</summary>

L1 applies a constant gradient of $\lambda \cdot \mathrm{sign}(W)$ regardless of weight magnitude, so it pushes every weight toward zero by the same fixed amount at each step. This can drive small weights exactly to zero. L2 uses a gradient of $2\lambda W$, which is proportional to the weight magnitude. Large weights receive large penalties, while small weights receive tiny penalties and usually do not reach exactly zero. That is why L1 produces sparse models, while L2 produces small but nonzero weights.
</details>

<details>
<summary>What is the difference between weight_decay in Adam and AdamW?</summary>

In `torch.optim.Adam`, `weight_decay` adds $\lambda W$ to the gradient before the adaptive update, so the decay is scaled by the optimizer's second-moment normalization. This weakens and distorts the regularization effect. In `AdamW`, weight decay is applied directly to the parameter after the adaptive step:

$$
\theta \leftarrow \theta - \eta \frac{\hat{m}}{\sqrt{\hat{v}}} - \eta \lambda \theta
$$

Because this decay is decoupled from adaptive scaling, AdamW applies a more faithful and predictable regularization strength. That is why AdamW is the right default for transformer training.
</details>

## Common mistakes

- Using both L1 and L2 in the same model without understanding the joint effect (L1 + L2 is called elastic net regularization).
- Setting $\lambda$ too high — both train and validation loss become high (the penalty dominates the data loss, underfitting).
- Regularizing the bias terms — biases are typically not regularized because they do not contribute to the complexity of the function class the way weights do.
- Using Adam's `weight_decay` instead of AdamW — the coupling is incorrect.

## Advanced perspective

L1 and L2 regularization have Bayesian interpretations: L2 is equivalent to placing a Gaussian prior on weights ($p(W) \propto e^{-\lambda\|W\|^2}$), and L1 is equivalent to a Laplace prior ($p(W) \propto e^{-\lambda\|W\|_1}$). Maximizing the posterior (MAP estimation) with these priors is exactly equivalent to minimizing the regularized loss. This perspective explains the sparsity property of L1: the Laplace prior has sharp peaks at zero (high probability mass), so the MAP solution concentrates weights at zero.

## Final takeaway

L2 regularization shrinks all weights smoothly and is the standard tool for controlling overfitting in neural networks. L1 regularization produces sparse models by driving some weights exactly to zero. Use L2 (via `weight_decay` in AdamW) as the default, and L1 only when sparse weight patterns are desired or interpretable.

## References

- Ng, A. Y. (2004). Feature selection, L1 vs. L2 regularization, and rotational invariance. ICML.
- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso.
