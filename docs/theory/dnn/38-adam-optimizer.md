---
id: adam-optimizer
title: "Adam optimizer"
sidebar_label: "38 · Adam"
sidebar_position: 38
slug: /theory/dnn/adam-optimizer
description: "Adam combines SGD momentum (first moment) with RMSProp adaptive scaling (second moment) plus bias correction, making it the default optimizer for most deep learning tasks."
tags: [adam, optimizers, momentum, adaptive-learning-rate, deep-learning]
---

# Adam optimizer

Adam (Adaptive Moment Estimation) is the most widely used optimizer in deep learning. It combines the directional smoothing of momentum with the per-parameter adaptation of RMSProp, and adds bias correction to handle the cold-start initialization problem. The result is an optimizer that works well across a wide variety of tasks without heavy tuning.

## One-line definition

Adam maintains an exponentially weighted moving average of both the gradient (first moment) and the squared gradient (second moment), applies bias correction to both, and uses them to compute an adaptive, momentum-corrected update for each parameter.

![SGD optimization path — different optimizers (SGD, momentum, Adam) take different trajectories through the loss landscape; Adam converges faster and more reliably than plain SGD](https://upload.wikimedia.org/wikipedia/commons/f/f3/Stogra.png)
*Source: [Wikimedia Commons — Stochastic gradient](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (CC BY-SA 3.0)*

## Why this topic matters

Before Adam, practitioners had to choose between momentum (good for direction) and adaptive methods (good for scaling), or tune complex combinations manually. Adam unifies both ideas in one algorithm that is robust to gradient scale differences across layers and works well out of the box. It is the default optimizer for transformers, RNNs, GANs, and most NLP tasks.

## The two statistics Adam tracks per parameter

**First moment** — EWMA of gradients (captures direction with momentum):

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

**Second moment** — EWMA of squared gradients (captures per-parameter scale):

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

where $g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$ is the gradient at step $t$.

Default values: $\beta_1 = 0.9$, $\beta_2 = 0.999$.

## Bias correction

Both moments are initialized at zero. Without correction, early estimates are biased toward zero (EWMA with zero initialization). The corrected estimates are:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

The denominator grows from $(1 - \beta)$ at $t=1$ toward $1$ as $t \to \infty$, scaling up early estimates and then fading away.

## The parameter update

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
$$

- $\eta$: global learning rate (default: $10^{-3}$)
- $\hat{m}_t$: bias-corrected first moment — provides momentum direction
- $\sqrt{\hat{v}_t}$: root of bias-corrected second moment — normalizes by typical gradient magnitude
- $\epsilon$: numerical stability constant (default: $10^{-8}$)

## Why the update is powerful

The ratio $\hat{m}_t / \sqrt{\hat{v}_t}$ can be interpreted as a signal-to-noise ratio:
- Parameters with consistently large gradients (high $\hat{v}_t$) get small updates
- Parameters with small but consistent gradients (low $\hat{v}_t$, nonzero $\hat{m}_t$) get larger updates
- This self-normalizing behavior makes Adam robust to gradient scale differences across layers

## The full algorithm

```
Initialize: m₀ = 0, v₀ = 0, t = 0

For each training step:
  t = t + 1
  g_t = ∇L(θ_{t-1})                        # compute gradient
  m_t = β₁ · m_{t-1} + (1-β₁) · g_t       # first moment update
  v_t = β₂ · v_{t-1} + (1-β₂) · g_t²      # second moment update
  m̂_t = m_t / (1 - β₁ᵗ)                   # bias correction
  v̂_t = v_t / (1 - β₂ᵗ)                   # bias correction
  θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)   # parameter update
```

## Effect of each hyperparameter

| Hyperparameter | Default | Role |
|----------------|---------|------|
| $\eta$ | 1e-3 | Overall step size — most important to tune |
| $\beta_1$ | 0.9 | First moment decay — controls momentum strength |
| $\beta_2$ | 0.999 | Second moment decay — controls adaptive scaling window |
| $\epsilon$ | 1e-8 | Prevents division by zero |

## PyTorch example

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Standard Adam
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8
)

# AdamW (preferred when using weight decay)
optimizer_w = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2
)

criterion = nn.CrossEntropyLoss()
x = torch.randn(64, 128)
y = torch.randint(0, 10, (64,))

optimizer.zero_grad()
loss = criterion(model(x), y)
loss.backward()
optimizer.step()

# Inspect optimizer state for one parameter
p = list(model.parameters())[0]
state = optimizer.state[p]
print("step:", state['step'])
print("first moment shape:", state['exp_avg'].shape)     # m_t
print("second moment shape:", state['exp_avg_sq'].shape) # v_t
```

## Adam vs AdamW

Vanilla Adam implements weight decay by adding $\lambda \theta$ to the gradient before the adaptive update:

$$
g_t' = g_t + \lambda \theta_t \quad \text{(Adam — incorrect coupling)}
$$

This means the weight decay gets divided by $\sqrt{\hat{v}_t}$, weakening it for parameters with large gradients.

**AdamW** applies weight decay directly to the parameter after the update — decoupled from the adaptive scaling:

$$
\theta_{t+1} = \theta_t - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t \quad \text{(AdamW — correct)}
$$

AdamW is the recommended choice whenever weight decay is used. It is the standard for training transformers (BERT, GPT, etc.).

## Interview questions

<details>
<summary>What are the two moment estimates in Adam and what does each one do?</summary>

The first moment ($m_t$) is an EWMA of past gradients — it acts like momentum, providing a smoothed gradient direction. The second moment ($v_t$) is an EWMA of squared gradients — it estimates the typical magnitude of gradients per parameter for adaptive scaling. The update divides the first moment by the square root of the second, normalizing each parameter's step size by its historical gradient magnitude.
</details>

<details>
<summary>Why does Adam need bias correction?</summary>

Both moment estimates are initialized at zero. At early steps, they are biased toward zero because the EWMA averages the true signal with the zero initialization. Bias correction divides by $(1 - \beta^t)$, which is small early on (scaling up the estimates to their true value) and approaches 1 after many steps (correction disappears). Without it, the first few hundred steps would use severely underestimated moment values.
</details>

<details>
<summary>What is the difference between Adam and AdamW?</summary>

Adam applies weight decay by adding $\lambda \theta$ to the gradient before the adaptive update, which inadvertently scales the decay by $1/\sqrt{\hat{v}_t}$. AdamW decouples weight decay from the adaptive mechanism — it applies decay directly to the parameter as a separate $\eta \lambda \theta$ subtraction. AdamW is theoretically correct and empirically better, especially for transformers.
</details>

<details>
<summary>When would you choose SGD + momentum over Adam?</summary>

SGD + momentum (with a cosine learning rate schedule) often achieves better generalization on image classification tasks (e.g., ResNet on ImageNet) because it tends to converge to flatter minima. Adam converges faster but may find sharper minima with slightly worse test accuracy. For transformers and NLP tasks, AdamW is standard because it handles highly heterogeneous gradient magnitudes across embedding and attention layers.
</details>

<details>
<summary>Does using Adam mean you no longer need to tune the learning rate?</summary>

No. Adam adapts relative update magnitudes across parameters, but all of them are still scaled by $\eta$. A poorly chosen $\eta$ still causes divergence (too large) or extremely slow convergence (too small). The learning rate remains the single most important hyperparameter even with Adam.
</details>

## Common mistakes

- Using `torch.optim.Adam` with `weight_decay` instead of `torch.optim.AdamW` — the former couples decay with adaptive scaling incorrectly.
- Not tuning $\eta$ at all and assuming the default 1e-3 works for every task.
- Assuming Adam always generalizes as well as SGD — it can converge to sharper minima on vision tasks.
- Fine-tuning a pretrained model without a warmup phase — a large initial $\eta$ can destroy pretrained representations in the first few steps.
- Forgetting `optimizer.zero_grad()` before backward — gradients accumulate by default in PyTorch.

## Advanced perspective

Adam is related to online mirror descent and can be viewed as a diagonal approximation to the natural gradient. Its second moment estimates the Fisher information matrix diagonal, giving updates that are approximately invariant to parameter scaling. Active variants include: **AMSGrad** (uses the running maximum of $\hat{v}_t$ for a monotone denominator), **Adafactor** (memory-efficient version for large models), **LAMB** (layer-wise adaptive rates for large-batch distributed training), and **Lion** (sign-based momentum update, requires less memory than Adam).

## Final takeaway

Adam is powerful because it solves two problems simultaneously: it smooths gradient direction with momentum, and it normalizes per-parameter update magnitudes adaptively. Understanding both components — and the bias correction — gives you the mental model to diagnose and adjust optimizer behavior whenever training goes wrong.

## References

- Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. ICLR 2015.
- Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR 2019.
