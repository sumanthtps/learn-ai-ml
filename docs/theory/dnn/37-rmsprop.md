---
id: rmsprop
title: "RMSProp"
sidebar_label: "37 · RMSProp"
sidebar_position: 37
slug: /theory/dnn/rmsprop
description: "RMSProp fixes Adagrad's monotonically decreasing learning rate by using an exponentially weighted moving average of squared gradients instead of an unbounded running sum."
tags: [rmsprop, adaptive-learning-rate, optimizers, deep-learning]
---

# RMSProp

Adagrad's key insight was per-parameter adaptive learning rates, but its implementation accumulated squared gradients forever, causing the learning rate to decay toward zero. RMSProp keeps the per-parameter adaptation but replaces the running sum with an exponentially weighted moving average (EWMA) — giving the optimizer a finite memory window instead of total accumulation.

## One-line definition

RMSProp maintains a decaying average of squared gradients per parameter, normalizing each update by the recent gradient magnitude rather than the total historical magnitude — preventing the learning rate collapse that plagues Adagrad.

## Why this topic matters

RMSProp was proposed by Geoffrey Hinton (unpublished, from his Coursera lecture) specifically to fix Adagrad's learning rate collapse. It works well for non-stationary objectives — tasks where the gradient statistics change over time, as they do during deep network training. RMSProp is also the adaptive component inside Adam: Adam essentially combines RMSProp's second moment with SGD momentum.

## The RMSProp update

**EWMA of squared gradients** (decaying memory of recent gradient magnitudes):

$$
v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot g_t^2
$$

where $\beta$ is the decay factor (typically $0.9$) and $g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$.

**Parameter update:**

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot g_t
$$

This is identical in form to Adagrad, except $G_t$ (monotonically increasing sum) is replaced by $v_t$ (EWMA that decays with time).

![RMSProp adaptive learning rate — per-parameter learning rates decay from large values when gradients are large, then recover as gradients shrink](https://commons.wikimedia.org/wiki/Special:Redirect/file/Artificial_neural_network.svg)
*Source: [Wikimedia Commons — Artificial Neural Network](https://commons.wikimedia.org/wiki/File:Artificial_neural_network.svg) (CC BY-SA 4.0)*

## The critical fix: EWMA vs running sum

| | Adagrad | RMSProp |
|---|---|---|
| Denominator | $G_t = \sum_{k=1}^t g_k^2$ | $v_t = \beta v_{t-1} + (1-\beta) g_t^2$ |
| Memory | All history since start | Recent history only (~window 1/(1-β)) |
| LR behavior | Monotonically decreasing | Can recover if gradients become smaller |
| Good for | Sparse, short training | Non-stationary, dense, long training |

With $\beta = 0.9$, RMSProp effectively averages over the last ~10 gradient steps. If gradient magnitudes decrease (as often happens when approaching a good region), $v_t$ decreases too, allowing the effective learning rate to recover.

## Why RMSProp helps on RNNs

RNNs are a classic case where gradient statistics change dramatically across time steps and training stages. Adagrad would quickly kill the learning rate; RMSProp's decay window adapts to the current regime. This is why RMSProp was for a long time the default for RNN training before Adam took over.

## PyTorch example

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10)
)

optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=1e-3,
    alpha=0.9,       # β — EWMA decay for squared gradients
    eps=1e-8,
    momentum=0.0     # RMSProp can also use momentum (rarely done)
)

criterion = nn.CrossEntropyLoss()
x = torch.randn(64, 128)
y = torch.randint(0, 10, (64,))

optimizer.zero_grad()
loss = criterion(model(x), y)
loss.backward()
optimizer.step()
```

Note: PyTorch calls the decay factor `alpha` in `RMSprop` (equivalent to $\beta$ in the formula above).

## Comparison with Adam

Adam extends RMSProp by also maintaining a first moment (momentum) and applying bias correction:

$$
\text{Adam} = \text{RMSProp's adaptive scaling} + \text{SGD momentum} + \text{bias correction}
$$

RMSProp without momentum has no directional smoothing, so it can still oscillate in ravines. Adam's first moment handles this. In most modern applications, Adam or AdamW is preferred over plain RMSProp.

## Interview questions

<details>
<summary>What is the fundamental difference between Adagrad and RMSProp?</summary>

Adagrad accumulates all squared gradients from the beginning of training as a running sum:

$$
G_t = \sum_{k=1}^{t} g_k^2
$$

This sum grows monotonically, causing the learning rate to decay toward zero and eventually stopping learning. RMSProp replaces that growing sum with an exponentially weighted moving average:

$$
v_t = \beta v_{t-1} + (1-\beta) g_t^2
$$

This gives the optimizer a finite memory window. If gradient magnitudes decrease, $v_t$ decreases too, and the effective learning rate can recover.
</details>

<details>
<summary>What does the α (alpha / β) parameter in RMSProp control?</summary>

The decay parameter controls the EWMA memory for squared gradients. The effective window is approximately $\frac{1}{1-\alpha}$ steps. With $\alpha = 0.9$, RMSProp considers roughly the last 10 gradient steps. A higher value such as $0.99$ gives a longer memory window and more smoothing, while a lower value such as $0.5$ reacts faster to recent changes but is noisier.
</details>

<details>
<summary>How does RMSProp relate to Adam?</summary>

Adam's second moment term $v_t$ is exactly RMSProp's squared-gradient EWMA. Adam adds two things on top of that: a first moment term $m_t$, which acts like momentum on the gradient direction, and bias correction for both moving averages. In that sense, Adam can be understood as RMSProp-style adaptive scaling plus momentum and bias correction.
</details>

## Common mistakes

- Confusing `alpha` in PyTorch's `RMSprop` with the learning rate `lr` — `alpha` is the squared-gradient decay factor (≈ 0.9), not the step size.
- Using RMSProp for modern transformer training instead of AdamW — Adam's first moment provides direction smoothing that RMSProp lacks.
- Expecting RMSProp to match Adam's performance on NLP tasks — the missing momentum term is significant.

## Advanced perspective

RMSProp can be interpreted as an online estimate of the second moment of the gradient distribution. Normalizing by this estimate makes the update scale-invariant: doubling all gradients would double both the numerator $g_t$ and the denominator $\sqrt{v_t}$, leaving the update unchanged. This scale invariance is valuable when different layers of a network have very different gradient magnitudes (which is typical).

## Final takeaway

RMSProp solves Adagrad's learning rate collapse by using a decaying average instead of a growing sum. It introduced the EWMA-based second moment that became the backbone of Adam. For most modern tasks, Adam is preferred over RMSProp, but understanding RMSProp is essential for understanding Adam.

## References

- Hinton, G. (2012). Lecture 6e: RMSProp: Divide the gradient by a running average of its recent magnitude. Coursera Neural Networks for Machine Learning.
- Tieleman, T., & Hinton, G. (2012). Lecture 6.5 — RMSProp.
