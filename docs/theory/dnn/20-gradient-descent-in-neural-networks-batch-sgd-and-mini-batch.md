---
id: gradient-descent-variants
title: "Gradient descent in neural networks: batch, SGD, and mini-batch"
sidebar_label: "20 · Gradient Descent Variants"
sidebar_position: 20
slug: /theory/dnn/gradient-descent-in-neural-networks-batch-sgd-and-mini-batch
description: "The three gradient descent variants — batch, stochastic, and mini-batch — differ in how many samples they use to estimate the gradient, with fundamental tradeoffs in convergence stability and computational efficiency."
tags: [gradient-descent, sgd, mini-batch, optimization, deep-learning]
---

# Gradient descent in neural networks: batch, SGD, and mini-batch

Backpropagation gives us the gradient of the loss for any given data. But on which data? The entire dataset? One sample at a time? A small random subset? The answer — which variant of gradient descent to use — has fundamental consequences for convergence speed, stability, and generalization.

## One-line definition

The three gradient descent variants differ in how many training samples are used to estimate the gradient at each update step: all samples (batch GD), one sample (SGD), or a random subset (mini-batch GD).

![Stochastic gradient descent — the noisy, sample-by-sample update path (blue) compared to the smooth batch gradient path; the noise is a feature: it helps escape sharp local minima](https://upload.wikimedia.org/wikipedia/commons/f/f3/Stogra.png)
*Source: [Wikimedia Commons — Stochastic gradient](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (CC BY-SA 3.0)*

## Why this topic matters

In theory, using all samples gives the most accurate gradient. In practice, mini-batch gradient descent with a good batch size is almost universally better: it converges faster, generalizes better, and is computationally efficient on modern hardware. Understanding why this is true requires understanding the noise-convergence tradeoff.

## Batch gradient descent

Use all $N$ training samples to compute the exact gradient before each update:

$$
\theta \leftarrow \theta - \eta \cdot \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \mathcal{L}(x_i, y_i; \theta)
$$

**Properties:**
- Exact gradient — no estimation noise
- Guaranteed to descend (with appropriate learning rate)
- Extremely slow: one parameter update requires a full pass through all $N$ samples
- Cannot fit in memory for large datasets
- Converges to a local minimum precisely (no noise to escape bad regions)

## Stochastic gradient descent (SGD)

Use a single randomly selected sample per update:

$$
\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}(x_i, y_i; \theta)
$$

where $x_i$ is drawn randomly at each step.

**Properties:**
- Very fast: one update per sample
- Very noisy gradient estimate — high variance
- Noisy updates can escape sharp local minima and saddle points
- Learning rate must be carefully decayed over time for convergence
- Harder to parallelize efficiently (GPUs thrive on batch operations)

## Mini-batch gradient descent

Use a random subset of $B$ samples (the **batch size**) per update:

$$
\theta \leftarrow \theta - \eta \cdot \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla_\theta \mathcal{L}(x_i, y_i; \theta)
$$

**This is the standard in practice.** Almost all deep learning uses mini-batch GD.

**Properties:**
- Moderate noise: better gradient estimate than SGD, noisier than batch GD
- Efficient: GPU parallelizes the batch computation well
- Balances convergence speed and stability
- Batch size $B$ is a key hyperparameter (typically 32–512)

## Comparison table

| | Batch GD | SGD | Mini-batch GD |
|---|---|---|---|
| Samples per update | All $N$ | 1 | $B$ (e.g., 32–512) |
| Gradient quality | Exact | Very noisy | Moderate noise |
| Memory requirement | High | Minimal | Moderate |
| GPU efficiency | Excellent | Poor | Excellent |
| Convergence | Stable, precise | Noisy, can oscillate | Balanced |
| Escape local minima | Poor | Good | Good |
| Practical use | Never (large data) | Rarely | Always |

## Why noise can be beneficial

Counter-intuitively, the noise in mini-batch/SGD gradient estimates can improve generalization:

1. **Escaping sharp minima**: Noisy updates can "jump out" of sharp, narrow minima. Batch GD precisely descends into the nearest minimum, which may be sharp and generalize poorly. SGD tends to find flatter minima with better generalization.

2. **Implicit regularization**: The noise acts like a regularizer, preventing the model from perfectly memorizing the training set.

3. **Saddle point escape**: In high-dimensional non-convex losses, saddle points are common. Gradient noise helps escape these regions where the exact gradient is nearly zero.

## The epoch concept

An **epoch** is one complete pass through the training dataset. For mini-batch GD with $N$ samples and batch size $B$:

$$
\text{updates per epoch} = \frac{N}{B}
$$

With $N = 50000$ and $B = 128$: ~390 updates per epoch. Convergence is tracked in epochs (not raw gradient steps) to make comparisons across batch sizes meaningful.

## Choosing batch size

| Batch size | Effect |
|---|---|
| Very small (1–4) | High noise, may not converge stably |
| Small (16–64) | Good regularization, moderate GPU utilization |
| Medium (128–512) | Standard — balances all tradeoffs |
| Large (2048+) | Fast wall-clock time but often worse generalization |

A key empirical finding: **large-batch training generalizes worse than small-batch training**, even if training loss is similar (the "sharp vs flat minima" effect). Large-batch methods need warmup and often a higher learning rate to compensate.

## PyTorch example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Create a dataset
X = torch.randn(10000, 64)
y = torch.randint(0, 10, (10000,))
dataset = TensorDataset(X, y)

# Mini-batch DataLoader — this IS mini-batch gradient descent
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for batch_x, batch_y in loader:          # each iteration = one mini-batch update
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1} — updates per epoch: {len(loader)}")
    # With N=10000, B=128: ~78 updates per epoch
```

## The learning rate and batch size interaction

A standard heuristic: **linear scaling rule**. If you increase batch size by a factor of $k$, scale the learning rate by $k$:

$$
\eta_{\text{large}} = k \cdot \eta_{\text{small}}
$$

Intuition: with $k$ times more samples per batch, the gradient has $k$ times lower variance, so a larger step is appropriate. This rule usually works well up to moderate batch sizes but breaks down at very large batches without warmup.

## Interview questions

<details>
<summary>What is the key practical difference between SGD and mini-batch gradient descent?</summary>

SGD uses a single sample per update, producing a very noisy but cheap gradient estimate. Mini-batch GD uses B samples per update, averaging out noise while remaining computationally efficient on GPUs (which can parallelize the batch). In practice, "SGD" in deep learning almost always means mini-batch SGD — the optimizer is called SGD but uses a batch size > 1.
</details>

<details>
<summary>Why does mini-batch gradient descent often generalize better than full-batch gradient descent?</summary>

Mini-batch GD introduces gradient noise that acts as an implicit regularizer. The noise causes the optimizer to prefer flatter minima (which generalize better) over sharp minima (which may have lower training loss but higher test loss). Full-batch GD descends precisely to the nearest local minimum, which tends to be sharper and overfit more.
</details>

<details>
<summary>What is an epoch and why is it used to track training progress?</summary>

An epoch is one complete pass through the training dataset. Since mini-batch GD sees a random subset each update, one epoch ensures every sample has been seen once on average. Training is tracked in epochs (rather than raw gradient steps) to fairly compare experiments with different batch sizes — 10 epochs means the same number of data passes regardless of batch size.
</details>

<details>
<summary>How does batch size affect the learning rate?</summary>

Larger batches produce more accurate (lower variance) gradient estimates, so larger steps are appropriate. The linear scaling rule: if batch size is multiplied by k, scale the learning rate by k. This rule is empirically valid up to moderate scales but typically requires a warmup period at the start of training to stabilize.
</details>

## Common mistakes

- Calling the standard deep learning optimizer "stochastic gradient descent" when it is actually mini-batch GD — a minor naming confusion but worth knowing.
- Setting batch size too large without increasing the learning rate — the gradient signal becomes too stable and the model may converge to sharp minima.
- Forgetting to shuffle the DataLoader (`shuffle=True`) — without shuffling, the model sees the same samples in the same order every epoch, harming generalization.
- Using batch size 1 (pure SGD) for deep networks — the variance is too high for stable convergence in practice.

## Advanced perspective

The implicit regularization of SGD is an active research area. Theoretical results show that the noise in mini-batch gradients biases SGD toward flat minima that generalize better — a phenomenon formalized in the "SGD as a regularizer" framework. The flatness of a minimum correlates with how well it generalizes, and the batch size directly controls this tradeoff. This is why distillation and large-scale language model training papers carefully study batch size schedules.

## Final takeaway

Mini-batch gradient descent is the universal standard because it hits the sweet spot: enough samples to give a useful gradient estimate, few enough to run efficiently on GPUs, and just enough noise to find well-generalizing minima. Understanding the batch size — learning rate tradeoff is one of the most practical skills in deep learning.

## References

- LeCun, Y. et al. (1998). Efficient BackProp.
- Keskar, N. et al. (2017). On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.
