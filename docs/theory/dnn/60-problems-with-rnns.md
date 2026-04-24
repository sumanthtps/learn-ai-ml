---
id: problems-with-rnns
title: "Problems with RNNs"
sidebar_label: "60 · RNN Problems"
sidebar_position: 60
slug: /theory/dnn/problems-with-rnns
description: "The fundamental limitations of vanilla RNNs — vanishing gradients, exploding gradients, sequential computation bottleneck, and fixed context — and the solutions each subsequent architecture introduced."
tags: [rnn, vanishing-gradients, exploding-gradients, sequence-modeling, deep-learning]
---

# Problems with RNNs

Vanilla RNNs introduced temporal dependencies in deep learning, but they come with fundamental limitations that prevent them from working well on long sequences. Understanding these problems is the prerequisite for understanding why LSTMs, GRUs, and eventually transformers were developed.

## One-line definition

Vanilla RNNs struggle with long sequences because gradients vanish or explode during backpropagation through time, and their sequential computation bottleneck prevents parallelization.

![The LSTM solution — compared to vanilla RNNs, the LSTM's cell state creates a protected gradient highway that prevents vanishing across many time steps](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png)
*Source: [Colah's Blog — Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (CC BY 4.0) — vanilla RNN repeating module with a single tanh layer*

## Why this topic matters

Every major advance in sequence modeling after RNNs — LSTMs, GRUs, attention, transformers — was a direct response to one or more of the problems described here. Understanding why RNNs fail tells you exactly what properties the better architectures had to provide.

## Problem 1: Vanishing gradients

In a vanilla RNN, the gradient from time step $T$ back to time step $1$ requires multiplying the Jacobian matrix $(W_h^T \cdot \text{diag}(\tanh'(z_t)))$ for every time step in between.

For a sequence of length $T$, the gradient at step 1 is proportional to:

$$
\prod_{t=2}^{T} W_h^T \cdot \text{diag}(\tanh'(z_t))
$$

The spectral norm of this product shrinks exponentially if $\|W_h\| \cdot \max|\tanh'| < 1$. Since $\tanh'(z) \leq 1$ and $\tanh$ saturates for large $|z|$, the derivative is often much less than 1.

**Consequence**: The RNN cannot learn dependencies between tokens that are far apart in the sequence. By the time the gradient flows back 50–100 steps, it is essentially zero. The early parts of the sequence have no influence on the parameters.

**Example**: In "The cat that the dog chased after a very long walk ran away," the RNN must connect "cat" (subject) to "ran" (verb) across 13 words. Vanilla RNNs fail at this.

## Problem 2: Exploding gradients

If $\|W_h\| > 1$, the same product grows exponentially:

$$
\prod_{t=2}^{T} \|W_h\| \to \infty \text{ as } T \to \infty
$$

The gradient becomes so large that a single update step destabilizes the entire model. In practice, this manifests as:
- NaN loss after a few steps
- Loss suddenly spiking after appearing to decrease
- Weights jumping to extreme values

**Solution**: Gradient clipping. Rescale the gradient norm to a maximum value $\tau$ before the update:

$$
\text{if } \|\nabla \mathcal{L}\| > \tau: \quad \nabla \mathcal{L} \leftarrow \tau \cdot \frac{\nabla \mathcal{L}}{\|\nabla \mathcal{L}\|}
$$

Gradient clipping fixes exploding gradients but does NOT fix vanishing gradients — the gradient direction is preserved, only the magnitude is capped.

```python
# PyTorch gradient clipping
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

## Problem 3: Sequential computation bottleneck

The recurrence $h_t = f(h_{t-1}, x_t)$ requires $h_{t-1}$ before computing $h_t$. This is inherently sequential — the computation at time step $t$ cannot begin until step $t-1$ is complete.

**Consequence**: RNNs cannot be parallelized across the sequence length dimension. For a sequence of length $T$:
- Training: $O(T)$ sequential steps
- Very long sequences (T = 1000+) are slow to train even on modern GPUs

In contrast, transformers compute all token interactions in parallel (matrix operations), making them much faster to train at scale.

## Problem 4: Fixed-size context bottleneck

In many applications, an RNN encoder compresses the entire input sequence into the final hidden state $h_T$:

$$
h_T = \text{RNN}(x_1, x_2, \ldots, x_T)
$$

This single fixed-size vector must carry all information from the sequence. For long sequences:
- Early information is often overwritten by later information
- The vector's capacity is bounded by the hidden dimension $h$
- The decoder has no direct access to early encoder states

This is the "fixed context bottleneck" that the attention mechanism (note 69) was designed to solve.

## Summary of problems and solutions

| Problem | Effect | Solution |
|---|---|---|
| Vanishing gradients | Cannot learn long-range dependencies | LSTM / GRU (additive cell state) |
| Exploding gradients | Training instability, NaN loss | Gradient clipping |
| Sequential computation | Slow training on long sequences | Transformers (parallel attention) |
| Fixed context bottleneck | Encoder loses early information | Attention mechanism (direct encoder access) |

## PyTorch demonstration: observing vanishing gradients

```python
import torch
import torch.nn as nn

# Build a simple RNN and observe gradient norms
rnn = nn.RNN(input_size=10, hidden_size=64, batch_first=True)
x = torch.randn(1, 100, 10, requires_grad=True)  # sequence of 100 steps
y_target = torch.zeros(1, 64)

h0 = torch.zeros(1, 1, 64)
output, h_n = rnn(x, h0)

# Loss on the final hidden state
loss = ((h_n.squeeze() - y_target) ** 2).sum()
loss.backward()

# Check gradient w.r.t. input at each time step
grad_norms = x.grad[0].norm(dim=1)  # (100,) — one norm per time step
print("Gradient norms (first 5 steps):", grad_norms[:5].tolist())
print("Gradient norms (last 5 steps):", grad_norms[-5:].tolist())
# Early steps should have much smaller gradients than late steps
```

## Interview questions

<details>
<summary>What is the vanishing gradient problem in RNNs and what causes it?</summary>

The gradient of the loss with respect to an early time step requires repeatedly multiplying a Jacobian term of the form $W_h^T \cdot \mathrm{diag}(\tanh'(z_t))$ once per time step between that step and the loss. If the spectral norm of this matrix is less than $1$, which is common when `tanh` saturates, the product shrinks exponentially. For long sequences, the gradient at the first step becomes essentially zero, so the RNN cannot learn to connect early inputs to late outputs.
</details>

<details>
<summary>What is gradient clipping and why does it only fix exploding gradients?</summary>

Gradient clipping rescales the gradient vector to have maximum norm τ when its norm exceeds τ. This prevents parameter updates from being catastrophically large (exploding gradient). It does NOT fix vanishing gradients because vanishing means the gradient is already near zero — rescaling a near-zero vector does not make it larger. Vanishing requires architectural changes (LSTM) rather than training tricks.
</details>

<details>
<summary>Why is sequential computation a fundamental limitation of RNNs?</summary>

The RNN update at time step $t$ depends on the hidden state from time step $t-1$, so each step is causally dependent on the previous one. No amount of hardware parallelism can compute step $t$ before step $t-1$ finishes. Transformers remove that dependency by computing token interactions through attention in parallel, which turns sequence processing from $O(T)$ sequential depth into effectively constant parallel depth per layer.
</details>

<details>
<summary>What is the context bottleneck problem and how does attention solve it?</summary>

An RNN encoder maps the entire input sequence to a single final hidden state h_T, which the decoder must use for all output steps. For long sequences, this vector cannot retain all important information. Attention solves this by allowing the decoder to directly access all encoder hidden states (h_1, ..., h_T) at each decoding step, computing a weighted combination based on relevance. No information is lost in a bottleneck.
</details>

## Common mistakes

- Assuming gradient clipping eliminates the vanishing gradient problem — it only prevents exploding gradients.
- Using vanilla RNNs for sequences longer than ~30–50 steps without acknowledging that they likely fail to capture long-range dependencies.
- Thinking that bidirectional RNNs (BiRNNs) solve the vanishing gradient problem — they use future context, but gradients still vanish through time in both directions.

## Advanced perspective

Vanishing gradients in RNNs and transformers are related phenomena that arise whenever gradients must pass through many multiplicative operations. The LSTM's additive cell state is the key fix because it provides a gradient highway that avoids the full multiplicative chain. Residual connections in transformers provide the same structural idea in a simpler form:

$$
h_{l+1} = F(h_l) + h_l
$$

This residual path gives the backward pass a direct route around the unstable transformation. That structural similarity, namely additive skip connections, is the common thread between LSTMs and ResNets.

## Final takeaway

Vanilla RNNs have four compounding problems: vanishing gradients (cannot learn long-range), exploding gradients (training instability), sequential computation (cannot parallelize), and context bottleneck (fixed-size encoding). LSTMs and GRUs fix the first two; attention and transformers fix the last two. Understanding these problems explains the entire arc of sequence modeling from 2010 to the present.

## References

- Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult.
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. ICML.
