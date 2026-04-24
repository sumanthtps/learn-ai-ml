---
id: relu-variants
title: "ReLU variants: Leaky ReLU, PReLU, ELU, and SELU"
sidebar_label: "28 · ReLU Variants"
sidebar_position: 28
slug: /theory/dnn/relu-variants-leaky-relu-prelu-elu-and-selu
description: "How Leaky ReLU, PReLU, ELU, and SELU extend plain ReLU to fix the dying neuron problem and improve gradient flow for negative pre-activations."
tags: [relu, leaky-relu, elu, selu, activation-functions, deep-learning]
---

# ReLU variants: Leaky ReLU, PReLU, ELU, and SELU

ReLU improved training by eliminating gradient shrinkage for positive inputs, but it introduced a new problem: neurons with always-negative pre-activations receive zero gradient and stop learning permanently — the dying ReLU problem. A family of ReLU variants addresses this by allowing small but nonzero gradients for negative inputs.

## One-line definition

ReLU variants keep the non-saturating behavior for positive inputs while fixing the zero-gradient problem for negative inputs, by replacing the hard zero with either a small linear leak, a learned slope, or a smooth exponential tail.

## Why this topic matters

Understanding ReLU variants is important when diagnosing dying neurons, choosing activations for specific architectures, and understanding why transformers use GELU instead of ReLU. The variants share a common goal but differ in how they handle negative pre-activations and whether they introduce learnable parameters.

## Leaky ReLU

$$
\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}
$$

where $\alpha$ is a small fixed constant (typically $0.01$).

**Derivative:**

$$
\text{LeakyReLU}'(z) = \begin{cases} 1 & z > 0 \\ \alpha & z \leq 0 \end{cases}
$$

**Key fix**: Negative inputs now have a small nonzero gradient ($\alpha = 0.01$), preventing neurons from dying permanently.

**When to use**: Direct drop-in replacement for ReLU when dying neurons are suspected. Common in GANs (discriminators benefit from the nonzero gradient for negative activations).

## PReLU (Parametric ReLU)

$$
\text{PReLU}(z) = \begin{cases} z & z > 0 \\ a \cdot z & z \leq 0 \end{cases}
$$

where $a$ is a **learned** parameter (initialized to 0.25, updated by backpropagation).

**Difference from Leaky ReLU**: $\alpha$ is fixed in Leaky ReLU; $a$ is trained in PReLU. The network learns the optimal negative slope for each layer (or even per neuron).

**PyTorch usage**: `nn.PReLU()` — note that PReLU has a weight parameter that counts toward the model's parameter total.

## ELU (Exponential Linear Unit)

$$
\text{ELU}(z) = \begin{cases} z & z > 0 \\ \alpha (e^z - 1) & z \leq 0 \end{cases}
$$

where $\alpha > 0$ (typically 1.0).

**Derivative:**

$$
\text{ELU}'(z) = \begin{cases} 1 & z > 0 \\ \alpha e^z & z \leq 0 \end{cases}
$$

**Key properties:**
- For negative $z$, output approaches $-\alpha$ (not 0): outputs are bounded below by $-\alpha$
- **Zero-centered outputs**: the mean output is closer to zero than ReLU, improving optimization
- Smooth at $z = 0$: the function is continuous and differentiable everywhere
- More expensive: requires computing $e^z$

**When to use**: When zero-centered activations are important. Empirically outperforms ReLU on some tasks but is less commonly used due to computational cost.

## SELU (Scaled ELU)

$$
\text{SELU}(z) = \lambda \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \leq 0 \end{cases}
$$

with specific constants: $\alpha \approx 1.6733$ and $\lambda \approx 1.0507$.

**The key property**: SELU is a **self-normalizing** activation. Under mild conditions (specific initialization + architecture), SELU networks maintain approximately zero mean and unit variance activations across layers — without batch normalization.

This is achieved because the exact values of $\alpha$ and $\lambda$ were derived analytically to make the fixed point of the mean and variance transformation equal to mean=0, variance=1.

**When to use**: Deep fully-connected networks where batch normalization is inconvenient (e.g., sequence lengths vary, very small batches). Must use `nn.AlphaDropout` (not regular Dropout) to maintain the self-normalizing property.

## GELU (Gaussian Error Linear Unit)

Though not an ReLU variant in the strict sense, GELU is worth knowing here:

$$
\text{GELU}(z) = z \cdot \Phi(z) \approx 0.5z \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(z + 0.044715 z^3)\right]\right)
$$

where $\Phi(z)$ is the Gaussian CDF.

GELU weights input $z$ by the probability of drawing a value less than $z$ from $\mathcal{N}(0,1)$. It is smooth, non-monotonic, and approximates ReLU for large positive inputs while being smoother near 0. It is the default activation in BERT, GPT-2, GPT-3, and most modern transformers.

## Summary table

| Activation | Negative gradient | Learnable? | Zero-centered? | Use case |
|---|---|---|---|---|
| ReLU | 0 | No | No | Default MLP/CNN |
| Leaky ReLU | $\alpha z$ ($\alpha=0.01$) | No | No | GANs, dying neuron fix |
| PReLU | $az$ (learned) | Yes | No | When $\alpha$ varies per layer |
| ELU | $\alpha(e^z - 1)$ | No | Yes | When zero-centering matters |
| SELU | (scaled ELU) | No | Self-normalizing | Deep FC networks, no BN |
| GELU | Smooth approx. | No | Yes | Transformers, NLP |

## PyTorch example

```python
import torch
import torch.nn as nn

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# All variants
print("ReLU:      ", torch.relu(x))
print("LeakyReLU: ", torch.nn.functional.leaky_relu(x, negative_slope=0.01))
print("ELU:       ", torch.nn.functional.elu(x, alpha=1.0))

# PReLU (learnable — needs module form)
prelu = nn.PReLU()
print("PReLU:     ", prelu(x))

# In a network
model = nn.Sequential(
    nn.Linear(64, 128),
    nn.LeakyReLU(negative_slope=0.01),  # good for GANs
    nn.Linear(128, 64),
    nn.ELU(alpha=1.0),                   # zero-centered
    nn.Linear(64, 10)
)
```

## Interview questions

<details>
<summary>What is the dying ReLU problem and which variants solve it?</summary>

A ReLU neuron "dies" when its pre-activation is always ≤ 0 — the gradient is exactly 0, so the weight never updates and the neuron is permanently inactive. Leaky ReLU solves this with a fixed small slope (α=0.01) for negative inputs. PReLU learns the slope. ELU and SELU replace the zero with a smooth exponential, also giving nonzero gradients everywhere. All of these ensure dead neurons can recover.
</details>

<details>
<summary>What is the difference between Leaky ReLU and PReLU?</summary>

In Leaky ReLU, the negative slope α is a fixed hyperparameter (typically 0.01). In PReLU, the equivalent parameter a is learned via backpropagation — the network discovers the optimal negative slope during training. PReLU can learn different slopes per channel or per neuron, giving it more flexibility but adding parameters and the risk of overfitting on small datasets.
</details>

<details>
<summary>Why is GELU used in transformers instead of ReLU?</summary>

GELU provides smoother gradients than ReLU (no sharp kink at z=0), which empirically improves optimization for the large transformer architectures. Its stochastic interpretation (weighting the input by its probability under a standard Gaussian) also provides implicit regularization. BERT, GPT, and most modern LLMs use GELU in their feed-forward sublayers.
</details>

<details>
<summary>What makes SELU "self-normalizing"?</summary>

SELU uses precisely tuned constants (α ≈ 1.6733, λ ≈ 1.0507) derived analytically so that the mean and variance of activations converge to 0 and 1 respectively under repeated SELU transformations. This means a deep SELU network maintains normalized activations throughout, without needing batch normalization. This only holds when using lecun_normal initialization and AlphaDropout.
</details>

## Common mistakes

- Using regular Dropout with SELU networks — this breaks the self-normalizing property. Use `nn.AlphaDropout` instead.
- Using PReLU without accounting for its additional parameters in model size estimates.
- Expecting GELU to be faster than ReLU — GELU is computationally more expensive (involves a tanh computation); the performance benefit is in training quality, not speed.
- Using Leaky ReLU with a very large α (e.g., 0.3) — at that point, the function is not really "fixing" dying neurons but introducing a different inductive bias.

## Advanced perspective

The choice of activation function defines the function class the network can represent. ReLU networks are piecewise linear; the number of linear pieces grows exponentially with depth. GELU and other smooth activations produce infinitely differentiable mappings, which may matter for tasks requiring fine-grained interpolation. Recent research includes Swish ($z \cdot \sigma(\beta z)$), Mish ($z \cdot \tanh(\ln(1 + e^z))$), and SiLU ($z \cdot \sigma(z)$) — all smooth, non-monotonic approximations that often outperform ReLU on image classification.

## Final takeaway

ReLU variants are targeted fixes for the dying neuron problem: Leaky ReLU (fixed leak), PReLU (learned leak), ELU (smooth exponential), SELU (analytically self-normalizing), GELU (transformer default). In practice, use ReLU as the default, switch to Leaky ReLU if you observe dying neurons, and use GELU for transformer architectures.

## References

- He, K., et al. (2015). Delving Deep into Rectifiers (PReLU).
- Clevert, D., et al. (2016). Fast and Accurate Deep Network Learning by Exponential Linear Units (ELU).
- Klambauer, G., et al. (2017). Self-Normalizing Neural Networks (SELU).
- Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELU).
