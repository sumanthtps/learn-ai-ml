---
id: batch-normalization
title: "Batch normalization in deep learning"
sidebar_label: "31 · Batch Normalization"
sidebar_position: 31
slug: /theory/dnn/batch-normalization-in-deep-learning
description: "Batch normalization normalizes layer activations across the mini-batch, stabilizing training by reducing internal covariate shift, enabling higher learning rates, and acting as implicit regularization."
tags: [batch-normalization, normalization, training-stability, deep-learning]
---

# Batch normalization in deep learning

As networks get deeper, the distribution of activations at each layer shifts during training as the preceding layers' weights change — a phenomenon called **internal covariate shift**. This makes training slower and more sensitive to initialization. Batch normalization addresses this by explicitly normalizing activations at each layer during training.

## One-line definition

Batch normalization normalizes each feature in a mini-batch to have zero mean and unit variance, then applies learnable scale and shift parameters — keeping activations in a stable range and dramatically accelerating training of deep networks.

![Residual connections and normalization in the transformer encoder — batch normalization (for CNNs) and layer normalization (for transformers) both stabilize activations; the principle is the same](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_2.png)
*Source: [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)*

## Why this topic matters

Batch normalization enabled training networks with tens or hundreds of layers that would have been impossible otherwise. It allows much higher learning rates, reduces sensitivity to weight initialization, and acts as a regularizer (often reducing the need for dropout). It is a standard component in nearly every modern CNN and many other architectures.

## The four-step batch normalization computation

Given a mini-batch of pre-activations $\{z_1, z_2, \ldots, z_m\}$ for one feature/channel:

**Step 1** — Compute batch mean:

$$
\mu_B = \frac{1}{m} \sum_{i=1}^{m} z_i
$$

**Step 2** — Compute batch variance:

$$
\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (z_i - \mu_B)^2
$$

**Step 3** — Normalize:

$$
\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

$\epsilon$ (typically $10^{-5}$) prevents division by zero when variance is near zero.

**Step 4** — Scale and shift with learnable parameters $\gamma$ and $\beta$:

$$
\text{BN}(z_i) = \gamma \hat{z}_i + \beta
$$

The parameters $\gamma$ (scale) and $\beta$ (shift) are learned during training via backpropagation. They allow the network to **undo** the normalization if that is what the task requires, giving the network full expressiveness.

## Why $\gamma$ and $\beta$ are necessary

Without $\gamma$ and $\beta$, batch normalization would force every layer to have activations with zero mean and unit variance — which may not be optimal for the task. For example, if a layer benefits from having saturated sigmoid activations, forcing them to $\mathcal{N}(0,1)$ would be harmful. The learnable parameters let the network decide the optimal distribution at each layer.

Special cases:
- $\gamma = \sqrt{\sigma_B^2 + \epsilon}$, $\beta = \mu_B$ → recovers the original distribution (normalization is undone)
- $\gamma = 1$, $\beta = 0$ → standard normalization

## Training vs inference

**During training**: $\mu_B$ and $\sigma_B^2$ are computed from the current mini-batch. Both are used for normalization, and both contribute to the backward pass.

**During inference**: The mini-batch statistics are not used — there may be only a single sample, making batch statistics meaningless. Instead, **running statistics** (exponential moving averages of batch statistics accumulated during training) are used:

$$
\mu_{\text{running}} = (1-\alpha) \mu_{\text{running}} + \alpha \mu_B
$$

$$
\sigma^2_{\text{running}} = (1-\alpha) \sigma^2_{\text{running}} + \alpha \sigma^2_B
$$

In PyTorch, `model.train()` uses batch statistics; `model.eval()` uses running statistics. **Forgetting to call `model.eval()` at inference time is one of the most common bugs in PyTorch.**

## Where to place batch normalization

The original paper placed BN between the linear/conv layer and the activation:

$$
\text{Linear} \to \text{BN} \to \text{Activation} \to \text{Next layer}
$$

Many practitioners now place it after the activation (post-activation BN) or use pre-activation residual blocks. For transformers, Layer Normalization is used instead (see note 79).

## PyTorch example

```python
import torch
import torch.nn as nn

# BatchNorm in a fully connected network
model = nn.Sequential(
    nn.Linear(128, 256),
    nn.BatchNorm1d(256),   # normalize across batch, per feature
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# BatchNorm in a CNN
cnn = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),    # normalize per channel across (H, W, batch)
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU()
)

# CRITICAL: use model.train() for training, model.eval() for inference
model.train()
x_train = torch.randn(64, 128)
out_train = model(x_train)

model.eval()
with torch.no_grad():
    x_test = torch.randn(1, 128)
    out_test = model(x_test)  # uses running statistics, not batch stats
```

## Why batch normalization accelerates training

1. **Reduces internal covariate shift**: Each layer receives inputs with a stable distribution, regardless of how upstream layers changed.
2. **Allows higher learning rates**: Without BN, high LRs cause activations to grow large, saturating activations and vanishing gradients. BN keeps activations bounded.
3. **Implicit regularization**: The noise in mini-batch statistics acts like a regularizer. Each sample's normalization depends on the rest of the batch — a form of data augmentation. This reduces overfitting (though BN should not replace explicit regularization on small datasets).
4. **Reduces sensitivity to initialization**: Since BN normalizes activations, the initial weight scale matters less.

## Limitations of batch normalization

- **Small batch sizes**: With batch size 1–4, the batch statistics are very noisy and BN performs poorly. For small batches, Layer Normalization or Group Normalization is preferred.
- **Online learning / streaming**: Single-sample inference requires stored running statistics.
- **Variable-length sequences**: In NLP with variable-length sequences, batch statistics are inconsistent. Layer Normalization is used instead.
- **Batch statistics introduce dependencies**: Samples in the same batch are not independent during BN, complicating distributed training.

## Interview questions

<details>
<summary>What are the four steps of batch normalization?</summary>

1. Compute the mini-batch mean μ_B = (1/m) Σz_i. 2. Compute the mini-batch variance σ²_B = (1/m) Σ(z_i - μ_B)². 3. Normalize: ẑ_i = (z_i - μ_B)/√(σ²_B + ε). 4. Scale and shift with learnable parameters: BN(z_i) = γẑ_i + β. The learnable γ and β allow the network to recover the original distribution if needed.
</details>

<details>
<summary>Why are the learnable parameters γ and β necessary in batch normalization?</summary>

Forcing every layer to have zero mean and unit variance may not be optimal for the task. γ and β let the network learn the optimal distribution for each layer. In the limit, if the network learns γ = σ_B and β = μ_B, it can exactly undo the normalization. Without these parameters, batch normalization would constrain network expressiveness.
</details>

<details>
<summary>What is the difference between batch normalization at training time vs inference time?</summary>

During training, normalization uses statistics computed from the current mini-batch. During inference, it uses running statistics (exponentially accumulated during training) because mini-batch statistics may be unavailable (single sample) or unreliable. In PyTorch, model.train() activates batch statistics mode; model.eval() switches to running statistics mode. Forgetting to call model.eval() is a common bug.
</details>

<details>
<summary>Why is batch normalization ineffective for very small batch sizes?</summary>

With batch size 1–4, the sample mean and variance are poor estimates of the true distribution. The normalization effectively adds large noise to the activations, and the learned running statistics are unstable. For small batches or variable-length sequences, Layer Normalization (normalizing across features within each sample) is preferred.
</details>

<details>
<summary>Where is batch normalization typically placed relative to activations?</summary>

The original paper placed it between the linear/conv layer and the activation (pre-activation). Many modern architectures use post-activation placement. In residual networks, pre-activation BN (BN → ReLU → Conv) tends to perform better than post-activation. For transformers, Layer Normalization replaces batch normalization entirely.
</details>

## Common mistakes

- Forgetting `model.eval()` before inference — the model will use noisy mini-batch stats instead of the stable running stats.
- Using `BatchNorm1d` when you need `BatchNorm2d` for convolutional feature maps.
- Applying BN with batch size 1 or 2 — the statistics are so noisy that BN degrades performance.
- Removing dropout after adding BN — for small datasets, explicit regularization is still needed alongside BN.
- Not setting `track_running_stats=True` (the PyTorch default) when you need stable inference.

## Advanced perspective

Batch normalization can be interpreted as an optimization technique that makes the loss surface smoother (reduces the Lipschitz constant of the gradient). This is why it allows higher learning rates: the optimization landscape becomes more predictable. From a statistical perspective, BN is a form of whitening the inputs at each layer, which is known to accelerate optimization in linear models and carries similar benefits in deep networks.

## Final takeaway

Batch normalization is one of the most impactful training techniques in deep learning. It solves internal covariate shift, enables high learning rates, reduces sensitivity to initialization, and adds regularization — all with a small computational overhead. The train/eval mode distinction is critical to use it correctly.

## References

- Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML 2015.
- Santurkar, S., et al. (2018). How Does Batch Normalization Help Optimization? NeurIPS 2018.
