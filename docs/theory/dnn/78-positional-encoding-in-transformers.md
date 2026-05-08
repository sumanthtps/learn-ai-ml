---
id: positional-encoding
title: "Positional encoding in transformers"
sidebar_label: "78 · Positional Encoding"
sidebar_position: 78
slug: /theory/dnn/positional-encoding-in-transformers
description: "Why transformers need explicit positional information, the sinusoidal encoding scheme from the original transformer paper, and learned positional embeddings used in modern LLMs."
tags: [positional-encoding, transformers, embeddings, deep-learning]
---

# Positional encoding in transformers

> **TL;DR.** Self-attention treats a sentence like a *bag of tokens* — shuffle the words and the math doesn't notice. Positional encoding fixes this by adding a unique, position-dependent vector to each token embedding so "The dog bit the man" and "The man bit the dog" produce different representations. The original transformer used fixed sinusoids; modern LLMs use learned embeddings, RoPE, or ALiBi.

Attention computes similarity between any pair of tokens regardless of their positions in the sequence. This is powerful for global context but means the model is inherently **permutation-invariant**: if you shuffle the tokens, the attention output changes only because the tokens themselves change, not because their positions changed. Without positional information, "The dog bit the man" and "The man bit the dog" would produce the same set of representations (a different permutation of the same vectors).

## One-line definition

Positional encoding adds a position-dependent signal to token embeddings so the transformer can distinguish tokens not just by their content but also by their position in the sequence.

![Positional encoding vectors — added to token embeddings to inject order information; each dimension oscillates at a different frequency](https://jalammar.github.io/images/t/transformer_positional_encoding_vectors.png)
*Source: [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)*

## Why this topic matters

Positional encoding is what gives transformers sequence awareness. Without it, the model is a bag-of-words at each layer. Understanding the different positional encoding schemes — sinusoidal, learned, rotary (RoPE), ALiBi — explains how modern LLMs handle different context lengths and why some generalize to sequences much longer than they were trained on.

## Try it interactively

If you want to *see* positional encoding before reading further, these tools render it live:

- **[Transformer Explainer](https://poloclub.github.io/transformer-explainer/)** — visualize a real GPT-2 forward pass, including positional embeddings, in your browser
- **[LLM Visualization (bbycroft.net)](https://bbycroft.net/llm)** — 3D walkthrough of a working transformer; positional embedding is one of the first stages
- **[Tensor2Tensor PE Colab](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)** — Google's notebook with a positional encoding plot you can re-run

You can also paste the [Python snippet below](#python-code-positional-encoding) into a notebook and call `plt.imshow(pe[0])` to plot the encoding matrix.

## The problem: attention is order-agnostic

In self-attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

If we permute the input rows of $X$ (shuffle tokens), the attention scores change, but only because different tokens are now in different positions in the matrix — not because the positions themselves are encoded. The model has no intrinsic way to know that token 3 comes after token 2.

### A 30-second thought experiment

Take the sentences:

| # | Sentence |
|---|----------|
| A | The cat sat on the mat |
| B | mat the on sat cat The |

Both contain the exact same six word vectors. After self-attention without positional encoding, the *set* of contextual embeddings is identical — only the row ordering differs. A pooled summary (mean of contextual vectors) would be **byte-for-byte identical**. That is the bag-of-words failure mode.

```mermaid
flowchart LR
    A["Sentence A\n'The cat sat on the mat'"] --> EMB["Token embeddings\n(no positional info)"]
    B["Sentence B (shuffled)\n'mat the on sat cat The'"] --> EMB
    EMB --> ATTN["Self-attention"]
    ATTN --> POOL["Mean pooling"]
    POOL --> OUT["Same vector for A and B!"]
```

An MLP has no position sensitivity either. **Position must be added explicitly.**

## Solution: add positional signal to embeddings

The input to the transformer is:

$$
\text{Input}_i = \text{TokenEmbedding}(w_i) + \text{PositionalEncoding}(i)
$$

where $i$ is the position index (0-based). After this addition, two tokens with identical content but different positions get different vectors — the model can now distinguish them.

### Why *add* instead of *concatenate*?

A natural alternative is to concatenate the position vector with the token embedding. Addition was chosen because:

| Choice | Pros | Cons |
|--------|------|------|
| **Add** (used in practice) | No extra parameters; preserves $d_{\text{model}}$; cheap | Position and token info share the same dimensions |
| Concatenate | Cleanly separated channels | Doubles input dimension; more parameters everywhere |

In practice, the model has plenty of capacity to disentangle position from content from the additive signal — empirically, addition works.

## Sinusoidal positional encoding (original transformer)

From "Attention is All You Need" (Vaswani et al., 2017):

$$
\text{PE}(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
\text{PE}(\text{pos}, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)
$$

where:
- $\text{pos}$: position in the sequence (0 to $n-1$)
- $i$: dimension index (0 to $d_{\text{model}}/2 - 1$)
- $d_{\text{model}}$: embedding dimension

Each dimension $2i$ uses a different frequency $1/10000^{2i/d_{\text{model}}}$.

### Worked example: $d_{\text{model}} = 4$, first 4 positions

Plug in concretely. The frequencies for $i \in \{0, 1\}$ are:

- $i = 0$: $1/10000^{0/4} = 1$ → wavelength $2\pi \approx 6.28$
- $i = 1$: $1/10000^{2/4} = 1/100 = 0.01$ → wavelength $200\pi \approx 628$

| pos | dim 0 = sin(pos·1) | dim 1 = cos(pos·1) | dim 2 = sin(pos·0.01) | dim 3 = cos(pos·0.01) |
|-----|--------------------|--------------------|--------------------|--------------------|
| 0 | 0.000 | 1.000 | 0.000 | 1.000 |
| 1 | 0.841 | 0.540 | 0.010 | 1.000 |
| 2 | 0.909 | −0.416 | 0.020 | 1.000 |
| 3 | 0.141 | −0.990 | 0.030 | 1.000 |

Notice dims 0 and 1 change rapidly; dims 2 and 3 barely move. Together they form a unique 4-dimensional fingerprint for each position.

### Intuition: positional binary clock

Think of the position as a binary counter. In binary, digit 0 (least significant) oscillates at the highest frequency; digit $k$ oscillates at frequency $2^{-k}$. The sinusoidal encoding does the analogous thing with continuous sines:

- Small $i$ → large frequency → oscillates rapidly with position (like the seconds hand of a clock)
- Large $i$ → small frequency → slowly varying with position (like the hour hand)

Together, the encoding provides a unique "fingerprint" for each position, the same way the joint reading of seconds, minutes, and hours uniquely identifies a moment.

```mermaid
flowchart LR
    pos["pos = 5"] --> d0["dim 0: sin(5·1) ≈ −0.96\n(fast)"]
    pos --> d1["dim 1: cos(5·1) ≈ 0.28\n(fast)"]
    pos --> d2["dim 2: sin(5·0.01) ≈ 0.05\n(slow)"]
    pos --> d3["dim 3: cos(5·0.01) ≈ 1.00\n(slow)"]
    d0 & d1 & d2 & d3 --> sig["Positional fingerprint\nfor pos = 5"]
```

### Properties of sinusoidal encoding

**Bounded**: $\text{PE}(\text{pos}, j) \in [-1, 1]$ — does not dominate the token embeddings.

**Relative position awareness**: For any fixed offset $k$, $\text{PE}(\text{pos}+k)$ can be expressed as a linear function of $\text{PE}(\text{pos})$:

$$
\text{PE}(\text{pos}+k)_j = \alpha_j \text{PE}(\text{pos})_j + \beta_j \text{PE}(\text{pos})_{j'}
$$

This allows the model to represent relative positions through attention operations.

**Extrapolation**: Since the encoding is a mathematical formula, it can be applied to positions beyond the training context length.

## Sinusoidal PE: visual pattern

The encoding matrix (rows = positions, columns = dimensions) has a distinctive striped pattern:

- Columns near $i=0$: rapidly alternating colors (high-frequency sines/cosines)
- Columns near $i=d/2$: slowly varying, nearly constant strips (low-frequency)

If you plot it as a heatmap, it looks like a fingerprint with vertical stripes that get fatter from left to right. Run the [Python snippet below](#python-code-positional-encoding) and `plt.imshow(pe.pe[0])` to see it.

```mermaid
flowchart LR
    pos["Position pos"] --> even["PE(pos, 2i) = sin(pos/10000^{2i/d})"]
    pos --> odd["PE(pos, 2i+1) = cos(pos/10000^{2i/d})"]
    even --> emb["d_model-dim vector\nunique per position"]
    odd --> emb
    emb --> add["+ Token embedding"]
    add --> transformer["Transformer input"]
```

## Learned positional embeddings

Modern early-LLMs (BERT, GPT-2, GPT-3) use learned embeddings instead of sinusoidal:

$$
\text{PE}(\text{pos}) = E_{\text{pos}}
$$

where $E_{\text{pos}} \in \mathbb{R}^{d_{\text{model}}}$ is a learnable vector for position $\text{pos}$, trained alongside the model.

**Advantages:**
- Can learn the optimal position representation for the task
- Often performs slightly better than sinusoidal on the training sequence length

**Disadvantages:**
- Fixed maximum sequence length — cannot generalize to longer sequences
- Requires seeing all positions during training (rare positions get under-trained)

## Rotary Position Embedding (RoPE)

Used in LLaMA, GPT-NeoX, Mistral, Qwen, and most modern open-source LLMs. Instead of *adding* a positional encoding to embeddings, RoPE **rotates** query and key vectors by a position-dependent rotation matrix:

$$
Q_{\text{pos}} = R_\theta^{\text{pos}} Q, \quad K_{\text{pos}} = R_\theta^{\text{pos}} K
$$

where $R_\theta^{\text{pos}}$ is a block-diagonal rotation matrix. The dot product $Q_{\text{pos}} \cdot K_{\text{pos}'}$ then naturally encodes the **relative** position $\text{pos} - \text{pos}'$ rather than absolute positions.

**Geometric picture:** Imagine each pair of dimensions in $Q$ as a 2D vector. RoPE rotates that vector by an angle that grows linearly with the token's position. When two tokens compute their dot product, the rotation angles partially cancel — leaving a function of `pos_query - pos_key`.

**Key advantage:** Superior length generalization. Because attention naturally sees relative positions, the same model trained on 2K context can be extended to 32K+ with techniques like NTK-aware scaling, YaRN, or LongRoPE.

## ALiBi (Attention with Linear Biases)

Instead of modifying embeddings, ALiBi adds a distance-based bias to attention scores:

$$
\text{score}(i, j) = Q_i K_j^T / \sqrt{d_k} - m \cdot |i - j|
$$

where $m$ is a head-specific slope. Closer tokens get smaller penalties (negative biases of smaller magnitude). This enforces a recency bias and improves length extrapolation significantly without any positional embeddings at all.

## Comparison of positional encoding methods

| Method | Type | Extrapolation | Parameters | Used in |
|---|---|---|---|---|
| Sinusoidal | Fixed | Good | 0 | Original transformer, T5 (relative variant) |
| Learned | Learned | Poor | $n_{\max} \cdot d$ | BERT, GPT-2, GPT-3 |
| RoPE | Fixed rotation | Very good | 0 | LLaMA, Mistral, GPT-NeoX, Qwen, Gemma |
| ALiBi | Fixed bias | Very good | 0 (per head $m$) | MPT, BLOOM, Falcon |

## Python code: positional encoding

A complete, runnable implementation of both sinusoidal and learned variants. Copy this into a notebook to experiment.

### Sinusoidal positional encoding

```python
import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from 'Attention is All You Need'.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create encoding matrix: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()         # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )   # (d_model/2,)  — equivalent to 1/10000^(2i/d_model)

        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions
        pe = pe.unsqueeze(0)   # (1, max_len, d_model) — broadcast over batch

        self.register_buffer("pe", pe)   # not a parameter, but saved in state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token embeddings, shape (batch, seq_len, d_model)
        Returns:
            x + positional encoding, shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Demo
d_model, max_len = 64, 100
pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)

batch, seq = 4, 20
x = torch.randn(batch, seq, d_model)   # token embeddings
out = pe(x)
print(f"Input:  {x.shape}")    # (4, 20, 64)
print(f"Output: {out.shape}")  # (4, 20, 64)

# ============================================================
# Visualize the encoding for first 50 positions, all dimensions
# ============================================================
import matplotlib.pyplot as plt

encoding_matrix = pe.pe[0, :50, :].numpy()

plt.figure(figsize=(12, 4))
plt.imshow(encoding_matrix.T, aspect="auto", cmap="RdYlBu", origin="lower")
plt.xlabel("Position")
plt.ylabel("Dimension")
plt.title("Sinusoidal Positional Encoding — fast oscillation at low dims, slow at high dims")
plt.colorbar()
plt.tight_layout()
plt.savefig("positional_encoding.png", dpi=150)
plt.show()
```

### Try it yourself: experiments

After the snippet above runs, try these one-line modifications:

| Question | Try this |
|----------|----------|
| What happens with a wavelength base of 100 instead of 10000? | Replace `10000.0` with `100.0` and re-plot |
| How distinguishable are positions 0 vs 1? Position 0 vs 100? | `print(torch.norm(pe.pe[0,0] - pe.pe[0,1]))` and same for 100 |
| Are any two positions identical? | `print((pe.pe[0] @ pe.pe[0].T).diag().min())` should be ≈ 1, off-diagonal smaller |
| Plot just one dimension as a function of position | `plt.plot(pe.pe[0,:,0])` then try `[:, 32]` |

### Learned positional embeddings

```python
class LearnedPositionalEncoding(nn.Module):
    """Learned position embeddings — used in BERT, GPT-2."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        return self.dropout(x + self.pos_embedding(positions))


# In a complete embedding layer (like BERT)
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = LearnedPositionalEncoding(d_model, max_len)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(token_ids)   # (batch, seq, d_model)
        return self.norm(self.pos_emb(x))
```

## What if you remove positional encoding?

A useful sanity check is to actually try training a tiny transformer with PE disabled. The pattern is reproducible:

| Setup | Outcome |
|-------|---------|
| Transformer with PE | Trains normally; learns word-order-dependent patterns |
| Transformer without PE | Trains, but loss plateaus much higher |
| Same model, but inputs shuffled per batch | With PE: large drop in accuracy. Without PE: no change at all |

The last row is the diagnostic: **if shuffling tokens within a sequence has no effect on the loss, your transformer is operating as a bag-of-words.**

## Cross-references

- **Prerequisite:** [73 — Self-Attention with Code](./73-self-attention-in-transformers-with-code.md) (why attention itself is permutation-equivariant)
- **Follow-up:** [80 — Transformer Encoder Architecture](./80-transformer-encoder-architecture.md) (where positional encoding sits in the full encoder)
- **Related:** [77 — Multi-Head Attention](./77-multi-head-attention-in-transformers.md) (PE is added once, before splitting into heads)

## Interview questions

<details>
<summary>Why does attention not inherently capture positional information?</summary>

Scaled dot-product attention computes Q @ K^T, where each row of Q corresponds to a token and each column of K corresponds to a token. The computation is purely based on content (the values in Q and K), not on which row/column the token occupies. If you permute the input sequence (shuffle rows of X), the output for each token changes only because different tokens are now co-occurring, not because the transformer "knows" the order changed. Positional encoding injects order information into the content.
</details>

<details>
<summary>What is the advantage of sinusoidal encoding over learned embeddings for length generalization?</summary>

Sinusoidal encoding is a mathematical formula that can be computed for any position, including positions beyond the training length. Learned embeddings have a fixed table size — position 513 in a model trained with max_len=512 has no embedding. Sinusoidal and relative encodings (RoPE, ALiBi) generalize to longer sequences without modification.
</details>

<details>
<summary>What is RoPE and why is it used in most modern LLMs?</summary>

RoPE (Rotary Position Embedding) encodes position by rotating query and key vectors by position-dependent angles. The dot product Q_pos · K_pos' naturally depends only on the relative offset (pos - pos'), not absolute positions. This makes attention inherently relative-position-aware. RoPE generalizes well to longer contexts and can be extended with NTK-aware scaling or YaRN to lengths far beyond training. It is used in LLaMA, Mistral, Gemma, Qwen, and most open-source LLMs.
</details>

<details>
<summary>Why use sin and cos instead of just sin?</summary>

Using both gives the encoding a useful linearity property: PE(pos+k) is a linear function of PE(pos). Concretely, the (sin, cos) pair at each frequency lets the model express "shift by k" as a fixed 2×2 rotation matrix — which the model can learn to apply through ordinary linear projections. With sin alone, that shift cannot be linearly recovered.
</details>

## Common mistakes

- Using sinusoidal encoding without adding it to the token embeddings — just computing it without adding is a no-op.
- Not using `register_buffer` for the positional encoding matrix — if it is not registered, it will not move to GPU when `.to(device)` is called, causing device mismatch errors.
- Setting max_len too small and encountering index-out-of-bounds errors at inference time for long inputs.
- Forgetting that positional encoding adds to embeddings, not replaces them — token identity and position are both preserved.
- Mixing sinusoidal PE (added to embeddings) with RoPE (rotates Q/K). Pick one; doubling up is meaningless and usually a bug.

## Final takeaway

Positional encoding is how transformers learn sequence order. Sinusoidal encoding is elegant and generalizes to longer sequences; learned embeddings perform well within the training length. Modern LLMs use RoPE or ALiBi to achieve superior length generalization. Without positional encoding, a transformer is a bag-of-words model that ignores word order entirely.

## References

- Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS.
- Su, J., et al. (2023). RoFormer: Enhanced Transformer with Rotary Position Embedding.
- Press, O., et al. (2022). Train Short, Test Long: Attention with Linear Biases (ALiBi).
- Jay Alammar — [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/).
- Brendan Bycroft — [LLM Visualization](https://bbycroft.net/llm).
