---
id: why-self-attention
title: "Why self-attention is called self-attention"
sidebar_label: "76 · Why Self-Attention"
sidebar_position: 76
slug: /theory/dnn/why-self-attention-is-called-self-attention
description: "The 'self' in self-attention means Q, K, and V all come from the same sequence — contrasted with cross-attention where queries come from one sequence and keys/values from another."
tags: [self-attention, cross-attention, transformers, attention, deep-learning]
---

# Why self-attention is called self-attention

> **TL;DR.** "Self-attention" decomposes cleanly. **"Attention"** because it follows the exact same three-step recipe as classical Bahdanau/Luong attention from 2015 (score → softmax → weighted sum). **"Self"** because Q, K, and V all come from the *same* sequence — a sequence attends to itself, rather than one sequence attending to another. Cross-attention is the contrast: Q from one sequence (decoder), K and V from another (encoder).

## Try it interactively

- **[Hugging Face MarianMT](https://huggingface.co/docs/transformers/model_doc/marian)** — load a translation model and inspect both self-attention (encoder & decoder) AND cross-attention in the same model
- **[BertViz](https://github.com/jessevig/bertviz)** — visualize self vs cross-attention patterns side by side
- **[Lena Voita — Seq2Seq with Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)** — interactive walkthrough of Bahdanau attention (the ancestor of self-attention)
- **[Tensor2Tensor cross-attention demo](https://colab.research.google.com/github/tensorflow/tensor2tensor/)** — visualize the rectangular cross-attention matrix on translation

The name has two pieces, and each carries a precise meaning:

1. **Why "attention"?** — Because self-attention performs the *exact same three operations* as classical Bahdanau / Luong attention: compute query-key alignment, normalize via softmax, weight-sum the values. The mathematical machinery is identical.
2. **Why "self"?** — Because queries, keys, and values all come from the **same sequence**. Unlike encoder-decoder attention, which connects two different sequences, self-attention is a sequence attending to itself.

Understanding both halves clarifies the role of each attention type in the full transformer architecture.

## One-line definition

Self-attention is attention where a sequence attends to itself — $Q$, $K$, and $V$ are all computed from the same input $X$, so each token can look at every other token in the same sequence to build its contextual representation.

![Self-attention visualization showing multiple heads — each head in self-attention draws Q, K, V from the same input sequence, while cross-attention (in the decoder) draws Q from one sequence and K, V from another](https://jalammar.github.io/images/t/transformer_self-attention_visualization_3.png)
*Source: [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)*

## Why this topic matters

The "self" vs. "cross" distinction determines what information a model can access at each layer. Self-attention builds rich within-sequence context; cross-attention bridges two sequences. Knowing which type is used where explains why encoder-only models (BERT) are good at understanding, why decoder-only models (GPT) generate autoregressively, and why encoder-decoder models (T5) work for translation.

## Why is it called "attention"?

Self-attention inherits the name from the original Bahdanau / Luong attention used in seq2seq models. Despite looking very different at first glance — no encoder, no decoder, no recurrent hidden states — the underlying mathematics is the same.

### Quick recap of seq2seq attention

In Bahdanau's setup, an encoder LSTM produces hidden states $h_1, \ldots, h_T$ for the input sentence. At each decoder step $i$, the decoder needs to focus on different parts of the input. Instead of using a single fixed context vector, attention computes a fresh context vector $c_i$ each step:

1. **Alignment scores:** $e_{ij} = \text{align}(s_i, h_j) = s_i \cdot h_j$
2. **Normalization:** $\alpha_{ij} = \text{softmax}(e_{ij})$
3. **Context vector:** $c_i = \sum_j \alpha_{ij} \cdot h_j$

Here $s_i$ (decoder state) plays the role of "what am I asking?", and $h_j$ (encoder states) play both "what do I advertise?" and "what content do I carry?".

### Self-attention written in the same form

Now apply self-attention to a single sentence (no encoder, no decoder), say "Turn off the lights". For the contextual embedding of "turn":

1. **Alignment scores:** $e_{1j} = q_{\text{turn}} \cdot k_j$ for each $j \in \{\text{turn, off, the, lights}\}$
2. **Normalization:** $\alpha_{1j} = \text{softmax}(e_{1j})$
3. **Output:** $y_{\text{turn}} = \sum_j \alpha_{1j} \cdot v_j$

The three steps are mathematically identical to Bahdanau's. The only thing that changed is *where* Q, K, V come from:

| Step | Bahdanau / Luong attention | Self-attention |
|------|----------------------------|----------------|
| 1. Alignment | $e_{ij} = s_i \cdot h_j$ | $e_{ij} = q_i \cdot k_j$ |
| 2. Normalization | $\alpha_{ij} = \text{softmax}(e_{ij})$ | $\alpha_{ij} = \text{softmax}(e_{ij})$ |
| 3. Context vector | $c_i = \sum_j \alpha_{ij} \cdot h_j$ | $y_i = \sum_j \alpha_{ij} \cdot v_j$ |

The role mapping:

| Bahdanau / Luong | Self-attention | Role |
|------------------|----------------|------|
| $s_i$ (decoder state) | $q_i$ (query vector) | Asks: "what info do I need?" |
| $h_j$ (encoder states) | $k_j$ (key vectors) | Answers: "here's what I have" |
| $h_j$ (same vectors) | $v_j$ (value vectors) | Provides actual content |

Because the three operations are identical, self-attention is genuinely a member of the attention family. Architecturally it looks different (no encoder/decoder, learnable QKV projections instead of recurrent states), but the mathematical core is the same.

## Why is it called "self"?

The "self" refers to where Q, K, V come from. In the general attention framework:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The source of $Q$, $K$, $V$ defines the attention type:

| Attention type | Q source | K source | V source | Used in |
|---|---|---|---|---|
| **Self-attention** | Same sequence $X$ | Same sequence $X$ | Same sequence $X$ | Encoder, decoder (first sublayer) |
| **Cross-attention** | Decoder state $H_{\text{dec}}$ | Encoder output $H_{\text{enc}}$ | Encoder output $H_{\text{enc}}$ | Decoder (second sublayer) |
| **Seq2Seq attention** (Bahdanau) | Decoder state | Encoder states | Encoder states | Pre-transformer models |

In self-attention, the sequence attends to itself:

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V \quad \text{(all from the same } X\text{)}
$$

Bahdanau's attention is **inter-sequence** (between two different sequences — input and output). Self-attention is **intra-sequence** (within one sequence).

| Property | Bahdanau / Luong | Self-attention |
|----------|------------------|----------------|
| Number of sequences | 2 (different) | 1 (same) |
| Query source | Decoder hidden state ($s_i$) | Query vector from same word ($q_i$) |
| Key source | Encoder hidden states ($h_j$) | Key vectors from same sequence ($k_j$) |
| Value source | Encoder hidden states ($h_j$) | Value vectors from same sequence ($v_j$) |
| Purpose | Align encoder and decoder | Learn relationships within a sequence |
| Architecture | Requires encoder + decoder | Standalone |
| Type | Inter-sequence | **Intra-sequence (self)** |

## What "self" enables: within-sequence context

Because Q, K, and V all come from the same sequence, every token can gather information from every other token in the same sequence.

**Example sentence**: *"The trophy didn't fit in the suitcase because it was too small."*

What does "it" refer to — the trophy or the suitcase? To resolve this, the model must relate "it" to both "trophy" and "suitcase" and use semantic knowledge about size. Self-attention does this directly: the query for "it" can attend to both noun tokens, compare their key representations, and blend the relevant information.

```mermaid
flowchart TD
    sentence["The trophy didn't fit in the suitcase because it was too small"]
    trophy["trophy (key)"]
    suitcase["suitcase (key)"]
    it["it (query)"]
    context["Contextual repr. of 'it'\n= blend of all values,\nweighted by query-key similarity"]

    it -->|"high attention weight"| trophy
    it -->|"moderate attention"| suitcase
    it -->|"low attention"| sentence
    trophy --> context
    suitcase --> context
```

## Cross-attention: attending to a different sequence

In the encoder-decoder transformer, the decoder contains a **cross-attention** sublayer where queries come from the decoder's current state and keys/values come from the encoder's output. The decoder asks: "given what I've generated so far, which parts of the input are most relevant?"

$$
Q = H_{\text{dec}} W^Q, \quad K = H_{\text{enc}} W^K, \quad V = H_{\text{enc}} W^V
$$

```mermaid
flowchart LR
    subgraph "Self-attention (Q, K, V same source)"
        SA_X["Sequence X"] -->|"W_Q"| SA_Q["Q"]
        SA_X -->|"W_K"| SA_K["K"]
        SA_X -->|"W_V"| SA_V["V"]
        SA_Q & SA_K & SA_V --> SA_out["Output: X attends to X"]
    end
    subgraph "Cross-attention (Q and K/V different sources)"
        CA_dec["Decoder state H_dec"] -->|"W_Q"| CA_Q["Q"]
        CA_enc["Encoder output H_enc"] -->|"W_K"| CA_K["K"]
        CA_enc -->|"W_V"| CA_V["V"]
        CA_Q & CA_K & CA_V --> CA_out["Output: decoder attends to encoder"]
    end
```

Notice that cross-attention is essentially a transformer-era reincarnation of Bahdanau attention: queries from the decoder, keys and values from the encoder, same three-step formula.

## Three attention patterns in the full transformer

| Position | Type | What it does |
|---|---|---|
| Encoder (all layers) | Self-attention | Build rich contextual representations of the input |
| Decoder (odd sublayers) | Masked self-attention | Build causal representations of the output so far |
| Decoder (even sublayers) | Cross-attention | Look up relevant input information from encoder |

In a GPT-style decoder-only model, there is no encoder — only causal self-attention at every layer. In a BERT-style encoder-only model, there is only bidirectional self-attention — no masking, no cross-attention.

## Why encoder self-attention is bidirectional

The encoder has no generation task — it processes the full input at once. Token $i$ can attend to all other tokens including those that appear later in the sequence. This bidirectional context produces richer representations:

- "bank" in "I went to the bank to withdraw money" correctly captures the financial sense because it attends to both "withdrew" (right side) and "money"
- A right-only or left-only model would miss half the context

## Why decoder self-attention is causal (masked)

The decoder generates tokens one by one. At training time, the whole target sequence is processed in parallel using teacher forcing — but with a **causal mask** that prevents position $t$ from attending to positions $> t$:

$$
\text{CausalAttn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V
$$

where $M[i, j] = -\infty$ if $j > i$, else $0$.

Without the causal mask, the model can "cheat" by looking at the answer during training. The mask enforces the same condition as inference — each token can only see its own past.

## Python code: self vs cross-attention shapes

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def self_attention(x: torch.Tensor, W_Q, W_K, W_V) -> torch.Tensor:
    """
    Self-attention: Q, K, V all from the same sequence x.
    x: (batch, seq_len, d_model)
    """
    Q = x @ W_Q   # (batch, seq_len, d_k)
    K = x @ W_K
    V = x @ W_V
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    return F.softmax(scores, dim=-1) @ V   # (batch, seq_len, d_k)


def cross_attention(query_seq, memory_seq, W_Q, W_K, W_V) -> torch.Tensor:
    """
    Cross-attention: Q from query_seq, K/V from memory_seq.
    query_seq: (batch, tgt_len, d_model)   — decoder state
    memory_seq: (batch, src_len, d_model)  — encoder output
    """
    Q = query_seq @ W_Q    # (batch, tgt_len, d_k)
    K = memory_seq @ W_K   # (batch, src_len, d_k)
    V = memory_seq @ W_V   # (batch, src_len, d_k)
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (batch, tgt_len, src_len)
    return F.softmax(scores, dim=-1) @ V   # (batch, tgt_len, d_k)


# Parameters
d_model, d_k = 64, 32
W_Q = torch.randn(d_model, d_k) * 0.1
W_K = torch.randn(d_model, d_k) * 0.1
W_V = torch.randn(d_model, d_k) * 0.1

# Self-attention: encoder processing a 10-token input
encoder_input = torch.randn(2, 10, d_model)
enc_output = self_attention(encoder_input, W_Q, W_K, W_V)
print(f"Self-attention: input={encoder_input.shape} → output={enc_output.shape}")
# (2, 10, 32) — same number of tokens, each has attended to all 10 tokens

# Cross-attention: decoder (5 tokens generated so far) attending to encoder (10 tokens)
decoder_state = torch.randn(2, 5, d_model)
enc_memory = torch.randn(2, 10, d_model)
cross_output = cross_attention(decoder_state, enc_memory, W_Q, W_K, W_V)
print(f"Cross-attention: query={decoder_state.shape}, memory={enc_memory.shape} → output={cross_output.shape}")
# (2, 5, 32) — 5 decoder tokens, each has attended to 10 encoder tokens


# Using PyTorch's built-in MultiheadAttention
mha = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)

# Self-attention: same tensor for Q, K, V
src = torch.randn(2, 10, 64)
self_out, self_weights = mha(src, src, src)
print(f"\nBuilt-in self-attention:  weights shape = {self_weights.shape}")  # (2, 10, 10)

# Cross-attention: different tensors for Q vs K/V
tgt = torch.randn(2, 5, 64)
mem = torch.randn(2, 10, 64)
cross_out, cross_weights = mha(tgt, mem, mem)
print(f"Built-in cross-attention: weights shape = {cross_weights.shape}")  # (2, 5, 10)
```

## The attention matrix shape tells you the type

| Attention type | Q length | K length | Score matrix shape |
|---|---|---|---|
| Self-attention | $n$ | $n$ | $(n \times n)$ — square |
| Cross-attention | $m$ (target len) | $n$ (source len) | $(m \times n)$ — rectangular |

A square attention matrix means self-attention. A rectangular matrix means cross-attention — the row dimension is decoder length, the column dimension is encoder length.

## Interview-ready answer

> "Self-attention is called *attention* because it follows the exact same three-step formulation as classical Bahdanau / Luong attention — query-key alignment, softmax normalization, weighted sum of values. It is called *self* because, unlike traditional attention which operates between two different sequences (encoder and decoder), self-attention computes attention **within a single sequence** — each element attends to every other element (including itself) in the same sequence. So it's attention where the sequence pays attention to itself."

## Interview questions

<details>
<summary>What is the precise meaning of "self" in self-attention?</summary>

"Self" means queries, keys, and values are all computed from the same input sequence. The sequence attends to itself — every token can look at every other token in the same sequence. In contrast, cross-attention uses queries from one sequence (decoder) and keys/values from another (encoder). Self-attention builds within-sequence context; cross-attention bridges two sequences.
</details>

<details>
<summary>How is self-attention related to Bahdanau attention?</summary>

The three operations are identical: alignment via dot product, softmax normalization, and weighted sum of values. In Bahdanau attention, the query is the decoder hidden state $s_i$ and the keys/values are the encoder hidden states $h_j$ (both reused). In self-attention, queries, keys, and values are all linear projections of the same input sequence. So self-attention inherits the math but changes the source of Q, K, V — that's exactly why it's still called "attention" but qualified with "self".
</details>

<details>
<summary>Why does the encoder use bidirectional self-attention while the decoder uses causal (masked) self-attention?</summary>

The encoder processes the complete input with no generation task — it can use the full context in both directions, producing the richest possible representations. The decoder generates tokens sequentially (one at a time at inference). At training time, it processes the whole target in parallel, but the causal mask enforces the rule that position $t$ can only see positions $\leq t$ — matching inference conditions and preventing the model from "cheating" by looking at future target tokens.
</details>

<details>
<summary>In a seq2seq transformer, how many attention operations happen per decoder layer?</summary>

Two: (1) masked self-attention — the decoder attends to itself causally, building context from previously generated tokens; (2) cross-attention — the decoder queries the encoder's output to retrieve relevant source information. These are two separate attention blocks, each with its own W_Q, W_K, W_V weights. Self-attention gives the decoder a representation of what it has generated; cross-attention gives it access to what the encoder understood about the source.
</details>

## Common mistakes

- Calling cross-attention "self-attention" because it uses the same formula — the formula is identical but the sources of Q, K, V are different.
- Thinking encoder self-attention is "self" because it's in the encoder — the word "self" refers to Q, K, V source, not the module name.
- Assuming all transformers have cross-attention — decoder-only models (GPT, LLaMA) have no encoder and therefore no cross-attention.
- Treating self-attention as a brand-new mechanism — mathematically it is the same Bahdanau formula with the source of Q, K, V swapped.

## Final takeaway

The name decomposes cleanly: "**attention**" because the three operations (align, softmax, weight-sum) are identical to classical seq2seq attention; "**self**" because Q, K, and V are all computed from the same sequence. A sequence attends to itself. Cross-attention is when Q comes from a different sequence than K and V. In the full transformer, encoder self-attention builds bidirectional input representations, decoder masked self-attention builds causal output representations, and decoder cross-attention bridges input and output via the encoder's memory.

## References

- Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS.
- Bahdanau, D., et al. (2015). Neural Machine Translation by Jointly Learning to Align and Translate.
