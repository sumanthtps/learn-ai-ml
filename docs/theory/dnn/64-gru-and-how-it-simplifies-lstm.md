---
id: gru-simplifies-lstm
title: "GRU and how it simplifies LSTM"
sidebar_label: "64 · GRU"
sidebar_position: 64
slug: /theory/dnn/gru-and-how-it-simplifies-lstm
description: "GRU equations, how it combines the forget and input gate into an update gate, its comparison to LSTM in performance and efficiency, and PyTorch implementation."
tags: [gru, lstm, recurrent-networks, gates, sequence-modeling, deep-learning]
---

# GRU and how it simplifies LSTM

The Gated Recurrent Unit (GRU), introduced by Cho et al. in 2014, achieves similar performance to LSTM with fewer parameters and a simpler structure. It eliminates the separate cell state, merges the forget and input gates into a single update gate, and adds a reset gate. The result is a two-gate recurrent unit that is faster to train and performs comparably to LSTM on most tasks.

## One-line definition

A GRU is a recurrent unit that uses two gates (update and reset) to control information flow, achieving similar long-range memory to LSTM without a separate cell state.

![GRU architecture — update gate and reset gate replace LSTM's three gates](https://commons.wikimedia.org/wiki/Special:Redirect/file/Gated_Recurrent_Unit,_type_1.svg)
*Source: [Wikimedia Commons — Gated Recurrent Unit](https://commons.wikimedia.org/wiki/File:Gated_Recurrent_Unit,_type_1.svg) (CC BY-SA 4.0)*

## GRU equations

Given input $x_t$ and previous hidden state $h_{t-1}$:

**Reset gate** — how much of the previous hidden state to forget:
$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$

**Update gate** — how much to keep from the previous state vs. add new information:
$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$

**Candidate hidden state** — new content to potentially add:
$$
\tilde{h}_t = \tanh\!\left(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h\right)
$$

**New hidden state** — interpolation between old and candidate:
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

## How the GRU gates work

### Update gate $z_t$

The update gate decides how much of the previous hidden state to keep:
- $z_t \approx 0$: keep most of $h_{t-1}$ unchanged (long-term memory preserved)
- $z_t \approx 1$: replace $h_{t-1}$ with new content $\tilde{h}_t$ (update with current input)

This is the GRU's equivalent of LSTM's combination of forget gate (erase old) + input gate (add new). GRU merges these into one:

$$
h_t = (1 - z_t) h_{t-1} + z_t \tilde{h}_t
$$

When $z_t = 0$: $h_t = h_{t-1}$ — perfect memory, no update. This is the GRU's version of LSTM's constant error carousel.

### Reset gate $r_t$

The reset gate controls how much of the previous hidden state influences the candidate:
- $r_t \approx 1$: the candidate $\tilde{h}_t$ is computed with full knowledge of $h_{t-1}$ (standard RNN behavior)
- $r_t \approx 0$: the candidate ignores $h_{t-1}$ — the model computes as if starting fresh

The reset gate lets the model "forget" the past when the current input starts a new, unrelated context.

![GRU variant from colah's LSTM blog — showing how GRU fuses the cell and hidden states](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png)
*Source: [Colah's Blog — Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (CC BY 4.0)*

## GRU vs LSTM comparison

| Property | LSTM | GRU |
|---|---|---|
| Cell state | Yes ($c_t$ separate from $h_t$) | No (single $h_t$) |
| Gates | 3: forget, input, output | 2: update, reset |
| Parameters | $4(E + H) \times H + 4H$ | $3(E + H) \times H + 3H$ |
| Parameter ratio | 1.0× | ~0.75× |
| Output | $h_t$ (from $o_t \odot \tanh(c_t)$) | $h_t$ directly |
| Long-range memory | Excellent | Excellent (comparable) |
| Training speed | Slower | ~25% faster |
| Empirical accuracy | Slightly better on some tasks | Comparable on most tasks |

### Parameter count example

For $E = 100$, $H = 256$:
- LSTM: $4 \times (100 + 256) \times 256 = 364{,}544$ parameters
- GRU: $3 \times (100 + 256) \times 256 = 273{,}408$ parameters — 25% fewer

## PyTorch implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Manual GRU cell — implements equations directly
# ============================================================
class GRUCell(nn.Module):
    """
    Manual implementation of a single GRU cell.
    Shows all gates explicitly.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Reset gate weights
        self.W_r = nn.Linear(input_size, hidden_size, bias=True)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)

        # Update gate weights
        self.W_z = nn.Linear(input_size, hidden_size, bias=True)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)

        # Candidate hidden state weights
        self.W_h = nn.Linear(input_size, hidden_size, bias=True)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor,
                h_prev: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, input_size)
        h_prev: (B, hidden_size)
        returns h_t: (B, hidden_size)
        """
        r = torch.sigmoid(self.W_r(x) + self.U_r(h_prev))         # reset gate
        z = torch.sigmoid(self.W_z(x) + self.U_z(h_prev))         # update gate
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(r * h_prev))  # candidate
        h_t = (1 - z) * h_prev + z * h_tilde                       # new hidden state
        return h_t


# ============================================================
# GRU sequence model using nn.GRU
# ============================================================
class GRUClassifier(nn.Module):
    """
    Many-to-one GRU classifier (e.g., sentiment analysis).
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_size: int = 256, num_layers: int = 2,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor = None) -> torch.Tensor:
        emb = self.dropout(self.embed(x))   # (B, T, E)

        if lengths is not None:
            # Use PackedSequence to ignore padding
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)       # h_n: (num_layers, B, H)
        else:
            _, h_n = self.gru(emb)

        # Use last layer's hidden state
        out = self.dropout(h_n[-1])         # (B, H)
        return self.fc(out)                  # (B, num_classes)


# ============================================================
# GRU vs LSTM: side-by-side comparison
# ============================================================
def compare_gru_lstm(vocab_size: int = 10000, embed_dim: int = 128,
                     hidden_size: int = 256, num_layers: int = 2):
    gru  = nn.GRU(embed_dim, hidden_size, num_layers, batch_first=True)
    lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)

    gru_params  = sum(p.numel() for p in gru.parameters())
    lstm_params = sum(p.numel() for p in lstm.parameters())

    print(f"GRU  parameters: {gru_params:>10,}")
    print(f"LSTM parameters: {lstm_params:>10,}")
    print(f"Parameter reduction: {1 - gru_params/lstm_params:.1%}")

    # Timing comparison
    import time
    x = torch.randn(32, 100, embed_dim)   # batch=32, seq=100, embed=128

    for name, model in [("GRU", gru), ("LSTM", lstm)]:
        # Warmup
        for _ in range(5):
            model(x)
        # Time
        start = time.perf_counter()
        for _ in range(100):
            model(x)
        elapsed = time.perf_counter() - start
        print(f"{name} 100 forward passes: {elapsed:.3f}s")

    return gru, lstm


# ============================================================
# Hidden state shapes reference
# ============================================================
batch, T, E, H, L = 4, 50, 128, 256, 2

x = torch.randn(batch, T, E)

gru = nn.GRU(E, H, L, batch_first=True)
lstm = nn.LSTM(E, H, L, batch_first=True)

gru_out, gru_h = gru(x)
lstm_out, (lstm_h, lstm_c) = lstm(x)

print(f"\nGRU output:     {gru_out.shape}")   # (4, 50, 256)
print(f"GRU h_n:         {gru_h.shape}")      # (2, 4, 256) — num_layers × batch × hidden
print(f"\nLSTM output:    {lstm_out.shape}")   # (4, 50, 256)
print(f"LSTM h_n:         {lstm_h.shape}")     # (2, 4, 256)
print(f"LSTM c_n:         {lstm_c.shape}")     # (2, 4, 256) — extra cell state
```

## When to use GRU vs LSTM

| Scenario | Recommendation |
|---|---|
| General sequence classification | GRU — fewer params, similar accuracy |
| Long documents (> 500 tokens) | LSTM — slightly better long-range memory |
| Speed-critical or resource-constrained | GRU — 25% fewer params, faster |
| Ablation / hyperparameter study | Start with GRU, switch to LSTM if needed |
| PyTorch default for most tasks | `nn.GRU` with 2 layers, bidirectional |
| Modern NLP | Both largely replaced by transformers |

## Gradient flow in GRU

The GRU gradient through the hidden state:

$$
\frac{\partial h_t}{\partial h_{t-1}} = (1 - z_t) + z_t \cdot \text{(terms involving reset gate)}
$$

When $z_t \approx 0$: gradient ≈ 1 — the identity path (same as LSTM's constant error carousel). The GRU achieves gradient preservation through the $(1 - z_t) h_{t-1}$ term in the hidden state update.

## Interview questions

<details>
<summary>How does GRU merge the LSTM forget and input gates?</summary>

In LSTM: $c_t = f_t c_{t-1} + i_t \tilde{c}_t$. The forget gate ($f_t$) erases old content; the input gate ($i_t$) adds new content. They are independent: in principle both could be near 1 simultaneously (add new while keeping old). In GRU: $h_t = (1 - z_t) h_{t-1} + z_t \tilde{h}_t$. The update gate $z_t$ controls both operations with a single value: high $z_t$ means "replace" (forget old, add new), low $z_t$ means "keep" (retain old, ignore new). This is a constrained version: forget and input are coupled via $z_t$ and $1 - z_t$ always summing to 1. The coupling reduces parameters by 1 gate but removes some expressivity.
</details>

<details>
<summary>When would you choose LSTM over GRU?</summary>

GRU and LSTM perform comparably on most sequence tasks. LSTM has a slight edge on: (1) very long sequences where the separate cell state and output gate provide more expressive memory management; (2) tasks requiring fine-grained control over what is output vs. what is stored — the output gate decouples reading from the cell state. GRU wins on: (1) speed and memory — 25% fewer parameters; (2) tasks where the extra expressivity of LSTM is not needed (most short-to-medium sequence classification). In practice, try both and validate on your specific task.
</details>

## Final takeaway

GRU simplifies LSTM by eliminating the cell state and merging the forget/input gates into a single update gate. The result is a two-gate unit with ~25% fewer parameters that achieves comparable performance on most sequence tasks. The update gate implements the same principle as LSTM's gradient highway: when $z_t \approx 0$, the hidden state is copied unchanged, allowing gradients to flow back unimpeded. For most practical applications, GRU is a strong default: faster to train, easier to implement, and performs similarly to LSTM.

## References

- Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.
- Chung, J., et al. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.
