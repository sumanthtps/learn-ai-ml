---
id: deep-stacked-rnn
title: "Deep and stacked recurrent networks"
sidebar_label: "65 · Stacked RNNs"
sidebar_position: 65
slug: /theory/dnn/deep-and-stacked-recurrent-networks
description: "Why stacking RNN layers creates hierarchical representations, how dropout is applied between layers, and the trade-offs between depth and width in recurrent models."
tags: [rnn, lstm, stacked-rnn, depth, hierarchical-features, deep-learning]
---

# Deep and stacked recurrent networks

A single-layer LSTM processes sequences with one level of abstraction. Stacking multiple LSTM layers — where each layer's output sequence becomes the next layer's input — creates a hierarchy of representations: lower layers capture local patterns and syntax, higher layers capture long-range dependencies and semantics. This is the recurrent analog of deep CNNs, where depth enables hierarchical feature learning.

## One-line definition

A stacked (deep) RNN has multiple recurrent layers in series, where the hidden states of layer $l$ at every time step form the input sequence to layer $l+1$, enabling hierarchical sequence representation.

## Architecture

```
Layer 2 (h²_1, h²_2, ..., h²_T):   Semantic / high-level context
         ↑         ↑          ↑
Layer 1 (h¹_1, h¹_2, ..., h¹_T):   Syntactic / local patterns
         ↑         ↑          ↑
Input   (x_1,  x_2,  ...,  x_T)
```

Layer 1 receives the raw input $x_t$ and produces $h^1_t$ at every step. Layer 2 receives the entire sequence $\{h^1_1, h^1_2, \ldots, h^1_T\}$ as its input and produces $h^2_t$. With $L$ layers, the recurrence at layer $l$ is:

$$
h^l_t = \text{LSTM}_l(h^{l-1}_t,\ h^l_{t-1})
$$

where $h^0_t = x_t$ (the raw input).

![Unrolled RNN showing how hidden states propagate through time steps — stacking layers adds depth in the vertical direction](https://commons.wikimedia.org/wiki/Special:Redirect/file/Recurrent_neural_network_unfold.svg)
*Source: [Wikimedia Commons — Recurrent neural network unfold](https://commons.wikimedia.org/wiki/File:Recurrent_neural_network_unfold.svg) (CC BY-SA 4.0)*

## Why depth helps

**Layer 1** sees raw token embeddings — it learns local patterns (word boundaries, short phrases, immediate context).

**Layer 2** sees sequences of hidden states — each $h^1_t$ already encodes local context. Layer 2 learns longer-range patterns that depend on those local summaries.

**Layer 3+** operates on increasingly abstract representations. Empirical findings (Google NMT, 2016):
- 1 layer: baseline
- 2 layers: significant improvement
- 4 layers: further gains
- 8 layers: diminishing returns (with residual connections and dropout)

## Dropout in stacked RNNs

Standard dropout between LSTM layers works by dropping output sequences between layers, not within the recurrent connections:

```
x → [Layer 1 LSTM] → dropout(0.3) → [Layer 2 LSTM] → dropout(0.3) → output
```

Applying dropout to the recurrent connections (inside the LSTM, on $h_{t-1}$) is also possible (variational dropout / recurrent dropout) but more complex. PyTorch's `nn.LSTM` applies dropout between layers only.

## PyTorch implementation

```python
import torch
import torch.nn as nn


# ============================================================
# Stacked LSTM using nn.LSTM built-in num_layers
# ============================================================
batch, T, E, H = 8, 50, 128, 256

# num_layers=3 → three stacked LSTM layers
lstm_stacked = nn.LSTM(
    input_size=E,
    hidden_size=H,
    num_layers=3,
    batch_first=True,
    dropout=0.3,          # applies between layers (not on last layer output)
    bidirectional=False,
)

x = torch.randn(batch, T, E)
output, (h_n, c_n) = lstm_stacked(x)

print(f"Input:  {x.shape}")             # (8, 50, 128)
print(f"Output: {output.shape}")        # (8, 50, 256) — last layer hidden states
print(f"h_n:    {h_n.shape}")           # (3, 8, 256)  — all layers' final hidden
print(f"c_n:    {c_n.shape}")           # (3, 8, 256)  — all layers' final cell

# Per-layer final hidden states
for layer in range(3):
    print(f"  Layer {layer} final h: {h_n[layer].shape}")  # (8, 256) each


# ============================================================
# Manual stacked LSTM — explicit layer-by-layer
# (Useful when you need per-layer access, residual connections, etc.)
# ============================================================
class StackedLSTM(nn.Module):
    """
    Stacked LSTM with optional residual connections between layers.
    Residual connections help gradient flow in very deep (6+) RNNs.
    """
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 4, dropout: float = 0.3,
                 residual: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.residual = residual and (input_size == hidden_size)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.layers.append(nn.LSTM(in_size, hidden_size, batch_first=True))

        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers - 1)
        ])

    def forward(self, x: torch.Tensor,
                hidden: list = None) -> tuple[torch.Tensor, list]:
        """
        x: (batch, T, input_size)
        Returns: output (batch, T, hidden_size), list of (h_n, c_n) per layer
        """
        if hidden is None:
            hidden = [None] * self.num_layers

        all_hidden = []
        current = x

        for i, layer in enumerate(self.layers):
            output, (h_n, c_n) = layer(current, hidden[i])
            all_hidden.append((h_n, c_n))

            # Residual connection (skip connection) — adds previous input to output
            if self.residual and i > 0:
                output = output + current

            # Dropout between layers (not after the last layer)
            if i < self.num_layers - 1:
                current = self.dropouts[i](output)
            else:
                current = output

        return current, all_hidden


# ============================================================
# Depth vs width trade-off experiment
# ============================================================
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


print("\nDepth vs Width parameter counts (E=128, total budget ~2M params):")
configs = [
    ("1 layer H=1024",  nn.LSTM(128, 1024, 1, batch_first=True)),
    ("2 layers H=512",  nn.LSTM(128, 512,  2, batch_first=True)),
    ("4 layers H=256",  nn.LSTM(128, 256,  4, batch_first=True)),
    ("8 layers H=128",  nn.LSTM(128, 128,  8, batch_first=True)),
]

for name, model in configs:
    n = count_params(model)
    print(f"  {name}: {n:>9,} params")


# ============================================================
# Sequence classification with stacked bidirectional GRU
# ============================================================
class DeepBiGRUClassifier(nn.Module):
    """
    3-layer bidirectional GRU for text classification.
    Uses the concatenation of forward and backward final hidden states.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_size: int = 256, num_layers: int = 3,
                 num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        # ×2 because bidirectional: forward + backward hidden
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embed(x))    # (B, T, E)
        _, h_n = self.gru(emb)               # h_n: (num_layers*2, B, H)

        # h_n layout: [fwd_layer0, bwd_layer0, fwd_layer1, bwd_layer1, ...]
        # Take final layer: last two entries (forward + backward)
        fwd = h_n[-2]                        # (B, H) — last layer forward
        bwd = h_n[-1]                        # (B, H) — last layer backward
        combined = torch.cat([fwd, bwd], -1) # (B, 2H)

        return self.fc(self.dropout(combined))  # (B, num_classes)


# Test
model = DeepBiGRUClassifier(vocab_size=10000, num_classes=5)
x = torch.randint(0, 10000, (4, 60))   # batch=4, seq_len=60
out = model(x)
print(f"\nDeepBiGRU output: {out.shape}")   # (4, 5)
params = count_params(model)
print(f"Parameters: {params:,}")
```

## Hidden state structure for stacked RNNs

When `num_layers > 1`, PyTorch returns all layers' hidden states stacked:

```python
# nn.LSTM with num_layers=3, hidden_size=256, batch=4
h_n shape: (3, 4, 256)
#           ↑  ↑  ↑
#         layers batch hidden

# To get each layer's final state:
h_layer1 = h_n[0]   # (4, 256) — layer 1 final hidden
h_layer2 = h_n[1]   # (4, 256) — layer 2 final hidden
h_layer3 = h_n[2]   # (4, 256) — layer 3 final hidden (deepest)
```

For **bidirectional** models, layout is `(num_layers * 2, batch, hidden)`:
```python
# Bidirectional, num_layers=2
h_n: (4, batch, hidden)   # [fwd_l1, bwd_l1, fwd_l2, bwd_l2]
```

## Practical guidelines

| Task | Recommended depth |
|---|---|
| Short sequence classification | 1–2 layers |
| Long document classification | 2–4 layers (bidirectional) |
| Machine translation | 4–8 layers (encoder and decoder) |
| Speech recognition | 4–6 layers (bidirectional encoder) |
| Language modeling | 2–4 layers |
| Any task with > 4 layers | Add residual connections |

## Interview questions

<details>
<summary>Why does stacking RNN layers improve performance?</summary>

Each LSTM layer processes the sequence at a different level of abstraction. Layer 1 takes raw inputs (token embeddings) and produces hidden states that encode local patterns — immediate context, word-level features, short phrases. Layer 2 takes these hidden states as input — each input is already a summary of local context — and can learn patterns at a longer time scale without the complexity of the raw input getting in the way. This hierarchy mirrors how deep CNNs build complex feature detectors from simpler ones. Empirically, 2–4 layers consistently outperform 1 layer on NMT, LM, and speech recognition tasks.
</details>

<details>
<summary>How does dropout work between stacked LSTM layers, and why not inside the recurrent connections?</summary>

PyTorch's `nn.LSTM(dropout=p)` applies dropout to the output of each layer before it is used as input to the next layer — between layers, not on the recurrent hidden-to-hidden connections. Applying dropout to recurrent connections (e.g., on $h_{t-1}$ → $h_t$) would disrupt the gradient highway that makes LSTMs work for long sequences. **Variational dropout** (Gal & Ghahramani, 2016) applies the same dropout mask at every time step for the recurrent connections, which is more principled. For most applications, inter-layer dropout is sufficient and the standard default.
</details>

## Common mistakes

- Accessing only `h_n[-1]` for classification when using bidirectional stacked RNN — you need `h_n[-2]` (forward, last layer) and `h_n[-1]` (backward, last layer) and concatenate them
- Not using dropout between layers for deep (> 2 layer) RNNs — deep RNNs overfit without it
- Using more than 4 layers without residual connections — gradient propagation through 6+ LSTM layers stacked vertically (in addition to temporal depth) requires skip connections

## Final takeaway

Stacking RNN layers creates hierarchical representations: lower layers capture local patterns, higher layers capture longer-range dependencies. PyTorch's `num_layers` parameter handles this automatically. Dropout is applied between layers (not on recurrent connections) to regularize. For very deep stacks (6+), residual connections between layers are needed for stable training. In practice, 2–4 layers with bidirectional encoding is the standard recipe for most sequence tasks.
