---
id: bidirectional-rnn-bilstm
title: "Bidirectional RNNs and BiLSTMs"
sidebar_label: "66 · BiLSTM"
sidebar_position: 66
slug: /theory/dnn/bidirectional-rnns-and-bilstms
description: "How bidirectional RNNs process sequences in both directions, why they produce richer token representations, and when you cannot use them."
tags: [bilstm, bidirectional-rnn, sequence-modeling, nlp, encoders, deep-learning]
---

# Bidirectional RNNs and BiLSTMs

A standard (unidirectional) LSTM processes a sequence left-to-right. At position $t$, the hidden state $h_t$ encodes information from $x_1, \ldots, x_t$ but nothing about $x_{t+1}, \ldots, x_T$. For many tasks — understanding a word in the context of the full sentence — this is too restrictive. A bidirectional LSTM (BiLSTM) runs two LSTMs: one left-to-right and one right-to-left. Their hidden states are concatenated, giving each position a representation informed by both past and future context.

## One-line definition

A bidirectional RNN runs two independent recurrent networks in opposite directions over the same sequence and concatenates their hidden states at each position, giving each token access to both left and right context.

## Architecture

```
Forward  LSTM:   h→₁  h→₂  h→₃  h→₄  h→₅   (left to right)
                  ↑    ↑    ↑    ↑    ↑
Input:           x₁   x₂   x₃   x₄   x₅
                  ↓    ↓    ↓    ↓    ↓
Backward LSTM:   h←₁  h←₂  h←₃  h←₄  h←₅   (right to left)

Output at t:  [h→_t ; h←_t]   (concatenation, size = 2H)
```

At position $t$:
- $\overrightarrow{h}_t$ encodes $x_1, \ldots, x_t$ (left context)
- $\overleftarrow{h}_t$ encodes $x_T, \ldots, x_t$ (right context)
- $h_t = [\overrightarrow{h}_t \; \| \; \overleftarrow{h}_t]$ encodes full sentence context at position $t$

![LSTM cell architecture — the same unit used in each direction of a BiLSTM](https://commons.wikimedia.org/wiki/Special:Redirect/file/Long_Short-Term_Memory.svg)
*Source: [Wikimedia Commons — Long Short-Term Memory](https://commons.wikimedia.org/wiki/File:Long_Short-Term_Memory.svg) (CC BY-SA 4.0)*

## Why bidirectional context matters

Consider named entity recognition. For the sentence:

```
"Apple announced a new iPhone yesterday"
```

Processing left-to-right, when we reach "Apple" at position 1, we have no future context — is "Apple" the fruit or the company? The full sentence makes it clear ("iPhone"). The backward LSTM sees "yesterday → iPhone → new → a → announced → Apple", so when it produces $\overleftarrow{h}_1$, it has already seen all the disambiguating context.

**Tasks that benefit from BiLSTM:**
- Named entity recognition (NER)
- Part-of-speech tagging
- Text encoding (for classification, Q&A, NLI)
- Reading comprehension encoders
- Machine translation encoders

**Tasks that cannot use BiLSTM:**
- Language modeling (next-word prediction) — you cannot see the future
- Autoregressive generation — at step $t$, future tokens don't exist yet
- Any online/streaming task — input arrives one token at a time

## PyTorch implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic BiLSTM with nn.LSTM
# ============================================================
batch, T, E, H = 4, 20, 128, 256

bilstm = nn.LSTM(
    input_size=E,
    hidden_size=H,
    num_layers=2,
    batch_first=True,
    dropout=0.3,
    bidirectional=True,   # ← the key flag
)

x = torch.randn(batch, T, E)
output, (h_n, c_n) = bilstm(x)

print(f"Input:   {x.shape}")              # (4, 20, 128)
print(f"Output:  {output.shape}")         # (4, 20, 512)  — 2H because bidirectional
print(f"h_n:     {h_n.shape}")            # (4, 4, 256)   — num_layers*2 × batch × H


# ============================================================
# Extracting the final hidden state correctly for BiLSTM
# ============================================================
num_layers = 2

# h_n layout for bidirectional, 2 layers:
# [fwd_layer0, bwd_layer0, fwd_layer1, bwd_layer1]
# indices:  0,           1,           2,            3

# For classification, use the LAST layer (deepest):
fwd_last = h_n[-2]    # (batch, H) — forward direction, last layer
bwd_last = h_n[-1]    # (batch, H) — backward direction, last layer
sentence_repr = torch.cat([fwd_last, bwd_last], dim=-1)   # (batch, 2H)
print(f"\nSentence representation: {sentence_repr.shape}")  # (4, 512)


# ============================================================
# BiLSTM for Named Entity Recognition (NER)
# ============================================================
class BiLSTMTagger(nn.Module):
    """
    BiLSTM for sequence labeling (NER, POS tagging).
    Produces one output label per input token.
    Architecture: Embed → BiLSTM → Dropout → Linear (per token)
    """
    def __init__(self, vocab_size: int, embed_dim: int = 100,
                 hidden_size: int = 200, num_layers: int = 2,
                 num_tags: int = 9,     # e.g., BIO tags: B-PER, I-PER, B-ORG, ...
                 dropout: float = 0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embed_dim, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        # Each position gets a tag: output is (B, T, num_tags)
        self.fc = nn.Linear(hidden_size * 2, num_tags)

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor = None) -> torch.Tensor:
        emb = self.dropout(self.embed(x))   # (B, T, E)

        if lengths is not None:
            # Pack to handle variable-length sequences efficiently
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.bilstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                out_packed, batch_first=True
            )
        else:
            output, _ = self.bilstm(emb)

        # output: (B, T, 2H) — both directions at each position
        output = self.dropout(output)
        return self.fc(output)   # (B, T, num_tags)


# ============================================================
# BiLSTM for sentence pair tasks (NLI, semantic similarity)
# ============================================================
class SiameseBiLSTM(nn.Module):
    """
    Encode two sentences with shared BiLSTM, then compare representations.
    Used for: textual entailment, paraphrase detection, sentence similarity.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 hidden_size: int = 512, num_classes: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_size, 2,
                               batch_first=True, bidirectional=True)
        # Classifier on combined representations
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2 * 4, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sentence to a fixed vector via mean pooling over positions."""
        emb = self.embed(x)             # (B, T, E)
        out, _ = self.encoder(emb)      # (B, T, 2H)
        return out.mean(dim=1)          # (B, 2H) — mean pooled

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        u = self.encode(x1)             # (B, 2H)
        v = self.encode(x2)             # (B, 2H)
        # InferSent-style combination: [u; v; |u-v|; u*v]
        combined = torch.cat([u, v, (u - v).abs(), u * v], dim=-1)  # (B, 8H)
        return self.fc(combined)        # (B, num_classes)


# ============================================================
# Shape reference
# ============================================================
print("\nBidirectional shape reference:")
configs = [
    (False, 1, "Unidirectional 1-layer"),
    (False, 2, "Unidirectional 2-layer"),
    (True,  1, "Bidirectional 1-layer"),
    (True,  2, "Bidirectional 2-layer"),
]
x_test = torch.randn(4, 20, 128)
for bidi, n_layers, name in configs:
    model = nn.LSTM(128, 256, n_layers, batch_first=True, bidirectional=bidi)
    out, (h, c) = model(x_test)
    print(f"  {name}: out={out.shape}, h_n={h.shape}")
```

## Mean pooling vs final hidden state

For classification with BiLSTM:

**Final hidden state**: concatenate forward's last hidden and backward's last hidden. Good for tasks where the final context summary is most relevant.

**Mean pooling**: average all output hidden states across time. Better for tasks where every position contributes equally to the sentence meaning (e.g., document classification where any part may be relevant).

**Max pooling**: take the maximum over time for each dimension. Good for detecting whether any position had a strong activation.

```python
# In a BiLSTM classifier
output, (h_n, _) = bilstm(emb)   # output: (B, T, 2H)

# Option 1: Final hidden state (standard for sentence classification)
repr_final = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, 2H)

# Option 2: Mean pooling
repr_mean = output.mean(dim=1)    # (B, 2H)

# Option 3: Max pooling
repr_max = output.max(dim=1).values  # (B, 2H)
```

## Interview questions

<details>
<summary>When can you not use a bidirectional RNN?</summary>

Bidirectional RNNs require the entire sequence to be available before processing — the backward LSTM needs to start from the last token. This makes them unsuitable for: (1) **Language modeling / next-word prediction**: at step $t$, only $x_1, \ldots, x_t$ are available, not future tokens; (2) **Autoregressive generation**: the model generates tokens one at a time and cannot look at tokens it hasn't generated yet; (3) **Streaming/online processing**: the input arrives one token at a time and the model must produce output immediately. Bidirectional models work for offline tasks where the full input is known: text classification, NER, machine translation encoding, reading comprehension.
</details>

<details>
<summary>Why does BiLSTM perform better than unidirectional LSTM for NER?</summary>

In NER, classifying a token requires both its left context (what preceded it) and right context (what follows it). "Washington" could be a person, a place, or an organization — the neighboring words disambiguate. A unidirectional left-to-right LSTM at position $t$ only knows what came before. A BiLSTM at position $t$ has $\overrightarrow{h}_t$ (left context) and $\overleftarrow{h}_t$ (right context) available simultaneously. Concatenating them gives the classifier a complete sentence-level context at every token position, which is why BiLSTM + CRF (the classic NER architecture) significantly outperforms unidirectional LSTM for token labeling tasks.
</details>

## Common mistakes

- Misreading `h_n` for a bidirectional LSTM — the shape is `(num_layers * 2, batch, hidden)`. The last forward layer is `h_n[-2]`, the last backward layer is `h_n[-1]`
- Using `output[:, -1, :]` as the representation — this is the last time step's output, which only contains the forward direction's context for that final position. Use `torch.cat([h_n[-2], h_n[-1]], -1)` for the full bidirectional sentence embedding
- Using BiLSTM for language modeling — produces information leakage (the backward pass sees future tokens), making the perplexity artificially low but useless at inference time

## Final takeaway

Bidirectional LSTMs process sequences in both directions and concatenate the hidden states, giving each token access to full sentence context. They are the standard encoder for NLP tasks that require understanding a token in context of the complete input: NER, POS tagging, sentence encoding, reading comprehension. They cannot be used for autoregressive generation or streaming tasks where future context is unavailable. The key implementation detail: for classification, use `torch.cat([h_n[-2], h_n[-1]], dim=-1)` to get the last layer's bidirectional representation.
