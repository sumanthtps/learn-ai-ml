---
id: rnn-mappings
title: "Types of RNN mappings"
sidebar_label: "58 · RNN Mappings"
sidebar_position: 58
slug: /theory/dnn/types-of-rnn-mappings
description: "One-to-one, one-to-many, many-to-one, and many-to-many RNN architectures — what each is used for, how the input/output is structured, and PyTorch implementations."
tags: [rnn, sequence-modeling, mappings, architectures, deep-learning]
---

# Types of RNN mappings

An RNN processes sequences — but not every task has the same relationship between input and output sequence lengths. Classifying a sentence (many inputs, one output) is structurally different from generating a caption (one input, many outputs) or translating text (many inputs, many outputs). These four structures — one-to-one, one-to-many, many-to-one, many-to-many — define the taxonomy of RNN architectures.

## One-line definition

RNN mappings categorize how input and output sequence lengths relate: one-to-one (fixed), one-to-many (generate from context), many-to-one (summarize a sequence), many-to-many (transform a sequence).

![Unrolled RNN — the same cell processes each time step; the output can be read at the last step (many-to-one) or every step (many-to-many)](https://commons.wikimedia.org/wiki/Special:Redirect/file/Recurrent_neural_network_unfold.svg)
*Source: [Wikimedia Commons — Recurrent neural network unfold](https://commons.wikimedia.org/wiki/File:Recurrent_neural_network_unfold.svg) (CC BY-SA 4.0)*

## The four mapping types

```
One-to-One:      x → y
                 (standard feedforward, no recurrence)

One-to-Many:     x → y₁ y₂ y₃ y₄
                 (one input, generate a sequence)

Many-to-One:     x₁ x₂ x₃ x₄ → y
                 (consume a sequence, one output)

Many-to-Many:    x₁ x₂ x₃ → y₁ y₂ y₃
(synced)         (same-length input and output)

Many-to-Many:    x₁ x₂ x₃ → [encoder] → [decoder] → y₁ y₂ y₄ y₅
(seq2seq)        (different-length input and output, via encoder-decoder)
```

## 1. One-to-many

**Architecture**: single input (or constant input), generate a variable-length output sequence.

**Examples:**
- Image captioning: image → sentence
- Music generation: genre tag → melody sequence
- Story continuation: first word → entire story

**How it works**: The input is used to initialize the hidden state. At each step, the RNN receives either the previous output or a constant input, and generates the next token.

```python
import torch
import torch.nn as nn


class OneToMany(nn.Module):
    """
    Image captioning structure: context vector → sequence of tokens.
    The image embedding initializes h_0; then the RNN generates one token per step.
    """
    def __init__(self, context_dim: int, hidden_size: int, vocab_size: int):
        super().__init__()
        self.context_proj = nn.Linear(context_dim, hidden_size)
        self.rnn = nn.GRUCell(hidden_size, hidden_size)  # input = prev embedding
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, context: torch.Tensor, max_len: int = 20,
                teacher_tokens: torch.Tensor = None) -> torch.Tensor:
        batch = context.size(0)
        h = torch.tanh(self.context_proj(context))   # (B, H) — init from image

        outputs = []
        # Start token is a zero embedding
        token_emb = torch.zeros(batch, h.size(-1), device=context.device)

        for t in range(max_len):
            h = self.rnn(token_emb, h)
            logit = self.out(h)                  # (B, vocab)
            outputs.append(logit)

            if teacher_tokens is not None:        # teacher forcing during training
                token_emb = self.embed(teacher_tokens[:, t])
            else:                                  # autoregressive at inference
                token_emb = self.embed(logit.argmax(-1))

        return torch.stack(outputs, dim=1)       # (B, max_len, vocab)
```

## 2. Many-to-one

**Architecture**: consume an entire input sequence, produce a single output (usually using the final hidden state).

**Examples:**
- Sentiment analysis: review → positive/negative
- Document classification: article → topic label
- Sequence regression: time series → next value prediction

```python
class ManyToOne(nn.Module):
    """
    Sentiment analysis: sequence of word embeddings → class label.
    Uses the final hidden state as the sequence representation.
    """
    def __init__(self, vocab_size: int, embed_dim: int,
                 hidden_size: int, num_classes: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T)
        emb = self.embed(token_ids)               # (B, T, E)
        _, (h_n, _) = self.rnn(emb)               # h_n: (1, B, H) — final hidden
        return self.classifier(h_n.squeeze(0))    # (B, num_classes)
```

## 3. Many-to-many (synchronized)

**Architecture**: output at every step, aligned with the input. Input length = output length.

**Examples:**
- Part-of-speech tagging: word → POS tag (for each word in the sentence)
- Named entity recognition: token → entity label
- Video frame labeling: frame → action label (per frame)

```python
class ManyToManySync(nn.Module):
    """
    Named entity recognition: one output label per input token.
    Input and output sequences have the same length.
    """
    def __init__(self, vocab_size: int, embed_dim: int,
                 hidden_size: int, num_tags: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_size,
                           batch_first=True, bidirectional=True)
        self.tagger = nn.Linear(hidden_size * 2, num_tags)  # ×2 for bidirectional

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T)
        emb = self.embed(token_ids)               # (B, T, E)
        outputs, _ = self.rnn(emb)                # (B, T, 2*H)
        return self.tagger(outputs)               # (B, T, num_tags)
```

## 4. Many-to-many (seq2seq)

**Architecture**: encoder consumes the entire input, decoder generates the output. Input and output lengths can differ.

**Examples:**
- Machine translation: English → French
- Text summarization: article → summary
- Speech recognition: audio frames → text tokens

The encoder-decoder architecture is covered in detail in note 68. The key structural point:

```python
class Seq2Seq(nn.Module):
    """
    Simplified seq2seq: encoder reads input, decoder generates output.
    Input and output lengths are independent.
    """
    def __init__(self, src_vocab: int, tgt_vocab: int,
                 embed_dim: int, hidden_size: int):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, embed_dim)
        self.tgt_embed = nn.Embedding(tgt_vocab, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, tgt_vocab)

    def encode(self, src: torch.Tensor):
        emb = self.src_embed(src)                   # (B, T_src, E)
        _, (h_n, c_n) = self.encoder(emb)           # context: (h_n, c_n)
        return h_n, c_n

    def decode(self, tgt: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        emb = self.tgt_embed(tgt)                   # (B, T_tgt, E)
        out, (h_n, c_n) = self.decoder(emb, (h, c))
        return self.out_proj(out), h_n, c_n         # (B, T_tgt, tgt_vocab)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        h, c = self.encode(src)
        logits, _, _ = self.decode(tgt, h, c)
        return logits
```

## Summary table

| Mapping type | Input | Output | Use case |
|---|---|---|---|
| One-to-one | Fixed | Fixed | Standard MLP (no recurrence needed) |
| One-to-many | Single context | Sequence | Image captioning, text generation |
| Many-to-one | Sequence | Single | Sentiment, classification, regression |
| Many-to-many (sync) | Sequence | Same-length sequence | POS tagging, NER, frame labeling |
| Many-to-many (seq2seq) | Sequence | Variable sequence | Translation, summarization, ASR |

## How the hidden state carries information

In many-to-one, the final hidden state must compress the entire input sequence into a fixed-size vector:

```
"The movie was absolutely brilliant" → h₅ → positive
```

This compression is the fundamental limitation of basic seq2seq — a single vector cannot carry all information for long sequences. This is precisely what attention (note 69) and transformers address.

## PyTorch shape reference

```python
import torch
import torch.nn as nn

batch, T_in, T_out = 4, 10, 7
embed_dim, hidden_size = 64, 128
vocab_src, vocab_tgt = 1000, 800

# Many-to-one: use final hidden state
rnn = nn.LSTM(embed_dim, hidden_size, batch_first=True)
x = torch.randn(batch, T_in, embed_dim)
out, (h_n, c_n) = rnn(x)
# out: (batch, T_in, hidden_size)  — all hidden states
# h_n: (1, batch, hidden_size)     — final hidden state (for many-to-one)
print(f"All hidden states: {out.shape}")       # (4, 10, 128)
print(f"Final hidden state: {h_n.shape}")      # (1, 4, 128)

# Many-to-many (sync): use all hidden states
print(f"Per-step output: {out.shape}")         # (4, 10, 128) — one per input step
```

## Interview questions

<details>
<summary>Why does many-to-one use the final hidden state rather than all hidden states?</summary>

The final hidden state $h_T$ has processed all $T$ input tokens — it is the RNN's summary of the entire sequence. Using it for classification means the network must compress the sequence into this single vector. Earlier hidden states only reflect the sequence up to that point; the final state sees everything. In practice, using the final hidden state works for short sequences but fails for very long ones because the LSTM/GRU cell can only retain a limited amount of information. This is why bidirectional LSTMs (concatenating forward and backward final states) and attention mechanisms improve long-document classification.
</details>

<details>
<summary>What is teacher forcing in a one-to-many RNN and why does it cause exposure bias?</summary>

Teacher forcing: during training, instead of feeding the RNN's own predicted token back as input at the next step, we feed the ground-truth token. This stabilizes training — early in training, the model's predictions are wrong, and feeding wrong predictions back creates compounding errors. However, it creates **exposure bias**: during inference, the model sees its own (possibly wrong) predictions, but during training it saw ground-truth tokens. The model was never trained to recover from its own mistakes. Solutions include scheduled sampling (mix teacher-forced and free-running with decreasing probability) and Professor Forcing.
</details>

## Final takeaway

The four RNN mapping types — one-to-many, many-to-one, many-to-many (sync), many-to-many (seq2seq) — determine the architectural choices: where to read input, where to produce output, and whether an encoder-decoder split is needed. Many-to-one classification uses the final hidden state. Synchronized tagging uses all hidden states. Seq2seq with encoder-decoder handles variable-length pairs. Understanding these structures is essential for choosing the right architecture before building any sequence model.
