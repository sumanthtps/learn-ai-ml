---
id: lstm-next-word-prediction
title: "Next-word prediction with an LSTM in PyTorch"
sidebar_label: "63 · LSTM Language Model"
sidebar_position: 63
slug: /theory/dnn/next-word-prediction-with-an-lstm-in-pytorch
description: "Building a character-level and word-level LSTM language model from scratch: tokenization, dataset, model, training loop, and text generation."
tags: [lstm, language-model, next-word-prediction, pytorch, text-generation, deep-learning]
---

# Next-word prediction with an LSTM in PyTorch

Next-word prediction — training a model to predict the next token given previous tokens — is the original language modeling task. It predates transformers and GPT by two decades. LSTM language models are conceptually simple, computationally tractable, and a natural bridge from sequence modeling theory to modern LLMs. This note builds a complete LSTM language model from scratch: tokenization, dataset, model, training, and text generation.

![The LSTM chain — at each step the model reads a token, updates the hidden and cell state, and produces a distribution over the next token](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)
*Source: [Colah's Blog — Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (CC BY 4.0)*

## How LSTM language modeling works

Given a sequence of tokens $w_1, w_2, \ldots, w_T$, train the model to maximize:

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(w_t \mid w_1, \ldots, w_{t-1})
$$

This is the **cross-entropy language modeling loss** — the same objective used by GPT.

In practice:
- Input: $[w_1, w_2, \ldots, w_{T-1}]$
- Target: $[w_2, w_3, \ldots, w_T]$
- At each step $t$, predict $w_{t+1}$ given all previous context encoded in $h_t$

## Character-level LSTM language model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter


# ============================================================
# Character-level tokenization
# ============================================================
class CharTokenizer:
    """Simple character-level vocabulary."""
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab = {c: i for i, c in enumerate(chars)}
        self.ivocab = {i: c for c, i in self.vocab.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.vocab[c] for c in text if c in self.vocab]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.ivocab.get(i, "?") for i in ids)


# ============================================================
# Sliding window dataset
# ============================================================
class SequenceDataset(Dataset):
    """
    Creates overlapping windows of length seq_len.
    Input: x[0:seq_len], Target: x[1:seq_len+1]
    """
    def __init__(self, token_ids: list[int], seq_len: int = 64):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


# ============================================================
# LSTM language model
# ============================================================
class LSTMLanguageModel(nn.Module):
    """
    Character/word level LSTM language model.
    Architecture: Embedding → LSTM (multi-layer) → Dropout → Linear → logits
    """
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Weight tying: share embedding and output projection weights
        # (reduces parameters, often improves perplexity)
        if embed_dim == hidden_size:
            self.fc.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor,
                hidden: tuple = None) -> tuple[torch.Tensor, tuple]:
        """
        x: (batch, seq_len) — token ids
        Returns logits (batch, seq_len, vocab) and new hidden state.
        """
        emb = self.embed(x)                   # (B, T, E)
        out, hidden = self.lstm(emb, hidden)   # (B, T, H), (h_n, c_n)
        out = self.dropout(out)
        logits = self.fc(out)                  # (B, T, vocab)
        return logits, hidden

    def init_hidden(self, batch_size: int,
                    device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states to zeros."""
        zeros = torch.zeros(self.num_layers, batch_size,
                            self.hidden_size, device=device)
        return zeros, zeros.clone()

    def detach_hidden(self, hidden: tuple) -> tuple:
        """Detach hidden state from computation graph between batches."""
        return tuple(h.detach() for h in hidden)


# ============================================================
# Training loop with perplexity tracking
# ============================================================
def train_language_model(text: str, seq_len: int = 64, batch_size: int = 64,
                          num_epochs: int = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize
    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)
    print(f"Vocab size: {tokenizer.vocab_size}, Tokens: {len(token_ids):,}")

    # Dataset
    dataset = SequenceDataset(token_ids, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True)

    # Model
    model = LSTMLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64, hidden_size=256, num_layers=2, dropout=0.3,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(loader)
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        hidden = model.init_hidden(batch_size, device)

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            hidden = model.detach_hidden(hidden)   # detach between batches

            optimizer.zero_grad()
            logits, hidden = model(x, hidden)      # (B, T, V)

            # Reshape for loss: (B*T, V) and (B*T,)
            loss = criterion(
                logits.reshape(-1, tokenizer.vocab_size),
                y.reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        perplexity = np.exp(avg_loss)
        print(f"Epoch {epoch:2d}: loss={avg_loss:.4f}  perplexity={perplexity:.1f}")

    return model, tokenizer


# ============================================================
# Text generation
# ============================================================
@torch.no_grad()
def generate_text(model: LSTMLanguageModel, tokenizer: CharTokenizer,
                  seed: str, length: int = 200,
                  temperature: float = 0.8,
                  device: str = "cpu") -> str:
    """
    Autoregressively generate text from a seed string.
    temperature < 1: more focused/predictable
    temperature > 1: more random/creative
    """
    model.eval()
    model = model.to(device)

    # Encode seed
    ids = tokenizer.encode(seed)
    x = torch.tensor(ids, device=device).unsqueeze(0)   # (1, seed_len)

    # Warm up the hidden state on the seed
    hidden = model.init_hidden(1, device)
    with torch.no_grad():
        _, hidden = model(x, hidden)

    # Generate
    generated = list(ids)
    current_id = torch.tensor([[ids[-1]]], device=device)

    for _ in range(length):
        logits, hidden = model(current_id, hidden)   # (1, 1, V)
        logits = logits[0, 0] / temperature          # (V,)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        generated.append(next_id)
        current_id = torch.tensor([[next_id]], device=device)

    return tokenizer.decode(generated)


# ============================================================
# Perplexity as a metric
# ============================================================
@torch.no_grad()
def compute_perplexity(model: LSTMLanguageModel, tokenizer: CharTokenizer,
                       text: str, seq_len: int = 64,
                       device: str = "cpu") -> float:
    """
    Perplexity = exp(average cross-entropy loss per token).
    Lower is better. A random model on V-class vocab has perplexity = V.
    """
    model.eval()
    ids = tokenizer.encode(text)
    total_loss = 0.0
    n_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for i in range(0, len(ids) - seq_len, seq_len):
        x = torch.tensor(ids[i : i + seq_len],
                         device=device).unsqueeze(0)       # (1, T)
        y = torch.tensor(ids[i + 1 : i + seq_len + 1],
                         device=device).unsqueeze(0)       # (1, T)
        logits, _ = model(x)                               # (1, T, V)
        loss = criterion(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
        total_loss += loss.item() * seq_len
        n_tokens += seq_len

    return np.exp(total_loss / n_tokens)
```

## What perplexity measures

Perplexity is the standard language model metric:

$$
\text{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^T \log P(w_t \mid w_{<t})\right)
$$

Intuitively: how "surprised" is the model at the test text on average. A perplexity of $k$ means the model is as uncertain as if it were choosing uniformly from $k$ options at each step.

| PPL (character level) | Interpretation |
|---|---|
| Equal to vocab size | Random model |
| ~50–100 | Clearly learning some patterns |
| ~10–30 | Decent character LM |
| ~2–5 | Very good character LM |

## Why `detach_hidden` between batches

During training, each mini-batch processes a window of the text. We reuse the hidden state from the previous batch (truncated BPTT) to maintain continuity. However, we detach it from the computation graph to prevent gradients from flowing back through previous batches (which would require storing the entire history):

```python
hidden = model.detach_hidden(hidden)  # breaks gradient graph, keeps values
```

Without this, PyTorch would try to backpropagate through all previous batches, consuming enormous memory and causing "graph was freed" errors.

## Interview questions

<details>
<summary>What is truncated BPTT and why is it used in language model training?</summary>

Full BPTT (Backpropagation Through Time) unrolls the entire sequence and computes gradients from the last step all the way to the first. For a long document (thousands of tokens), this requires storing the computation graph for every single step — prohibitive in memory. Truncated BPTT unrolls only $k$ steps and computes gradients over that window, then detaches the hidden state and continues. The values of the hidden state are preserved (maintaining continuity of the sequence), but gradients only flow $k$ steps back. This limits gradient-based learning of dependencies longer than $k$ steps, but is essential for practical training.
</details>

<details>
<summary>What is temperature in text generation and what does setting it to 0 do?</summary>

Temperature $\tau$ divides the logits before softmax: $p_i = \text{softmax}(z_i / \tau)$. High temperature ($\tau > 1$) flattens the distribution — tokens with moderate probability become more likely, producing more varied and creative text. Low temperature ($0 < \tau < 1$) sharpens the distribution — the most likely token becomes even more dominant, producing more predictable, repetitive text. Temperature → 0 is equivalent to greedy decoding: always pick the argmax token.
</details>

## Common mistakes

- Not calling `model.detach_hidden()` between batches — gradients will try to flow through the entire training history, causing memory errors
- Reshaping logits incorrectly before `CrossEntropyLoss` — must be `(B*T, vocab)` not `(B, T, vocab)`. Use `.reshape(-1, vocab_size)`
- Using `model(x)` during generation without passing the current hidden state — each call would reset to h₀=0, making the model ignore all previous context

## Final takeaway

An LSTM language model trains to predict the next token at every position, learning the statistical structure of text via cross-entropy loss. The hidden state carries context across time steps; truncated BPTT enables efficient gradient computation. Perplexity measures how well the model predicts held-out text. Text generation uses temperature-controlled sampling from the predicted distribution. LSTM language models were the precursor to GPT — the objective is identical, only the architecture differs.
