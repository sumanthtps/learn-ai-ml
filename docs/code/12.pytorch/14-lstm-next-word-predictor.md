---
id: 14-lstm-next-word-predictor
title: "Video 14: LSTM using PyTorch — Next Word Predictor"
sidebar_label: "14 · LSTM — Next Word Predictor"
sidebar_position: 14
description: Building a Next Word Prediction system using LSTM — language modeling, text generation, temperature sampling, and beam search.
tags: [pytorch, lstm, nlp, language-model, text-generation, next-word, campusx]
---

# LSTM using PyTorch — Next Word Predictor
**📺 CampusX — Practical Deep Learning using PyTorch | Video 14**

> **What you'll learn:** How to build a complete Next Word Predictor (language model) using LSTM — from data preparation to text generation with temperature and beam search.

---

## 1. What is a Language Model?

A **Language Model** (LM) learns the probability distribution over sequences of words:

```
P("the cat sat on the ___") → {mat: 0.42, floor: 0.18, table: 0.12, ...}
```

This is the foundation of:
- Next word prediction (keyboard autocomplete)
- Text generation (GPT, LLaMA)
- Speech recognition (most likely word sequence)
- Machine translation

**Next Word Prediction as classification:** given the previous N words, predict the next word from the vocabulary.

---

## 2. LSTM Deep Dive

### Why LSTM over vanilla RNN?

Vanilla RNN suffers from **vanishing gradients** — it forgets context beyond ~15 words. LSTM solves this with:

1. A **Cell State** `c_t` — a "conveyor belt" that carries long-term information with minimal modification
2. **Gates** that control what to remember, forget, and output

### LSTM Gates

```
┌─────────────────────────────────────────────────────────────┐
│                         LSTM CELL                           │
│                                                             │
│  Input: x_t (current word), h_{t-1} (prev hidden), c_{t-1} │
│                                                             │
│  Forget Gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)        │
│  → What fraction of c_{t-1} to erase (0=forget, 1=keep)    │
│                                                             │
│  Input Gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)        │
│  → What to write to cell state                              │
│                                                             │
│  Cell Update:  g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)     │
│  → Candidate values to add to cell state                    │
│                                                             │
│  Cell State:   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t            │
│  → Updated long-term memory (minimal gradient vanishing!)   │
│                                                             │
│  Output Gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)        │
│  → What part of cell state to expose as hidden state        │
│                                                             │
│  Hidden State: h_t = o_t ⊙ tanh(c_t)                       │
│  → Short-term memory / output                               │
└─────────────────────────────────────────────────────────────┘
```

## Visual Reference

![LSTM chain — cell state conveyor belt with forget, input, output gates](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

Use the image to map the formulas to the data flow: the horizontal line across the top is the **cell state** `c_t` — the long-term memory that flows through with minimal modification. The forget gate (σ) erases irrelevant past context, the input gate (σ × tanh) writes new information, and the output gate (σ) controls what fraction of the cell state becomes the hidden state `h_t`. The additive update to `c_t` is the key reason gradients don't vanish over hundreds of timesteps.

**Key insight:** The cell state `c_t` passes through the network with only element-wise multiplication and addition — gradients can flow directly without vanishing over hundreds of timesteps.

```python
import torch
import torch.nn as nn

# PyTorch LSTM layer
lstm = nn.LSTM(
    input_size=128,     # Embedding dimension
    hidden_size=256,    # Hidden state size
    num_layers=2,       # Stack 2 LSTMs
    batch_first=True,   # (batch, seq, features)
    dropout=0.3,        # Applied between layers
    bidirectional=False # One direction for language modeling
)

x = torch.rand(32, 50, 128)       # (batch=32, seq=50, embed=128)
h0 = torch.zeros(2, 32, 256)     # (num_layers, batch, hidden)
c0 = torch.zeros(2, 32, 256)

output, (h_n, c_n) = lstm(x, (h0, c0))
print(output.shape)  # (32, 50, 256)  — hidden state at each timestep
print(h_n.shape)     # (2, 32, 256)   — final hidden per layer
print(c_n.shape)     # (2, 32, 256)   — final cell per layer
```

---

## 3. Data Preparation for Language Modeling

Before the code, here is the basic task:

### What is a language model trying to learn?

A language model learns this pattern:

> "Given the previous words, what word is likely to come next?"

So the model is not learning facts directly. It is learning next-token prediction from text sequences.

That means our raw text has to be converted into a form the model can train on:

1. clean the text
2. split it into tokens
3. build a vocabulary
4. convert each token to an integer ID

The code below is doing exactly that preparation pipeline.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re, collections

# ── Sample corpus (replace with any text file) ──────────────
CORPUS = """
Deep learning is a subset of machine learning that uses neural networks.
Neural networks are inspired by the human brain and consist of layers.
Each layer transforms the input data into a more abstract representation.
The power of deep learning comes from its ability to learn hierarchical features.
Recurrent neural networks process sequential data like text and time series.
Long short-term memory networks solve the vanishing gradient problem in RNNs.
Language models predict the next word given the previous context.
PyTorch makes it easy to build and train deep learning models.
Transfer learning allows us to reuse pretrained models for new tasks.
Attention mechanisms allow models to focus on relevant parts of the input.
"""

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

tokens = preprocess(CORPUS)
print(f"Total tokens: {len(tokens)}")

# ── Build Vocabulary ─────────────────────────────────────────
counter = collections.Counter(tokens)
# Sort by frequency (most common first)
vocab_words = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + \
              [w for w, _ in counter.most_common()]
word2idx = {w: i for i, w in enumerate(vocab_words)}
idx2word = {i: w for w, i in word2idx.items()}
VOCAB_SIZE = len(word2idx)
print(f"Vocabulary size: {VOCAB_SIZE}")

# Encode the entire corpus
encoded = [word2idx.get(t, word2idx['<UNK>']) for t in tokens]
```

### Code walkthrough

- preprocessing normalizes text so `"PyTorch"` and `"pytorch"` are treated as the same token
- `word2idx` and `idx2word` create the bidirectional mapping needed for both training and generation
- the whole corpus becomes a sequence of integers before it ever reaches the LSTM

---

## 4. Creating Training Sequences

For next word prediction, we create **sliding window** sequences:

### Why do we need training sequences?

After encoding the corpus, we have one long list of token IDs.

Example:

```python
[deep, learning, is, a, subset, ...]
```

But the model cannot train on an entire book or corpus as one giant sequence in a beginner example. So we break the stream into many smaller training examples.

### What is a sliding window?

A sliding window means:

- take a short chunk of consecutive tokens
- use it as input
- shift by one position and create the target

So the window "slides" one step at a time across the corpus.

This gives us many supervised training pairs from one text stream.

### Why is the target shifted by one?

Because the task is next-word prediction.

If the input is:

```python
[the, cat, sat, on]
```

the model should learn to predict:

```python
[cat, sat, on, the]
```

Each position is trained to predict the next token after it.

```python
class LanguageModelDataset(Dataset):
    """
    Creates (input_sequence, target_word) pairs using a sliding window.

    Example with seq_len=4:
    tokens: [the, cat, sat, on, the, mat]
    pairs:
      ([the, cat, sat, on], the)   ← predict "the" from 4-word context
      Wait — actually we predict NEXT word:
      ([the, cat, sat, on], mat)   ← no, let's use:
      input:  [the, cat, sat, on]
      target: [cat, sat, on, the]  ← shift by 1 (teacher forcing)
    """
    def __init__(self, encoded_tokens, seq_len):
        self.seq_len = seq_len
        self.data    = encoded_tokens

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # Input: words at positions [idx, idx+seq_len)
        # Target: words at positions [idx+1, idx+seq_len+1) (shifted by 1)
        x = torch.tensor(self.data[idx     : idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + self.seq_len + 1], dtype=torch.long)
        return x, y

SEQ_LEN = 10  # Use 10 previous words to predict next word
dataset = LanguageModelDataset(encoded, SEQ_LEN)
print(f"Training samples: {len(dataset)}")

# Show a sample
x, y = dataset[0]
print("Input:", [idx2word[i.item()] for i in x])
print("Target:", [idx2word[i.item()] for i in y])
# Input:  [deep, learning, is, a, subset, of, machine, learning, that, uses]
# Target: [learning, is, a, subset, of, machine, learning, that, uses, neural]

train_size = int(0.9 * len(dataset))
train_ds, val_ds = torch.utils.data.random_split(
    dataset, [train_size, len(dataset) - train_size]
)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
```

---

## 5. LSTM Language Model

Before the code, let’s pin down the architecture idea.

### Why does the model output a prediction at every time step?

In next-word prediction, every position in the input sequence is useful.

Example:

- after `"deep"` predict `"learning"`
- after `"deep learning"` predict `"is"`
- after `"deep learning is"` predict `"a"`

So the LSTM does not just produce one final answer for the whole sequence. It produces a hidden state at each time step, and each hidden state is turned into a vocabulary prediction.

That is why the output shape is:

```python
(batch, seq_len, vocab_size)
```

not just:

```python
(batch, vocab_size)
```

### What are the main parts of the model?

- `Embedding`: converts token IDs into dense vectors
- `LSTM`: reads the sequence one token at a time and maintains memory
- `Dropout`: regularizes the model
- `Linear`: maps each hidden state to vocabulary logits

With that in mind, the code below is easier to read as a pipeline rather than a bag of layers.

```python
class NextWordLSTM(nn.Module):
    """
    LSTM Language Model for next-word prediction.

    Forward pass:
    Input:  (batch, seq_len)              — token indices
    Embed:  (batch, seq_len, embed_dim)
    LSTM:   (batch, seq_len, hidden_dim)  — output at EACH step
    Linear: (batch, seq_len, vocab_size)  — logits for next word at each position
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        # Layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm       = nn.LSTM(
            embed_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, vocab_size)

        # Tie embedding and output weights (reduces parameters, improves perplexity)
        # This works when embed_dim == hidden_dim
        if embed_dim == hidden_dim:
            self.fc.weight = self.embedding.weight

    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))   # (batch, seq, embed)
        output, hidden = self.lstm(embedded, hidden)  # output: (batch, seq, hidden)
        output = self.dropout(output)
        logits = self.fc(output)                      # (batch, seq, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden and cell state to zeros."""
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return h0, c0

    def detach_hidden(self, hidden):
        """Detach hidden state to prevent backprop through entire history."""
        h, c = hidden
        return h.detach(), c.detach()


device = "cuda" if torch.cuda.is_available() else "cpu"
model  = NextWordLSTM(
    vocab_size=VOCAB_SIZE,
    embed_dim=128,
    hidden_dim=128,
    n_layers=2,
    dropout=0.3
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Model walkthrough

- the embedding layer learns a dense vector for every vocabulary item
- the LSTM produces a hidden state for every time step, not just the last one
- `self.fc(output)` converts each hidden state into logits over the full vocabulary
- tying output weights to embedding weights reduces parameters and often improves language modeling quality

---

## 6. Training the Language Model

Training a language model is slightly different from classification. In classification, every batch is independent — you can shuffle them freely. In language modeling, the text is one continuous stream. We use **teacher forcing**: at each timestep, we feed the true previous word as input (not the word the model predicted). This makes training stable, but it means during generation we use the model's own predictions — a known gap called "exposure bias".

### What is teacher forcing?

This term often appears before it is explained clearly.

Teacher forcing means:

- during training, the model is given the true previous token as input
- it is not forced to rely on its own imperfect prediction from the previous step

That makes optimization much easier, especially early in training.

### Why is the loss reshaped with `view(-1, VOCAB_SIZE)`?

This is another place that often feels abrupt.

The model outputs:

```python
(batch, seq_len, vocab_size)
```

But `CrossEntropyLoss` expects:

```python
(N, C)
```

where:

- `N` = number of predictions
- `C` = number of classes

So we flatten all time steps from all sequences into one big list of predictions. That is what the `view` call is doing: it is not changing the meaning of the data, only reshaping it into the format the loss function expects.

```python
import math

# ignore_index=0: don't compute loss on <PAD> tokens.
# Padding positions should not penalize the model — they're not real text.
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# CosineAnnealingLR: smoothly decays LR from 1e-3 → near zero over 30 epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    hidden = None   # Will be initialized on first batch

    for X, Y in loader:
        # X: (batch, seq_len) — input tokens (the context)
        # Y: (batch, seq_len) — target tokens (what should come next at each position)
        # Y is X shifted by 1 position: Y[i] = X[i+1] for all i
        X, Y = X.to(device), Y.to(device)

        # ── Hidden state management ───────────────────────────
        if hidden is None:
            # First batch: initialize h and c to zeros (no prior context)
            hidden = model.init_hidden(X.size(0), device)
        else:
            # For subsequent batches, we REUSE the hidden state from the previous batch.
            # This means the LSTM carries memory from one batch to the next —
            # the model reads the corpus as one continuous stream, not independent chunks.
            #
            # BUT: we DETACH the hidden state from the computation graph.
            # Why? Without detach(), loss.backward() would try to backpropagate
            # all the way through every previous batch — potentially thousands of steps.
            # That's both computationally infeasible and causes vanishing gradients.
            # detach() says: "keep the hidden STATE values (for forward pass context),
            # but cut the GRADIENT path (don't backprop before this batch)."
            hidden = model.detach_hidden(hidden)

            # Handle the last batch which may be smaller than the rest
            # (e.g., 1000 samples ÷ 32 batch size = 31 full batches + 1 batch of 8)
            # The hidden state has shape (n_layers, batch_size, hidden_dim).
            # If batch size changes, the old hidden can't be used — reset it.
            if hidden[0].size(1) != X.size(0):
                hidden = model.init_hidden(X.size(0), device)

        optimizer.zero_grad()

        # Forward pass: returns predictions at every timestep + updated hidden state
        logits, hidden = model(X, hidden)   # logits: (batch, seq_len, vocab_size)

        # CrossEntropyLoss expects:
        #   input:  (N, C) where N = total predictions, C = num_classes (vocab_size)
        #   target: (N,)
        # But logits is (batch, seq_len, vocab_size) and Y is (batch, seq_len).
        # view(-1, VOCAB_SIZE) flattens (batch × seq_len) into one long list of predictions.
        # Y.view(-1) flattens the targets the same way.
        loss = criterion(
            logits.view(-1, VOCAB_SIZE),    # (batch*seq_len, vocab_size)
            Y.view(-1)                       # (batch*seq_len,)
        )

        loss.backward()
        # Gradient clipping is especially important for LMs: sequences are long,
        # and LSTMs can produce large gradients when predicting rare words.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    # Perplexity = exp(cross-entropy loss). The intuition: if PPL=10, the model
    # is as uncertain as if it had to pick uniformly from 10 equally likely words.
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            hidden = model.init_hidden(X.size(0), device)
            logits, _ = model(X, hidden)
            loss = criterion(logits.view(-1, VOCAB_SIZE), Y.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss, math.exp(avg_loss)

best_val_ppl = float('inf')
for epoch in range(1, 31):
    train_loss, train_ppl = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss,   val_ppl   = eval_epoch(model, val_loader, criterion, device)
    scheduler.step()

    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        torch.save(model.state_dict(), "best_lm.pth")

    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d} | Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")

print(f"\nBest Perplexity: {best_val_ppl:.2f}")
```

---

## 7. Text Generation — Greedy, Temperature, and Beam Search

Once the language model is trained, we want to generate text from it. At each step, the model produces a probability distribution over the entire vocabulary ("the next word could be 'the' with 30%, 'a' with 15%, 'neural' with 5%, ..."). The question is: how do you pick a word from that distribution?

There are three strategies, each with different tradeoffs:

### Why do we need decoding strategies at all?

Because the model does not directly output a word. It outputs **scores or probabilities** over the whole vocabulary.

So generation always has a second step:

1. model predicts a distribution over possible next words
2. decoding rule chooses one word from that distribution

Different decoding rules create different text behavior:

- safer and more repetitive
- more diverse but riskier
- more globally optimized but slower

That is why decoding is an important concept separate from the model itself.

```python
# ── Method 1: Greedy Decoding ────────────────────────────────
# Strategy: always pick the single most probable word.
# Advantage: fast, deterministic, reproducible.
# Problem: produces repetitive, generic text because the most probable word
# is usually a common filler ("the", "a", "is") that leads to bland sequences.

def generate_greedy(model, seed_text, num_words, vocab, device):
    model.eval()
    # Convert seed text into token indices
    tokens = [word2idx.get(w, word2idx['<UNK>']) for w in preprocess(seed_text)]
    tokens = tokens[-SEQ_LEN:]   # Keep only the last SEQ_LEN tokens (context window)

    generated = list(tokens)
    hidden = None

    with torch.no_grad():
        for _ in range(num_words):
            # Feed last SEQ_LEN tokens as context
            x = torch.tensor([tokens[-SEQ_LEN:]], dtype=torch.long, device=device)
            if hidden is None:
                hidden = model.init_hidden(1, device)   # batch_size=1 for inference

            logits, hidden = model(x, hidden)  # logits: (1, seq_len, vocab_size)
            # We want predictions for the LAST position in the sequence
            # (what comes after all the context tokens)
            next_token = logits[0, -1].argmax().item()  # index of most probable word
            tokens.append(next_token)

    return ' '.join(idx2word.get(t, '<UNK>') for t in tokens[len(generated)-len(preprocess(seed_text)):])


# ── Method 2: Temperature Sampling ──────────────────────────
# Strategy: sample from the probability distribution, scaled by temperature T.
#
# Temperature is the key creative control:
# T = 0.0 → extremely focused: almost always picks the #1 word (greedy-like)
# T = 0.7 → conservative: favors high-probability words but allows variety
# T = 1.0 → raw model distribution: balanced
# T = 1.5 → creative: gives rare words more chance (unexpected, sometimes incoherent)
#
# Analogy: temperature in thermodynamics controls how "energetic" particles are.
# Low T = calm, predictable. High T = chaotic, random.
# For text: low T = factual, repetitive. High T = creative, surprising.

def generate_with_temperature(model, seed_text, num_words, vocab, device,
                               temperature=1.0, top_k=0, top_p=0.9):
    """
    temperature: controls randomness (0.7–1.2 typical range)
    top_k: only sample from the top k words (0 = disabled, use all)
    top_p: nucleus sampling — only sample from words that together account for
           top_p probability mass (e.g., 0.9 = use the words that cover 90% of prob)
    """
    model.eval()
    tokens = [word2idx.get(w, word2idx['<UNK>']) for w in preprocess(seed_text)]

    generated_words = []
    hidden = None

    with torch.no_grad():
        for _ in range(num_words):
            x = torch.tensor([tokens[-SEQ_LEN:]], dtype=torch.long, device=device)
            if hidden is None:
                hidden = model.init_hidden(1, device)

            logits, hidden = model(x, hidden)
            logits = logits[0, -1]   # Get last timestep: (vocab_size,)

            # ① Apply temperature: dividing logits by T before softmax
            # makes the distribution sharper (T<1) or flatter (T>1).
            # Lower T → high-prob words get even more probability mass.
            # Higher T → probability is spread more evenly across all words.
            logits = logits / temperature

            # ② Top-K filtering: keep only the top k words, set rest to -inf.
            # -inf → softmax gives these words probability 0.
            # This prevents the model from sampling very unlikely (nonsense) words.
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, top_k)
                min_top_k = top_k_values[-1]   # The k-th highest logit value
                logits[logits < min_top_k] = -float('inf')  # Mask everything below

            # ③ Top-P (Nucleus) sampling: keep words until their cumulative
            # probability reaches top_p. This is adaptive — sometimes 10 words
            # cover 90% of the probability, sometimes 50 words.
            # This handles cases where the model is very confident (use few words)
            # or uncertain (allow more words).
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                # Compute cumulative softmax probabilities (sorted by descending prob)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                # Find words that push cumulative prob above top_p — exclude them
                sorted_idx_to_remove = cumulative_probs > top_p
                # Shift right: keep the word that crossed top_p (otherwise we'd have <top_p)
                sorted_idx_to_remove[1:] = sorted_idx_to_remove[:-1].clone()
                sorted_idx_to_remove[0] = False   # Always keep the most probable word
                indices_to_remove = sorted_idx[sorted_idx_to_remove]
                logits[indices_to_remove] = -float('inf')

            # ④ Convert filtered logits to probabilities and sample
            probs = torch.softmax(logits, dim=-1)
            # torch.multinomial: samples one index from probs, weighted by probability
            # (higher probability = more likely to be sampled)
            next_token = torch.multinomial(probs, num_samples=1).item()

            tokens.append(next_token)
            generated_words.append(idx2word.get(next_token, '<UNK>'))

    return seed_text + ' ' + ' '.join(generated_words)


# ── Method 3: Beam Search ────────────────────────────────────
# Strategy: maintain the B most probable SEQUENCES at each step (not just words).
# Greedy picks the best word at step 1, then the best at step 2, etc. —
# it never revisits the step 1 decision. Beam search keeps multiple options alive,
# allowing the overall best sequence to win even if it doesn't start with the
# single most probable word.
#
# Example with beam_width=3, step 1:
#   Greedy picks: "the" (probability 0.30)
#   Beam keeps:   "the" (0.30), "a" (0.15), "neural" (0.05)
# At step 2, all 3 beams are expanded. Maybe "neural network" ends up with a
# higher total probability than "the quick" — beam search finds this, greedy doesn't.

def beam_search(model, seed_text, num_words, vocab, device, beam_width=3):
    model.eval()
    tokens = [word2idx.get(w, word2idx['<UNK>']) for w in preprocess(seed_text)]

    # Each beam is a tuple: (cumulative_log_prob, token_sequence, hidden_state)
    # We use LOG probabilities to avoid numerical underflow when multiplying many small probs
    with torch.no_grad():
        # Step 0: get the first predictions from the seed
        x = torch.tensor([tokens[-SEQ_LEN:]], dtype=torch.long, device=device)
        hidden = model.init_hidden(1, device)
        logits, hidden = model(x, hidden)
        # log_softmax: converts logits to log probabilities (avoids underflow)
        log_probs = torch.log_softmax(logits[0, -1], dim=-1)

        # Initialize: keep the top beam_width starting tokens
        top_log_probs, top_tokens = log_probs.topk(beam_width)
        beams = [
            (top_log_probs[i].item(),          # Cumulative log probability
             tokens + [top_tokens[i].item()],   # Sequence so far
             hidden)                             # Hidden state (carries LSTM memory)
            for i in range(beam_width)
        ]

    # Step 1 to num_words: expand each beam, keep top beam_width sequences
    for step in range(1, num_words):
        all_candidates = []
        for log_prob, seq, hidden in beams:
            # Run one step for this beam's current sequence
            x = torch.tensor([seq[-SEQ_LEN:]], dtype=torch.long, device=device)
            # clone() because multiple beams share state — modifying in-place would corrupt others
            logits, new_hidden = model(x, (hidden[0].clone(), hidden[1].clone()))
            step_log_probs = torch.log_softmax(logits[0, -1], dim=-1)

            # Expand: try the top beam_width next words for this beam
            top_log_probs, top_tokens = step_log_probs.topk(beam_width)
            for i in range(beam_width):
                # Sum log probs = multiply raw probs (log rule: log(a×b) = log(a)+log(b))
                candidate_log_prob = log_prob + top_log_probs[i].item()
                candidate_seq = seq + [top_tokens[i].item()]
                all_candidates.append((candidate_log_prob, candidate_seq, new_hidden))

        # From all beam_width × beam_width candidates, keep the best beam_width
        # Sort by cumulative log probability (higher = better)
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:beam_width]

    # The beam with the highest cumulative probability is our answer
    best_seq = beams[0][1]
    return ' '.join(idx2word.get(t, '<UNK>') for t in best_seq)
```

---

## 8. Inference — Interactive Next Word Prediction

Training teaches the model from data. Inference is the separate step where we actually use the trained model to generate predictions.

### What does inference mean here?

Here inference means:

- load the trained weights
- give the model a seed phrase
- ask it what words are likely to come next

So this section is not teaching a new model. It is showing how to use the one we already trained.

```python
# Load best model
model.load_state_dict(torch.load("best_lm.pth", map_location=device))
model.eval()

def predict_next_words(seed, n=5, temperature=0.8, top_k=10):
    print(f"\nSeed: '{seed}'")
    print(f"Greedy:      '{generate_greedy(model, seed, n, word2idx, device)}'")
    print(f"Temperature: '{generate_with_temperature(model, seed, n, word2idx, device, temperature, top_k=top_k)}'")
    print(f"Beam(3):     '{beam_search(model, seed, n, word2idx, device, beam_width=3)}'")

predict_next_words("deep learning is")
predict_next_words("recurrent neural networks")
predict_next_words("the power of")
```

---

## 9. Evaluating a Language Model — Perplexity

### Why do language models need a different metric?

For classification, accuracy is often enough.

For language modeling, every step predicts a probability distribution over a vocabulary. We need a metric that measures how good those probabilities are, not just whether one final label was correct.

Perplexity is the standard answer to that problem.

Beginner intuition:

- low perplexity = the model is less surprised by the true text
- high perplexity = the model is unsure and spreads probability too widely

```python
# Perplexity: how "surprised" the model is by the test text
# Lower = better (less surprised = more confident predictions)
# PPL = exp(average cross-entropy loss)

# PPL = 1   → Perfect model (knows exactly what comes next)
# PPL = 10  → On average, model is unsure between 10 equally likely words
# PPL = 100 → Very uncertain
# GPT-2 achieves PPL ~35 on Penn Treebank
# GPT-3 achieves PPL ~20 on Penn Treebank

def compute_perplexity(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            hidden = model.init_hidden(X.size(0), device)
            logits, _ = model(X, hidden)

            # Ignore padding tokens in perplexity
            mask = Y != 0
            loss = criterion(logits.view(-1, VOCAB_SIZE), Y.view(-1))
            total_loss   += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

    avg_loss   = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

test_ppl = compute_perplexity(model, val_loader, criterion, device)
print(f"Test Perplexity: {test_ppl:.2f}")
```

### Why perplexity matters

Perplexity is the standard language-model metric because it answers a practical question: "How uncertain is the model about the next token?" Lower perplexity means the model assigns higher probability to the true continuation.

---

## 10. GRU vs LSTM for Language Modeling

### Why compare GRU and LSTM?

Because they solve a similar sequence-learning problem with slightly different internal designs.

For a beginner, the practical takeaway is:

- LSTM has a bit more memory machinery
- GRU is simpler and often faster
- both are much better than vanilla RNN for longer sequences

So this comparison is less about memorizing gate formulas and more about understanding the tradeoff between model complexity and efficiency.

```python
# Drop-in replacement: just change nn.LSTM → nn.GRU
class NextWordGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru        = nn.GRU(          # ← Only change from LSTM
            embed_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded, hidden)   # GRU returns (output, h_n) only!
        logits = self.fc(self.dropout(output))         # No c_n for GRU
        return logits, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        # Note: GRU only needs h_0, not (h_0, c_0) like LSTM!
```

| | LSTM | GRU |
|---|---|---|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| States | Cell + Hidden | Hidden only |
| Params (same hidden) | ~33% more | Fewer |
| Training speed | Slower | Faster |
| Long sequences | Slightly better | Comparable |
| Default choice | ✅ For NLP | ✅ For time series |

---

## 11. Interview Questions

<details>
<summary><strong>Q1: What is a language model and what is perplexity?</strong></summary>

A language model assigns probabilities to sequences of words: `P(w₁, w₂, ..., wₙ)`. For next word prediction, it models `P(wₜ | w₁, ..., wₜ₋₁)`. Perplexity measures how well the model predicts — it's `exp(average cross-entropy loss)`. Intuitively: PPL=10 means the model is, on average, as confused as if it had to choose uniformly among 10 equally likely next words. Lower PPL = better model. GPT-class models achieve PPL in the 20–50 range on standard benchmarks.
</details>

<details>
<summary><strong>Q2: What is teacher forcing in language model training?</strong></summary>

Teacher forcing feeds the **true previous token** as input at each training step, even if the model predicted something different. This makes training faster and more stable. Without teacher forcing (free-running): if the model predicts the wrong word at step t, the error compounds through the rest of the sequence. Downside: creates a train/inference mismatch ("exposure bias") — at inference, the model must use its own predictions, which may differ from training conditions.
</details>

<details>
<summary><strong>Q3: Why do we detach the hidden state between batches?</strong></summary>

LSTM hidden state `h_t` from the end of one batch is used as the initial state `h_0` for the next batch (stateful LM). If we don't call `.detach()`, the gradient flows back through the entire training history — potentially thousands of timesteps — which is: (1) computationally infeasible, (2) causes vanishing/exploding gradients. Detaching cuts the gradient graph between batches while still passing the hidden state values forward.
</details>

<details>
<summary><strong>Q4: What is temperature in text generation?</strong></summary>

Temperature scales the logits before softmax: `probs = softmax(logits / T)`. Low T (< 1): distribution becomes sharper — the most probable word dominates. High T (> 1): distribution becomes flatter — more words have similar probabilities → more random/creative. T=1 uses the raw model distribution. In practice: T=0.7 for factual, coherent text; T=1.0 for balanced; T=1.5 for creative/diverse; T→0 approaches greedy decoding.
</details>

<details>
<summary><strong>Q5: What is beam search and why is it better than greedy decoding?</strong></summary>

Greedy decoding always picks the single most probable next word at each step — a locally optimal choice that may not be globally optimal. Beam search maintains `B` (beam width) candidate sequences simultaneously. At each step, it expands all B candidates by their top B next tokens and keeps the B highest-probability sequences overall. This explores the space of sequences more thoroughly. Beam search is the standard for machine translation and summarization; greedy is faster; temperature sampling is used for diverse generation (LLMs).
</details>

<details>
<summary><strong>Q6: What is the difference between LSTM's h_n and c_n?</strong></summary>

`h_n` (hidden state): the "short-term memory" — a filtered view of the cell state through the output gate. It's the output used for predictions and passed to the next timestep as `h_{t-1}`. `c_n` (cell state): the "long-term memory" — stores accumulated information over many timesteps via the forget gate mechanism. It passes through time with only element-wise operations, enabling the gradient to flow without vanishing. Only LSTMs have `c_n`; GRUs merge both into a single `h_n`.
</details>

---

## 12. Complete Model Comparison

This final comparison is here to help place the sequence models in context instead of treating them as isolated APIs.

The main question is:

> "When would I use vanilla RNN, LSTM, GRU, or Transformer?"

The summary below is meant to answer that at a high level.

```python
# Summary of models for sequence tasks:

models_comparison = {
    "Vanilla RNN": {
        "pros":  "Simple, few parameters",
        "cons":  "Vanishing gradients, max ~15-word context",
        "use":   "Very short sequences, toy tasks"
    },
    "LSTM": {
        "pros":  "Long-range dependencies, stable gradients",
        "cons":  "More parameters, slower than GRU",
        "use":   "NLP, language modeling, machine translation"
    },
    "GRU": {
        "pros":  "Fewer params than LSTM, faster training",
        "cons":  "Slightly worse than LSTM on very long sequences",
        "use":   "Time series, when speed matters"
    },
    "Transformer": {
        "pros":  "Parallel processing, any-range attention",
        "cons":  "O(n²) memory, needs large data",
        "use":   "BERT, GPT, modern NLP — best in class"
    }
}

for name, info in models_comparison.items():
    print(f"\n{name}:")
    for k, v in info.items():
        print(f"  {k:6s}: {v}")
```

### Modern context

LSTMs are historically important and still useful for lightweight sequence tasks, but large-scale next-token prediction in industry is now dominated by transformer architectures. This lesson is still valuable because it builds the mental bridge from simple sequence models to modern LLM training.

### What you've now learned — the full 14-video arc

This is the last video in the series. Looking back at what the 14 lessons built up together:

| Videos | What you learned |
|---|---|
| 1–2 | PyTorch as a framework; tensors as the foundational data structure |
| 3–4 | Autograd and the training loop — how neural networks actually learn |
| 5–6 | `nn.Module` and DataLoader — the PyTorch abstractions for real projects |
| 7 | Building a complete ANN — all concepts applied to a real classification task |
| 8 | GPU training — making training fast enough to matter |
| 9 | Optimization — diagnosing overfitting/underfitting, BatchNorm, Dropout, LR scheduling |
| 10 | Hyperparameter tuning — automating what used to be manual guesswork |
| 11 | CNNs — the right architecture for spatial/image data |
| 12 | Transfer learning — standing on shoulders of giants instead of training from scratch |
| 13–14 | RNNs and LSTMs — sequence models for text and time series data |

The natural next steps from here are:
- **Transformers and attention**: the architecture behind BERT, GPT, and all modern LLMs
- **Object detection**: CNNs for localizing objects in images (YOLO, Faster R-CNN)
- **Deployment**: TorchScript, ONNX, serving models in production
- **Distributed training**: multi-GPU and multi-machine training for large models

---

## 🔗 References
- [Understanding LSTMs (Colah's Blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of RNNs (Karpathy)](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [PyTorch Language Model Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [CampusX Video 14](https://www.youtube.com/watch?v=pnK1p0kz-3Y)
