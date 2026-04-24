---
id: 13-rnn-question-answering
title: "Video 13: RNN using PyTorch — Question Answering System"
sidebar_label: "13 · RNN — Q&A System"
sidebar_position: 13
description: Building a Question Answering System using Recurrent Neural Networks in PyTorch — from RNN fundamentals to a working NLP project.
tags: [pytorch, rnn, nlp, question-answering, sequence, embedding, campusx]
---

# RNN using PyTorch — Question Answering System
**📺 CampusX — Practical Deep Learning using PyTorch | Video 13**

> **What you'll learn:** Why RNNs exist, how they work step by step, and how to build a complete text classification / question answering system using RNNs in PyTorch.

---

## 1. Why Do We Need RNNs?

ANNs and CNNs have a critical limitation: **they treat each input independently**. For text, this fails completely:

```
"The cat sat on the mat"
"Mat sat on the cat the"  ← same words, different meaning, ANN would give same output!
```

Language is **sequential** — meaning depends on order. Also:
- Sequences have **variable lengths** — sentences aren't all 10 words
- Each position depends on what came **before**

**RNNs solve this** by maintaining a **hidden state** that carries information from previous timesteps.

## Visual Reference

![RNN unrolled through time — shared weights, evolving hidden state](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

The important visual idea is that one RNN cell (left, looped) is unrolled into a chain (right). The same weights `W_hh` and `W_xh` process every token, but the hidden state `h_t` accumulates context from all previous tokens. The final `h_T` is a fixed-size summary of the entire sequence — used for classification or fed into a decoder for generation.

---

## 2. How RNNs Work

At each timestep `t`, the RNN receives:
- `x_t` — current input (word embedding)
- `h_{t-1}` — previous hidden state (the "memory")

And produces:
- `h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b)` — new hidden state
- `y_t` — optional output at this step

```
x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → [RNN] → h₄
       ↑              ↑              ↑              ↑
      h₀=0          h₁             h₂             h₃
```

The same weights `W_hh, W_xh` are shared across ALL timesteps. The hidden state `h_T` after the last word contains information about the **entire sentence**.

---

## 3. Text Preprocessing Pipeline

Before feeding text to an RNN, you need to convert it into numbers. Neural networks work on tensors of floats — not strings. This conversion pipeline has three steps: tokenize → build vocabulary → encode.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import collections, re

# ── Step 1: Tokenize text ────────────────────────────────────
# Tokenization = split text into individual tokens (words or subwords).
# Here we use a simple word-level tokenizer: lowercase, remove punctuation, split on spaces.
def tokenize(text):
    text = text.lower()                      # "What" and "what" should be the same token
    text = re.sub(r'[^\w\s]', '', text)      # Remove punctuation: "India?" → "india"
    return text.split()                       # Split on whitespace: ["what", "is", "india"]

sample = "What is the capital of India?"
print(tokenize(sample))   # ['what', 'is', 'the', 'capital', 'of', 'india']

# ── Step 2: Build vocabulary ─────────────────────────────────
# A vocabulary is a bidirectional mapping: word ↔ integer index.
# Why integers? nn.Embedding expects integer indices to look up vectors.
class Vocabulary:
    def __init__(self, min_freq=1):
        # Reserve index 0 for <PAD> and index 1 for <UNK> — these special tokens
        # must always have consistent indices across all vocabularies.
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}   # word → index lookup
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}    # index → word lookup (for decoding)
        self.min_freq  = min_freq   # Ignore rare words to reduce vocabulary size

    def build(self, texts):
        # Count how often each word appears across all texts
        counter = collections.Counter()
        for text in texts:
            counter.update(tokenize(text))  # Add all tokens from this text to the count

        # Add words that appear at least min_freq times to the vocabulary
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)     # Assign the next available index
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        return self   # Return self to enable method chaining: Vocabulary().build(texts)

    def encode(self, text, max_len=None):
        tokens = tokenize(text)
        # Convert each token to its index; unknown words get index 1 (<UNK>)
        # dict.get(key, default) returns default if key is not in dict
        indices = [self.word2idx.get(t, 1) for t in tokens]

        if max_len:
            if len(indices) < max_len:
                # Sequences shorter than max_len get padded with 0 (<PAD>)
                # This makes all sequences in a batch the same length for batching
                indices += [0] * (max_len - len(indices))
            else:
                # Sequences longer than max_len are truncated (keep first max_len tokens)
                indices = indices[:max_len]
        return indices

    def __len__(self):
        return len(self.word2idx)   # Total vocabulary size including special tokens

# Usage
texts = ["What is the capital of India?",
         "Who wrote Hamlet?",
         "What is photosynthesis?"]

vocab = Vocabulary(min_freq=1).build(texts)
print(f"Vocabulary size: {len(vocab)}")   # 2 special + all unique words

encoded = vocab.encode("What is India?", max_len=10)
print(encoded)
# 'what' → index 2, 'is' → index 3, 'india' → index 8, then 7 PADs
# Output: [2, 3, 8, 0, 0, 0, 0, 0, 0, 0]  (padded to length 10)
```

### Why all three components are necessary

- **Tokenization**: standardizes text (lowercase, no punctuation) so variations of the same word map to the same token
- **Vocabulary**: the bridge between string world and number world; without it, the RNN has nothing to process
- **Padding**: batching requires all sequences to be the same length. `<PAD>=0` is excluded from loss computation (via `ignore_index=0` in CrossEntropyLoss) so padding doesn't influence learning
- **`<UNK>`**: at inference time, the model may see words not in the training vocabulary. Instead of crashing, map them to `<UNK>` — "I've seen something like this before but not this exact word"

---

## 4. Word Embeddings

Before the code, we need one key idea:

### Why can't we feed word indices directly into an RNN?

After tokenization and vocabulary building, each word becomes an integer like:

- `"what"` -> `2`
- `"is"` -> `3`
- `"india"` -> `8`

But those numbers are just IDs. They do **not** mean:

- `8` is "bigger" than `3`
- `"india"` is mathematically close to `"is"`

If we feed raw token IDs into a neural network, the model may treat those arbitrary numbers as if they have numeric meaning.

### What is an embedding?

An embedding layer solves that problem by turning each token ID into a learned dense vector.

So instead of:

```python
2
```

the model sees something more like:

```python
[0.12, -0.44, 0.91, ...]
```

This vector is learned during training, and words used in similar contexts often end up with somewhat similar vectors.

### Mental model

- vocabulary maps word -> integer ID
- embedding maps integer ID -> dense vector
- RNN reads those vectors, not the raw IDs

That is why `nn.Embedding` is usually one of the first layers in NLP models.

```python
# nn.Embedding: a lookup table (vocab_size × embed_dim)
# Each word index → dense vector representation

embedding = nn.Embedding(
    num_embeddings=len(vocab),   # Vocabulary size
    embedding_dim=128,            # Embedding dimension
    padding_idx=0                 # <PAD> token → zero vector (not updated)
)

# Input: (batch, seq_len) — token indices
x = torch.tensor([[2, 3, 8, 0, 0]], dtype=torch.long)  # Padded sequence
out = embedding(x)
print(out.shape)   # (1, 5, 128)  — each token → 128-dim vector

# Using pretrained GloVe embeddings
import numpy as np

def load_glove(glove_path, vocab, embed_dim=100):
    """Load GloVe vectors for vocab words."""
    glove = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            glove[values[0]] = np.array(values[1:], dtype=np.float32)

    weights = np.random.randn(len(vocab), embed_dim) * 0.01   # Random init
    weights[0] = 0   # <PAD> = zero vector
    for word, idx in vocab.word2idx.items():
        if word in glove:
            weights[idx] = glove[word]

    embedding = nn.Embedding(len(vocab), embed_dim, padding_idx=0)
    embedding.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
    return embedding
```

---

## 5. Question Answering as Text Classification

We'll frame Q&A as **question classification** — predict the category of a question to determine how to answer it.

### Why frame Q&A as classification?

Real-world question answering can mean many different things. A full open-domain QA system is quite advanced.

For learning RNN fundamentals, classification is a much simpler starting point:

- input: a question
- output: one label or one answer class

This lets us focus on sequence modeling without needing retrieval, large knowledge bases, or generative decoding.

So in this lesson, "question answering" really means:

- read the question text
- convert it to a vector representation with the RNN
- predict one class from a fixed set of outputs

Categories: `FACTOID` (Who/What/When/Where), `YES_NO`, `HOW_MANY`, `WHY`, `HOW`

```python
# Sample dataset
QA_DATA = [
    ("What is the capital of France?",        "FACTOID"),
    ("Who invented the telephone?",            "FACTOID"),
    ("When did World War 2 end?",             "FACTOID"),
    ("Is Python a programming language?",      "YES_NO"),
    ("Can birds fly?",                         "YES_NO"),
    ("How many planets are in solar system?",  "HOW_MANY"),
    ("Why is the sky blue?",                   "WHY"),
    ("How does a computer work?",              "HOW"),
]

LABEL2IDX = {"FACTOID": 0, "YES_NO": 1, "HOW_MANY": 2, "WHY": 3, "HOW": 4}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

class QADataset(Dataset):
    def __init__(self, data, vocab, max_len=20):
        self.data    = data
        self.vocab   = vocab
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        question, label = self.data[idx]
        x = torch.tensor(self.vocab.encode(question, self.max_len), dtype=torch.long)
        y = torch.tensor(LABEL2IDX[label], dtype=torch.long)
        return x, y
```

### CampusX project framing: predict an answer word, then use a confidence threshold

The transcript's concrete project is slightly different from generic question
type classification. The model is trained on a small set of question-answer
pairs and predicts an answer from a fixed output vocabulary. At inference time,
if confidence is too low, the system returns **"I don't know"** instead of
forcing a bad answer.

That teaching setup is useful because it introduces a real deployment idea:
**abstain when confidence is low**.

```python
QA_PAIRS = [
    ("What is the capital of France?", "paris"),
    ("Who developed the theory of relativity?", "einstein"),
    ("Largest planet in our solar system?", "jupiter"),
    ("Who wrote Hamlet?", "shakespeare"),
]

answer_vocab = sorted({answer for _, answer in QA_PAIRS})
answer2idx = {answer: i for i, answer in enumerate(answer_vocab)}
idx2answer = {i: answer for answer, i in answer2idx.items()}
```

### Transcript-style pipeline for the whole project

The project flow in the transcript is:

1. load the CSV of question-answer pairs
2. tokenize text
3. build a vocabulary with an unknown token
4. convert words to indices
5. embed those indices into vectors
6. run the RNN
7. map the final output to an answer-word distribution
8. apply a confidence threshold before returning an answer

That step-by-step view is important because the project is really teaching the
full NLP preprocessing-to-inference path, not just the RNN layer itself.

---

## 5.5 Why Vanilla RNN Has a Memory Problem (BPTT)

Before building the model, let's understand why we need LSTMs specifically for this task — and why vanilla RNNs often fail on longer sequences.

When training an RNN, gradients flow backward through time via **Backpropagation Through Time (BPTT)**. At each timestep, the gradient is multiplied by the RNN's weight matrix. If this matrix has values slightly less than 1 (very common), multiplying it repeatedly over 20+ timesteps makes the gradient exponentially small by the time it reaches early timesteps:

```
Gradient at step 20:  1.0
Gradient at step 15:  0.9^5  ≈ 0.59
Gradient at step 10:  0.9^10 ≈ 0.35
Gradient at step 5:   0.9^15 ≈ 0.21
Gradient at step 1:   0.9^20 ≈ 0.12

...and with tanh activations (derivative ≤ 0.25):
Gradient at step 1:   0.25^20 ≈ tiny. The early layers learn nothing.
```

This means: **vanilla RNNs can't remember context from more than ~10–15 words ago**. A question like "What was the name of the composer who wrote Beethoven's 9th?" would lose the word "composer" by the time the model reaches "9th".

**LSTMs solve this** with a **cell state** `c_t` that uses addition (not multiplication) to accumulate information across timesteps. Gradients through an additive path don't vanish — the gradient of a sum is 1 for each term, not a decaying product. This is why we build a Bidirectional LSTM below, not a vanilla RNN.

## 6. Building the RNN Model

Before reading the class, it helps to separate the model into stages.

### What is this model doing, step by step?

1. take token IDs
2. convert them to embedding vectors
3. run those vectors through an LSTM or GRU
4. compress the sequence into one final representation
5. map that representation to class logits

So even though the code looks longer than an ANN, it is still a pipeline.

### Why is there a "final hidden state"?

For classification, we want one prediction for the whole question, not one prediction per word.

That means the model needs one vector that summarizes the sequence. In recurrent models, that summary is usually taken from the final hidden state of the last recurrent layer.

In the bidirectional case, we get two summaries:

- one from reading left to right
- one from reading right to left

Then we concatenate them.

```python
class QuestionClassifierRNN(nn.Module):
    """
    Architecture:
    Input:  (batch, seq_len)          — token indices
    Embed:  (batch, seq_len, embed_dim)
    RNN:    (batch, seq_len, hidden_dim)   — hidden at each step
    Last:   (batch, hidden_dim)            — final hidden state
    FC:     (batch, num_classes)
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 n_layers=2, dropout=0.3, bidirectional=True, rnn_type='LSTM'):
        super().__init__()
        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        # RNN layer (choose LSTM or GRU)
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,         # input: (batch, seq, features)
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output dim
        rnn_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_out_dim, rnn_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(rnn_out_dim // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len) — integer token indices
        # Step 1: Look up embedding vectors for each token
        embedded = self.embed_dropout(self.embedding(x))   # (batch, seq_len, embed_dim)

        # Step 2: Run LSTM over the sequence
        if isinstance(self.rnn, nn.LSTM):
            # LSTM returns TWO things:
            # - output: (batch, seq_len, hidden*dirs) — hidden state at EVERY timestep
            # - (h_n, c_n): tuple of final hidden state and final cell state
            # h_n shape: (n_layers * num_directions, batch, hidden_dim)
            output, (h_n, c_n) = self.rnn(embedded)
        else:  # GRU returns only output and h_n (no cell state)
            output, h_n = self.rnn(embedded)

        # Step 3: Extract the final hidden state to represent the whole sequence
        if self.bidirectional:
            # With bidirectional LSTM, h_n has shape (n_layers*2, batch, hidden_dim).
            # The last two entries are the final hidden states of the last layer:
            # - h_n[-2]: forward direction's final hidden (has seen tokens 1...T)
            # - h_n[-1]: backward direction's final hidden (has seen tokens T...1)
            # We concatenate them: the forward pass knows "what came before",
            # the backward pass knows "what comes after". Together they give full context.
            h_forward  = h_n[-2, :, :]   # (batch, hidden_dim) — forward final
            h_backward = h_n[-1, :, :]   # (batch, hidden_dim) — backward final
            final_hidden = torch.cat([h_forward, h_backward], dim=-1)
            # final_hidden: (batch, hidden_dim*2) — full sequence representation
        else:
            # Unidirectional: just use the last layer's final hidden state
            final_hidden = h_n[-1]   # (batch, hidden_dim)

        # Step 4: Map the sequence representation to class logits
        return self.classifier(final_hidden)


# Build vocabulary from all questions
all_questions = [q for q, _ in QA_DATA]
vocab = Vocabulary().build(all_questions)

# Model
model = QuestionClassifierRNN(
    vocab_size=len(vocab),
    embed_dim=64,
    hidden_dim=128,
    num_classes=len(LABEL2IDX),
    n_layers=2,
    dropout=0.3,
    bidirectional=True,
    rnn_type='LSTM'
)

print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Model walkthrough

- `nn.Embedding` turns token IDs into dense vectors the RNN can learn from
- the recurrent layer reads the sequence one time step at a time
- in the bidirectional case, forward and backward summaries are concatenated before classification
- the final linear layers convert the sequence representation into label logits

### Why `nn.Sequential` is not always the right tool here

One transcript detail that matters a lot in practice: recurrent layers often do
not fit neatly inside `nn.Sequential`.

Why?

- embedding layer returns one tensor
- RNN/LSTM often returns multiple outputs
- later code may need only one of those outputs, or may need to reshape it

So for sequence models, a manual `forward()` is often cleaner than trying to
force everything into one `Sequential` container.

### Two implementation details CampusX debugs explicitly

The transcript calls out two easy-to-miss details for PyTorch RNN code:

```python
self.rnn = nn.RNN(
    input_size=embed_dim,
    hidden_size=hidden_dim,
    batch_first=True
)
```

- `batch_first=True` makes the input shape `(batch, seq_len, features)`, which
  is usually easier to reason about in the rest of the code.

And sometimes after taking the final hidden output, you may need to remove an
extra size-1 dimension before the final linear layer:

```python
final_hidden = final_hidden.squeeze(0)
logits = self.fc(final_hidden)
```

The general lesson is bigger than this single project: sequence models often
need more explicit shape handling than feedforward models.

---

## 7. Training the Q&A Classifier

This part is mostly a standard supervised training loop, but applied to text batches instead of tabular or image batches.

### What is being trained here?

- input: tokenized question
- target: question class label
- prediction: logits over possible labels

So the core pattern is still the familiar one:

1. forward pass
2. compute loss
3. backward pass
4. optimizer step

The main difference is just that the model handling the input is recurrent instead of feedforward.

```python
from torch.utils.data import DataLoader, random_split

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = model.to(device)

dataset = QADataset(QA_DATA, vocab, max_len=20)
# For demonstration — in practice you'd have 1000s of examples
train_size = int(0.8 * len(dataset))
train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(100):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            correct = sum((model(x.to(device)).argmax(1) == y.to(device)).sum().item()
                         for x, y in val_loader)
        print(f"Epoch {epoch:3d} | Val Acc: {correct / len(val_ds):.4f}")

# ── Inference ────────────────────────────────────────────────
def predict_question_type(question, model, vocab, device):
    model.eval()
    x = torch.tensor([vocab.encode(question, max_len=20)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=-1)[0]
        pred   = logits.argmax(1).item()
    label = IDX2LABEL[pred]
    confidence = probs[pred].item()
    print(f"Question: {question}")
    print(f"Type: {label} (confidence: {confidence:.2%})")
    return label

predict_question_type("What is quantum physics?", model, vocab, device)
predict_question_type("Is the earth flat?", model, vocab, device)
```

### Transcript-style inference with fallback

```python
def predict_answer_with_threshold(question, model, vocab, device, threshold=0.5):
    model.eval()
    x = torch.tensor([vocab.encode(question, max_len=20)], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

    if confidence < threshold:
        return "I don't know", confidence

    return idx2answer[pred_idx], confidence
```

### Why the threshold matters

- every classifier will produce *some* class, even for nonsense or unseen input
- the threshold stops us from pretending low-confidence guesses are reliable
- this is exactly the kind of simple decision rule that makes toy NLP demos feel more realistic

### Important realism note

This lesson is useful for learning sequence modeling fundamentals, but modern production Q&A systems usually use transformers rather than vanilla RNN or LSTM classifiers. The RNN viewpoint still matters because it teaches how hidden-state based sequence models work internally.

---

## 8. Variable-Length Sequences (Packed Sequences)

This section builds directly on the `collate_fn` idea from the `Dataset` and `DataLoader` lesson, so let's make the workflow explicit before jumping into the optimization.

### What problem are we solving?

RNNs and LSTMs process sequences token by token. Real text data is messy because different sentences have different lengths:

- `"hi"` has 2 tokens
- `"how are you"` has 3 tokens
- `"what time is the meeting tomorrow"` has many more

But tensors inside a batch must have a rectangular shape. So we usually:

1. take several sequences of different lengths
2. pad the shorter ones with a special padding token such as `0`
3. stack them into one batch tensor

That is exactly why we need a custom `collate_fn`.

### Why is plain padding not always ideal?

Padding makes batching possible, but it also creates fake tokens. If one sentence has length 30 and another has length 4, the shorter sentence may get lots of trailing padding. A plain RNN/LSTM will still do computation on those padding positions unless we tell it not to.

So there are really two steps here:

- `collate_fn` pads sequences so batching works
- `pack_padded_sequence` tells the LSTM to skip useless work on padding tokens

### Why do we sort by length?

Packed sequences store sequence data in a compact form. For the classic packed-sequence workflow, PyTorch expects examples to be ordered from longest to shortest before packing. That is why the collate function sorts the batch first.

So this custom collate function is doing three jobs:

1. separate texts and labels
2. compute the true lengths
3. sort and pad the texts so they are ready for packing

```python
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def padded_collate(batch):
    """Custom collate to handle variable-length sequences efficiently."""
    texts, labels = zip(*batch)

    # Sort by length (required for pack_padded_sequence)
    lengths = torch.tensor([len(t) for t in texts])
    sorted_lengths, sort_idx = lengths.sort(descending=True)
    texts_sorted  = [texts[i] for i in sort_idx]
    labels_sorted = torch.stack([labels[i] for i in sort_idx])

    # Pad to max length in this batch
    padded = pad_sequence(texts_sorted, batch_first=True, padding_value=0)
    return padded, sorted_lengths, labels_sorted

class EfficientRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)   # (batch, max_len, embed)

        # Pack: skip padding tokens during LSTM computation
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True)
        output_packed, (h_n, c_n) = self.lstm(packed)
        # Unpack if you need all timestep outputs:
        # output, _ = pad_packed_sequence(output_packed, batch_first=True)

        # h_n[-1] is the final hidden state for each sequence (at its true end, not pad)
        return self.fc(h_n[-1])
```

### Mental model for the full flow

- dataset returns one tokenized sentence and one label
- `padded_collate` turns many sentences into one padded batch and also returns their real lengths
- `pack_padded_sequence` removes the wasted computation on padding positions
- the LSTM now processes only real tokens

This is why the code looks more complicated than the earlier RNN examples: we are not changing the model's basic idea, we are teaching it how to work efficiently with variable-length input.

---

## 9. Interview Questions

<details>
<summary><strong>Q1: What is an RNN and what problem does it solve?</strong></summary>

A Recurrent Neural Network processes sequential data by maintaining a hidden state that is updated at each timestep: `h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b)`. This hidden state acts as a "memory" that carries information from earlier in the sequence. It solves the problem of processing variable-length sequences where the order and context of elements matters — unlike ANNs that treat each input independently.
</details>

<details>
<summary><strong>Q2: What is the vanishing gradient problem in RNNs?</strong></summary>

During backpropagation through time (BPTT), gradients are multiplied by the weight matrix at each timestep. If the weight matrix has eigenvalues < 1, repeated multiplication makes gradients exponentially small for early timesteps — they effectively contribute nothing to learning. This means RNNs struggle to learn dependencies > 10–15 timesteps apart. Solution: LSTMs and GRUs with gating mechanisms that create protected "highways" for gradients.
</details>

<details>
<summary><strong>Q3: What is an embedding layer? Why use it instead of one-hot encoding?</strong></summary>

An embedding layer is a learnable lookup table that maps each token index to a dense vector. One-hot encoding maps 10,000-word vocabulary to a 10,000-dimensional sparse vector (one 1, rest 0). Embeddings map each word to a dense 100–300 dimensional vector that: (1) is dramatically smaller; (2) captures semantic similarity — "king" and "queen" have similar vectors; (3) is learned end-to-end with the task. Pretrained embeddings (GloVe, Word2Vec) can be used as initialization.
</details>

<details>
<summary><strong>Q4: What is the difference between output and h_n in PyTorch's LSTM?</strong></summary>

`output`: shape `(batch, seq_len, hidden_dim * num_directions)` — the hidden state `h_t` at every timestep (only from the last layer). `h_n`: shape `(num_layers * num_directions, batch, hidden_dim)` — the final hidden state `h_T` at the last timestep for every layer. For classification, use `h_n[-1]` (final layer's final hidden) — it summarizes the whole sequence. For sequence-to-sequence tasks, use `output` (predictions at each position).
</details>

---

## 🔗 References
- [PyTorch RNN Docs](https://pytorch.org/docs/stable/nn.html#recurrent-layers)
- [Understanding LSTMs (Colah)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [CampusX Video 13](https://www.youtube.com/watch?v=xjzWrPQ66VQ)
