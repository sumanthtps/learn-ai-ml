---
id: 07-building-ann
title: "Video 7: Building an ANN using PyTorch"
sidebar_label: "07 · Building an ANN"
sidebar_position: 7
description:
  Full end-to-end Artificial Neural Network — architecture design, training,
  evaluation, and common pitfalls.
tags: [pytorch, ann, neural-network, classification, regression, campusx]
---

# Building an ANN using PyTorch

**📺 CampusX — Practical Deep Learning using PyTorch | Video 7**

> **What you'll learn:** How to build, train, and evaluate a complete Artificial
> Neural Network (ANN) for both classification and regression tasks — the first
> time all previous concepts come together in one real project.

---

## 1. What is an ANN?

An **Artificial Neural Network (ANN)** — also called a **Multi-Layer Perceptron
(MLP)** or **Feedforward Neural Network** — is the simplest kind of deep neural
network. It has layers of neurons, where each neuron in one layer is connected
to every neuron in the next layer:

```
Input Layer  →  Hidden Layers  →  Output Layer

(features)   →  (learned repr)  →  (prediction)
   [20]           [128][64]           [5]
```

Each layer computes: `output = activation(W × input + b)`

The "deep" in deep learning refers to having multiple hidden layers — each layer
transforms the representation from the previous layer into something more
abstract and useful for the final prediction.

## Visual Reference

![Single-hidden-layer ANN: input, hidden, output layers](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/images/nn1.png)

Read the image as information flowing left to right: input features are mixed by
weighted connections, transformed in the hidden layer with a non-linear
activation, and finally mapped to output neurons. The number on each edge is a
learned weight; training adjusts all weights to minimize loss. Without the
activation function, stacking multiple linear layers would be equivalent to a
single linear layer — the network could only learn linear boundaries.

### This lesson in the learning arc

This is the first time all the previous lessons come together in one end-to-end
project. The training pipeline from Video 4, the `nn.Module` from Video 5, and
the DataLoader from Video 6 all combine here:

- **Videos 1-3** taught you the building blocks: tensors, autograd, and
  computation graphs
- **Video 4** showed the training loop structure
- **Videos 5-6** taught you how to build models and load data
- **This video** combines all of it into a working neural network for a real
  dataset

After this, the next three videos improve this same ANN: Video 8 moves it to
GPU, Video 9 optimizes it with regularization, and Video 10 tunes it with
Optuna.

### Choosing the right output + loss combination

The output layer activation and the loss function must be chosen together based
on the task:

| Task                  | Output activation | Loss function         | Why                                           |
| --------------------- | ----------------- | --------------------- | --------------------------------------------- |
| Binary classification | None (raw logits) | `BCEWithLogitsLoss`   | Fused sigmoid + BCE for stability             |
| Multi-class           | None (raw logits) | `CrossEntropyLoss`    | Fused softmax + NLL for stability             |
| Regression            | None              | `MSELoss` or `L1Loss` | Predict unbounded continuous values           |
| Multi-label           | None (raw logits) | `BCEWithLogitsLoss`   | Each output is an independent binary decision |

The pattern is: **never apply sigmoid or softmax before a fused loss function**
— the fused versions handle this internally with better numerical stability.

---

## 2. ANN for Classification (Tabular Data)

We'll use the **Breast Cancer Wisconsin** dataset — 30 numerical features,
binary output (malignant vs benign). This is a clean, well-understood dataset
that makes it easy to see how the model performs:

### What is the model trying to do here?

This is supervised classification on tabular data.

That means:

- input: one row of numerical features
- output: which class that row belongs to

So the ANN in this lesson is not doing image recognition or sequence modeling. It is learning from a fixed-size vector of numbers.

### Why do tabular models still need preprocessing?

Because neural networks are sensitive to feature scale.

If one feature is very small and another is very large, the large-scale feature can dominate the gradients. That makes learning uneven and harder to optimize.

That is why scaling appears before the model code. It is not optional ceremony; it is part of making training behave well.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# ── 1. Data Preparation ──────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target   # X: (569, 30) features, y: (569,) binary labels

# CRITICAL: Always scale features before training a neural network!
# Neural network weights are initialized in a small range (~[-0.1, 0.1])
# If input features have vastly different scales (e.g., one feature in [0, 0.1]
# and another in [0, 10000]), gradients for the large-scale feature dominate
# and the small-scale feature learns very slowly — or not at all.
scaler = StandardScaler()   # Normalize each feature to mean=0, std=1
X = scaler.fit_transform(X) # Fit on all data (we'll do proper split next)

# Convert NumPy arrays to PyTorch tensors with the right dtypes
X_tensor = torch.tensor(X, dtype=torch.float32)        # Features → float32
y_tensor = torch.tensor(y, dtype=torch.long)           # Labels → long (for CrossEntropyLoss)

# Wrap in a Dataset and split 80/20 train/validation
dataset = TensorDataset(X_tensor, y_tensor)
n_train = int(0.8 * len(dataset))
train_ds, val_ds = random_split(
    dataset,
    [n_train, len(dataset) - n_train],
    generator=torch.Generator().manual_seed(42)   # Reproducible split
)

# DataLoaders: shuffle training, don't shuffle validation
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

# ── 2. Model Architecture ────────────────────────────────────
#
# Beginner reading guide:
# - Linear layers learn feature combinations
# - BatchNorm stabilizes hidden activations
# - ReLU adds non-linearity
# - Dropout reduces overfitting
class BreastCancerNet(nn.Module):
    def __init__(self, input_dim=30, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            # Layer 1: reduce from 30 features to 64 hidden neurons
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),     # Normalize activations: stabilizes training
            nn.ReLU(),              # Non-linearity: allows learning non-linear boundaries
            nn.Dropout(0.3),        # Regularization: randomly zero 30% of neurons

            # Layer 2: further compression from 64 to 32 neurons
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),        # Less dropout deeper in the network

            # Output layer: 2 logits (one per class)
            # NO softmax/sigmoid here — CrossEntropyLoss applies LogSoftmax internally
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)   # Shape: (batch, num_classes)

# ── 3. Setup ─────────────────────────────────────────────────
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = BreastCancerNet().to(device)         # Move model to GPU if available
criterion = nn.CrossEntropyLoss()                # Multi-class classification loss
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# ReduceLROnPlateau: reduce LR when validation loss stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

# ── 4. Training Function ──────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()   # Enable Dropout + BatchNorm training mode
    total_loss, correct = 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)   # Move batch to GPU
        optimizer.zero_grad()                # Clear previous gradients
        logits = model(X)                    # Forward: (batch, 2) raw scores
        loss   = criterion(logits, y)        # Compare scores to labels
        loss.backward()                      # Compute gradients
        optimizer.step()                     # Update weights

        total_loss += loss.item()
        # argmax(1): for each sample, pick the class with the highest score
        correct    += (logits.argmax(1) == y).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

# ── 5. Validation Function ─────────────────────────────────
def eval_epoch(model, loader, criterion, device):
    model.eval()   # Disable Dropout, freeze BatchNorm stats
    total_loss, correct = 0, 0

    with torch.no_grad():   # No gradient computation — saves memory
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item()
            correct    += (logits.argmax(1) == y).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

# ── 6. Training Run ──────────────────────────────────────────
best_val_acc = 0
for epoch in range(1, 101):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss,   val_acc   = eval_epoch(model, val_loader,   criterion, device)

    # Pass validation loss to scheduler — it will reduce LR if val_loss stagnates
    scheduler.step(val_loss)

    # Save the best model checkpoint (use validation accuracy, not training accuracy)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

    if epoch % 20 == 0:
        print(f"E{epoch:3d} | Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f}")

print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

# ── 7. Final Evaluation ──────────────────────────────────────
# Load the best checkpoint — not the last epoch's weights!
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in val_loader:
        logits = model(X.to(device))
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.numpy())

print(classification_report(all_labels, all_preds, target_names=data.target_names))
```

### The evaluation pattern — reading logits → class predictions

The most important evaluation pattern to understand is converting raw logits to
class predictions:

```python
model.eval()
correct = 0
total   = 0

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        logits = model(X_batch.to(device))
        # logits shape: (batch_size, num_classes)
        # Each row = one sample, each column = score for one class
        # argmax(dim=1): for each row, find the column index with the highest score
        # → that index IS the predicted class
        _, predicted = torch.max(logits, dim=1)
        # Alternative: logits.argmax(dim=1) — same result

        total   += y_batch.shape[0]
        correct += (predicted.cpu() == y_batch).sum().item()

accuracy = correct / total
print("accuracy:", accuracy)
```

Why `torch.max(..., dim=1)` works here:

- Each row of logits corresponds to one sample
- Each column corresponds to one class score (higher = more confident)
- The class with the highest score is the prediction
- `argmax(dim=1)` returns the column index of the maximum for each row

---

## 3. ANN for Multi-Class Classification

The architecture changes minimally for multi-class tasks — the main difference
is the number of output neurons:

### What changes from binary to multi-class classification?

In binary classification, the model chooses between two classes.

In multi-class classification:

- each sample belongs to exactly one class out of several
- the model needs one output score per class
- the predicted class is the output with the highest score

So the network structure stays very similar. The key difference is the final layer size.

For the Iris dataset:

- input features: 4
- classes: 3 iris species
- output layer: 3 logits

That is why the last linear layer below has `3` outputs.

```python
from sklearn.datasets import load_iris

data    = load_iris()
X, y    = data.data, data.target   # (150, 4) features, 3 classes

# Always scale features for neural networks
scaler  = StandardScaler()
X       = scaler.fit_transform(X)

# Convert to tensors with correct dtypes
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)   # long required for CrossEntropyLoss

class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),    # 4 input features → 32 hidden neurons
            nn.ReLU(),
            nn.Linear(32, 16),   # 32 → 16 hidden neurons
            nn.ReLU(),
            nn.Linear(16, 3)     # 16 → 3 output logits (one per iris species)
            # No activation! CrossEntropyLoss applies softmax internally.
        )
    def forward(self, x): return self.net(x)

# Everything else stays the same: CrossEntropyLoss, AdamW, same training loop
```

---

## 4. ANN for Regression

For regression, only two things change: the output layer has a single neuron
(predicting a continuous value) and the loss function is MSELoss:

### What makes regression different from classification?

Classification predicts a category.

Regression predicts a number.

Examples:

- classification: spam vs not spam
- regression: house price, temperature, sales, age

That difference changes the last part of the network:

- classification output = class scores
- regression output = one continuous value

So for regression:

- the output layer usually has one neuron per target value
- there is no final softmax or sigmoid for ordinary scalar regression
- the loss measures numeric distance, not class mismatch

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler as SS

housing = fetch_california_housing()
X, y    = housing.data, housing.target   # (20640, 8) features, continuous target

# Scale features — same as classification
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# Scale the target too! Neural networks are sensitive to output scale.
# If y is in [0, 500000], the initial loss will be enormous and gradients unstable.
# Scaling y to roughly [-1, 1] or [0, 1] makes training smoother.
scaler_y = SS()
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)   # Shape: (N, 1) for MSELoss

class HousingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128),    # 8 input features → 128 hidden
            nn.ReLU(),
            nn.Linear(128, 64),   # 128 → 64
            nn.ReLU(),
            nn.Linear(64, 32),    # 64 → 32
            nn.ReLU(),
            nn.Linear(32, 1)      # 32 → 1 (single regression output)
            # NO activation on last layer! We want any real number, not (0,1) or probabilities.
        )
    def forward(self, x): return self.net(x)

criterion = nn.MSELoss()   # Mean Squared Error: (predicted - true)²

# After prediction, inverse-transform to get predictions in original scale:
# pred_original = scaler_y.inverse_transform(pred.cpu().detach().numpy())
```

**Key differences from classification:**

- Output layer has exactly `1` neuron (predicts one continuous value)
- No activation on the final layer (predictions can be any real number)
- Loss is `MSELoss` (measures squared distance from true value)
- Target `y` should be scaled for stable training

---

## 5. Common ANN Design Decisions

### How many layers and neurons?

There is no formula for the right architecture — it depends on your data. These
rules of thumb are a starting point:

| Complexity | Architecture                       | Typical dataset size |
| ---------- | ---------------------------------- | -------------------- |
| Simple     | 1–2 hidden layers, 32–128 neurons  | < 10K samples        |
| Medium     | 2–4 hidden layers, 64–512 neurons  | 10K–100K             |
| Deep       | 4+ hidden layers, 128–1024 neurons | > 100K               |

**General guidelines:**

- **Start small, increase if underfitting.** Adding capacity is cheap;
  diagnosing overfitting is harder.
- **Funnel shape**: each layer ≈ 50–75% the width of the previous one (e.g., 128
  → 64 → 32)
- **BatchNorm after linear, before activation**: stabilizes inputs to the
  activation
- **Dropout after activation**: regularizes the activated representation
- **No activation on the last layer**: the loss function handles the final
  transformation

### Activation function choice

```text
Hidden layers:  ReLU (default) → Leaky ReLU (if dead neurons) → GELU (Transformers)
Output layer:
  Regression:          None
  Binary classification:   None + BCEWithLogitsLoss
  Multi-class classification: None + CrossEntropyLoss
  Multi-label classification: None + BCEWithLogitsLoss
```

Dead neurons occur when ReLU always receives negative input and outputs 0 — the
gradient is 0, so the neuron never learns. Using Leaky ReLU (small non-zero
slope for negative inputs) prevents this.

---

## 6. Interview Questions

<details>
<summary><strong>Q1: Why must you scale features before training a neural network?</strong></summary>

Neural networks are sensitive to feature scale because: (1) **Initialization
mismatch** — weights are initialized in a small range (e.g., Kaiming:
~0.01–0.1). If one feature has values in [0, 100,000], the gradient for that
feature's weight is enormous relative to features in [0, 1], causing the
large-scale feature's weight to update much faster and dominate. (2)
**Activation saturation** — large input values push sigmoid/tanh into saturation
(zero gradient region), blocking learning. (3) **Loss surface shape** — unscaled
features create an elongated loss surface where gradient descent oscillates.
Standardization (zero mean, unit variance) fixes all three.

</details>

<details>
<summary><strong>Q2: Why use BatchNorm before activation, not after?</strong></summary>

The original BatchNorm paper (Ioffe & Szegedy, 2015) recommended placing it
before the activation. The intuition: BatchNorm normalizes pre-activations to
mean≈0, std≈1, putting them in the non-saturating region of sigmoid/tanh where
gradients are largest. If placed after ReLU, half the values are already zeroed
— the normalization statistics are less meaningful. In practice, both orderings
work; before activation is the standard default. For modern architectures like
Transformers (using LayerNorm), the norm is often placed at the beginning of
each block (pre-norm), which provides more training stability.

</details>

<details>
<summary><strong>Q3: What is the dead ReLU problem and how do you fix it?</strong></summary>

A "dead" ReLU neuron always receives negative pre-activation values, so it
always outputs 0 and has a gradient of 0 — it never updates. Causes: too-high
learning rate causing weights to go very negative, or poor initialization.
Fixes: (1) **Leaky ReLU**: `max(0.01x, x)` — non-zero gradient for negative
inputs prevents permanent death; (2) **ELU**: smooth negative region with
non-zero gradient; (3) **Lower learning rate**: prevents catastrophic weight
updates; (4) **Kaiming initialization**: specifically calibrated for ReLU,
reduces the probability of dead neurons from the start.

</details>

<details>
<summary><strong>Q4: What is the role of Dropout in an ANN?</strong></summary>

Dropout randomly zeroes a fraction `p` of neurons during each training forward
pass — different neurons are zeroed each time. This forces each neuron to learn
useful representations independently rather than co-adapting (relying on
specific other neurons always being present). The effect is like training an
ensemble of exponentially many different sub-networks simultaneously. During
inference (`model.eval()`), all neurons are active but their outputs are
implicitly scaled (PyTorch uses inverted dropout, scaling by `1/(1-p)` during
training instead). Typical rates: 0.3–0.5 for fully connected layers.

</details>

---

## 🔗 References

- [CampusX Video 7](https://www.youtube.com/watch?v=6EJaHBJhwDs)
