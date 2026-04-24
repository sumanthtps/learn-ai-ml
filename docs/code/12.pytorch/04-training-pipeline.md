---
id: 04-training-pipeline
title: "Video 4: PyTorch Training Pipeline"
sidebar_label: "04 · Training Pipeline"
sidebar_position: 4
description: End-to-end training pipeline — forward pass, loss computation, backward pass, optimizer step, and evaluation.
tags: [pytorch, training, pipeline, loss, optimizer, campusx]
---

# PyTorch Training Pipeline
**📺 CampusX — Practical Deep Learning using PyTorch | Video 4**

> **What you'll learn:** How to connect tensors, autograd, models, loss functions, and optimizers into a complete working training pipeline — the 7-step loop that every PyTorch model uses.

---

## 1. The Big Picture: What Training Means

At this point you have seen tensors (data containers), autograd (gradient computation), and a brief glimpse of the training loop. Now it's time to understand how all of those pieces connect into a full training pipeline.

Training a neural network means repeatedly doing one thing: **making predictions, measuring how wrong they are, computing gradients, and adjusting weights to be less wrong**. Every training pipeline, regardless of complexity, follows the same fundamental cycle:

```
┌──────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
│                                                              │
│  Data           →  Model           →  Loss   →  Update      │
│  ──────            ──────────         ────       ──────      │
│  Dataset            nn.Module         MSE        SGD         │
│  DataLoader         Linear/Conv       CE         Adam        │
│  transforms         Activation        BCE        AdamW       │
│                     Dropout                                   │
└──────────────────────────────────────────────────────────────┘
```

The **7-step training loop** that you will write in every PyTorch project:

```python
for epoch in range(num_epochs):          # Outer loop: how many full passes over data
    for X_batch, y_batch in train_loader:  # Inner loop: one mini-batch at a time
        # 1. Move data to device (GPU if available)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # 2. Reset gradients from the previous step (they accumulate by default!)
        optimizer.zero_grad()
        # 3. Forward pass: run the model, get predictions
        predictions = model(X_batch)
        # 4. Compute loss: measure how wrong the predictions are
        loss = criterion(predictions, y_batch)
        # 5. Backward pass: compute gradients for every weight
        loss.backward()
        # 6. Update weights: optimizer adjusts weights using computed gradients
        optimizer.step()
        # 7. Log/track metrics (loss, accuracy, etc.)
```

### Understanding the roles: batch, iteration, epoch

These three terms are often confused. They refer to different levels of the nested loop:

```text
Sample     = one training example (one row in your dataset)
Batch      = a small group of samples processed together (e.g., 32 samples)
Iteration  = one optimizer update — processes one batch
Epoch      = one full pass over the ENTIRE dataset
```

Concrete example:
- dataset size = `10,000` samples
- batch size = `100`
- → iterations per epoch = `10,000 / 100 = 100`
- → training for `20` epochs = `2,000` total optimizer steps

---

## 2. From scikit-learn to PyTorch: Bridging the Gap

If you come from classical machine learning, PyTorch training feels familiar in spirit but much more explicit. Understanding what PyTorch does manually vs what scikit-learn hides is key to appreciating why PyTorch works the way it does.

### scikit-learn style: everything hidden inside `.fit()`

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Data starts as NumPy arrays / pandas tables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 3. Train — one line hides everything: forward pass, loss, gradients, updates
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predict / evaluate
preds = model.predict(X_test)
score = model.score(X_test, y_test)
```

### PyTorch style: explicit at every step

```python
# 1. Data often still starts as NumPy arrays
# 2. Preprocessing may still use scikit-learn (they work together well!)
# 3. Convert arrays to tensors
# 4. Build model, define loss and optimizer
# 5. Write the training loop manually (you control every step)
# 6. Run validation manually (you decide what to measure)
```

### Why PyTorch requires writing more code

In scikit-learn, `model.fit(...)` hides forward computation, loss computation, gradient computation, parameter updates, and the number of optimization steps. In PyTorch, **you write those pieces yourself**. That is extra work, but it gives you the flexibility needed for:
- Deep neural networks (any architecture you can imagine)
- Custom losses (beyond what any framework provides out of the box)
- GPU training (you choose when and how data moves to the device)
- Mixed precision (you decide where to use float16)
- Sequence models, CNNs, Transformers — none of which fit in `.fit()`

### Mental mapping from sklearn to PyTorch

| Classical ML / sklearn | PyTorch equivalent |
|---|---|
| NumPy arrays / DataFrame | Tensor |
| `train_test_split()` | manual split / `random_split()` |
| `StandardScaler()` | sklearn preprocessing before tensor conversion |
| `model.fit(X, y)` | explicit epoch + batch loop |
| `model.predict(X)` | `model.eval()` + forward pass |
| `model.score(X, y)` | manual metric computation |

:::tip
For tabular deep learning, it is completely normal to use **scikit-learn for preprocessing** and **PyTorch for modeling** in the same project. They complement each other.
:::

---

## 3. NumPy / scikit-learn Preprocessing Before PyTorch

This is the most common bridge pattern in practice: preprocess with sklearn, model with PyTorch.

```python
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Step 1: Create or load data as NumPy ──────────────────────
X = np.random.randn(1000, 20)            # 1000 samples, 20 features
y = np.random.randint(0, 2, size=1000)   # binary labels

# ── Step 2: Split using sklearn ───────────────────────────────
# Always split BEFORE any preprocessing to prevent data leakage!
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # stratify: preserve class ratio
)

# ── Step 3: Scale using sklearn ───────────────────────────────
# fit_transform on train: compute mean/std from training data only
# transform on val: apply training statistics (never fit on val!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit to train, then transform
X_val   = scaler.transform(X_val)         # only transform (use train's mean/std)

# ── Step 4: Convert to PyTorch tensors ────────────────────────
# Feature inputs: float32 (standard for neural network math)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
# Classification labels: long (int64) — required by CrossEntropyLoss
y_train = torch.tensor(y_train, dtype=torch.long)
y_val   = torch.tensor(y_val,   dtype=torch.long)
```

---

## 4. Building the Pipeline from Scratch — One Neuron

Before using the full `nn.Module` + `torch.optim` stack, it helps to build the training pipeline **manually from scratch**. This single-neuron example makes every step explicit, so there's no magic:

```python
import torch

torch.manual_seed(42)   # Reproducible results

# ── Toy dataset ───────────────────────────────────────────────
X_train = torch.randn(8, 3)               # 8 samples, 3 features
y_train = torch.tensor([1., 0., 1., 1., 0., 0., 1., 0.])   # binary labels
X_test  = torch.randn(4, 3)
y_test  = torch.tensor([1., 0., 1., 0.])

# ── Learnable parameters ──────────────────────────────────────
# These are the only things that change during training
# requires_grad=True: autograd will compute gradients for these
weights = torch.randn(3, requires_grad=True)   # one weight per feature
bias    = torch.zeros(1, requires_grad=True)   # single bias term
lr      = 0.1                                   # learning rate (step size)

# ── Model: one sigmoid neuron ─────────────────────────────────
def forward(X):
    logits = X @ weights + bias   # linear combination: dot(inputs, weights) + bias
    return torch.sigmoid(logits)  # sigmoid maps any real number to (0,1) probability

# ── Loss: binary cross-entropy ────────────────────────────────
def binary_cross_entropy(y_pred, y_true):
    eps = 1e-7   # Small constant to avoid log(0) = -inf
    y_pred = torch.clamp(y_pred, eps, 1 - eps)   # Clip predictions away from 0 and 1
    # BCE formula: -[y*log(ŷ) + (1-y)*log(1-ŷ)], averaged over samples
    loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return loss.mean()

# ── Training loop ─────────────────────────────────────────────
for epoch in range(25):
    # Step 1: Forward pass — generate predictions from current weights
    y_pred = forward(X_train)

    # Step 2: Compute loss — measure how wrong those predictions are
    loss = binary_cross_entropy(y_pred, y_train)

    # Step 3: Backward pass — compute gradients for weights and bias
    loss.backward()

    # Step 4: Update parameters — move weights in direction of decreasing loss
    # Must use no_grad() here! We're manually updating weights,
    # and we don't want this update itself to be tracked by autograd
    with torch.no_grad():
        weights -= lr * weights.grad   # gradient descent step
        bias    -= lr * bias.grad

    # Step 5: Zero gradients — must do this EVERY epoch
    # If we skip this, next epoch's backward() will ADD to existing gradients
    weights.grad.zero_()
    bias.grad.zero_()

    print(f"epoch={epoch+1:02d} loss={loss.item():.4f}")

# ── Evaluation ────────────────────────────────────────────────
# No learning happens during evaluation — just prediction
with torch.no_grad():
    test_probs = forward(X_test)
    test_preds = (test_probs > 0.5).float()        # threshold: prob > 0.5 → class 1
    accuracy   = (test_preds == y_test).float().mean()

print("test probabilities:", test_probs)
print("test predictions:  ", test_preds)
print("test accuracy:     ", accuracy.item())
```

### What this manual example teaches

Reading this code in order reveals the full training logic, step by step:

1. `weights` and `bias` are created first — these are the **only things training changes**
2. `forward(X)` defines prediction: dot product → sigmoid
3. `binary_cross_entropy(...)` converts predictions into one error scalar
4. The `for epoch in range(25):` loop repeats the learning process 25 times
5. `y_pred = forward(X_train)` runs current weights on training data
6. `loss = ...` measures current error
7. `loss.backward()` computes `dLoss/dweights` and `dLoss/dbias`
8. Inside `torch.no_grad()`, we manually step the weights
9. `weights.grad.zero_()` and `bias.grad.zero_()` clear accumulated gradients
10. After training, `with torch.no_grad():` at the end evaluates without learning

This is exactly the bridge into the next lesson: `nn.Module` and `torch.optim` automate the manual weight creation, sigmoid, loss, update, and zero-grad steps.

---

## 5. Building Blocks of the Pipeline

### 5.1 Loss Functions

A loss function converts model predictions + true labels into a single scalar that measures how wrong the model is. The training goal is to minimize this scalar.

```python
import torch
import torch.nn as nn

# ── REGRESSION losses ──────────────────────────────────────────
# MSELoss: mean squared error — penalizes large errors heavily (quadratic)
criterion = nn.MSELoss()
preds  = torch.tensor([2.5, 0.5, 2.0, 8.0])   # model predictions (continuous)
target = torch.tensor([3.0, 0.0, 2.0, 7.0])   # true values
print(criterion(preds, target))   # mean((preds - target)²)

# L1Loss: mean absolute error — more robust to outliers (linear penalty)
# Large errors are penalized less severely than with MSE
criterion_mae = nn.L1Loss()

# ── BINARY CLASSIFICATION ──────────────────────────────────────
# BCEWithLogitsLoss: the correct choice for binary classification
# Input: raw logits (BEFORE sigmoid), Target: float values in {0.0, 1.0}
# Why not BCELoss? BCEWithLogitsLoss numerically stable (avoids sigmoid → overflow)
criterion_bce = nn.BCEWithLogitsLoss()
logits  = torch.tensor([2.0, -1.0, 0.5])   # raw model outputs (no sigmoid applied)
targets = torch.tensor([1.0,  0.0, 1.0])   # binary labels as FLOAT (0.0 or 1.0)
print(criterion_bce(logits, targets))

# ── MULTI-CLASS CLASSIFICATION ─────────────────────────────────
# CrossEntropyLoss: the correct choice for multi-class classification
# Input: raw logits shape (N, C), Target: class indices shape (N,) as LONG
# CrossEntropyLoss = LogSoftmax + NLLLoss, fused for numerical stability
criterion_ce = nn.CrossEntropyLoss()
logits  = torch.rand(4, 5)                              # 4 samples, 5 classes
targets = torch.tensor([0, 2, 4, 1], dtype=torch.long) # class indices (MUST be long)
print(criterion_ce(logits, targets))

# ── Why NOT use Softmax + NLLLoss manually? ───────────────────
# Softmax(x) can overflow for large x values
# CrossEntropyLoss uses the log-sum-exp trick internally: numerically stable
```

### Loss-function quick reference

| Task | Model output shape | Target shape & type | Loss to use |
|---|---|---|---|
| Regression | `(N, 1)` or `(N,)` | `(N, 1)` or `(N,)` float | `MSELoss` |
| Binary classification | `(N,)` or `(N, 1)` — raw logits | `(N,)` or `(N, 1)` float | `BCEWithLogitsLoss` |
| Multi-class classification | `(N, C)` — raw logits | `(N,)` long (int64) | `CrossEntropyLoss` |

Many beginner training bugs are mismatched output shapes or wrong target dtypes. When in doubt, check the table above.

### 5.2 Optimizers

An optimizer reads the gradients stored in `.grad` after `loss.backward()` and updates the model's parameters. Different optimizers have different update rules that affect convergence speed and quality:

Before looking at the code, keep this mental model:

- `loss.backward()` computes gradients
- the optimizer uses those gradients to change the weights
- the learning rate controls how big each change is

So if gradients answer:

> "Which direction should each weight move?"

the optimizer answers:

> "How exactly should we move it?"

### Why do we need different optimizers?

In theory, plain gradient descent is enough. In practice, training can be slow, noisy, or unstable.

Different optimizers try to improve one or more of these:

- speed: reach a good solution faster
- stability: avoid wild oscillation
- adaptivity: use different step sizes for different parameters
- regularization: reduce overfitting while training

You do not need to memorize all update equations at the start. For a beginner, the most important distinction is:

- `SGD`: simple, classic, very interpretable
- `Adam` / `AdamW`: more automatic, usually easier defaults

Then the code below becomes a menu of choices rather than a wall of unfamiliar names.

```python
import torch.optim as optim

model = nn.Linear(10, 1)  # simple model for demonstration

# ── SGD: Stochastic Gradient Descent — the classic ────────────
# Pure update rule: w = w - lr * grad
optimizer = optim.SGD(model.parameters(), lr=0.01)

# SGD with momentum: accumulates velocity to accelerate in consistent directions
# Like a ball rolling downhill — gathers speed in the direction of the gradient
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# SGD with weight decay: adds L2 regularization term to penalize large weights
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# ── Adam: adaptive learning rates per parameter ───────────────
# Maintains per-parameter learning rates based on gradient history
# Good default for most tasks — handles sparse gradients well
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ── AdamW: Adam with DECOUPLED weight decay — RECOMMENDED ─────
# Regular Adam applies weight decay incorrectly (mixed into gradient)
# AdamW separates weight decay from gradient update — better regularization
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# ── RMSprop: adaptive LR, good for RNNs ───────────────────────
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
```

### Optimizer comparison

| Optimizer | Core idea | When to use |
|---|---|---|
| SGD | `w -= lr × grad` | When you need a well-understood, tunable baseline |
| SGD+momentum | `v = β*v + grad; w -= lr*v` | Training CNNs from scratch |
| Adam | Adaptive LR per param + momentum | Default for most tasks |
| AdamW | Adam + proper L2 regularization | **Best default choice for most projects** |
| RMSprop | Adaptive LR, no momentum | RNNs, non-stationary problems |

## Visual Reference

![Six optimizers compared on a loss surface contour](https://www.ruder.io/content/images/2016/09/contours_evaluation_optimizers.gif)

*Six optimizers navigating the same loss surface. Momentum-based methods cut across valleys faster; adaptive methods adjust step size per parameter. This is why AdamW — which combines adaptive rates with proper weight decay — is the standard default for most PyTorch training jobs.*

---

## 6. Learning Rate Schedulers

A fixed learning rate is often suboptimal. Starting high helps converge quickly; ending low helps settle into a good minimum without oscillating. Schedulers adjust the learning rate automatically during training:

### What is a scheduler?

A scheduler is just a small controller attached to the optimizer. The optimizer updates the weights, and the scheduler updates the optimizer's learning rate over time.

So:

- optimizer changes model weights
- scheduler changes the optimizer's `lr`

Why is that useful? Because the "best" learning rate is usually not constant for the whole training run.

Typical pattern:

- early training: larger LR helps the model make fast progress
- later training: smaller LR helps the model settle down and fine-tune

### Beginner intuition

Think of training like trying to reach the bottom of a valley in fog:

- if your steps are too small from the start, progress is painfully slow
- if your steps stay too large near the end, you keep bouncing around the bottom

Schedulers solve that by changing step size over time.

### Do beginners always need one?

No. A fixed learning rate is fine for simple experiments. Schedulers become more important when:

- training for many epochs
- fine-tuning pretrained models
- the loss stops improving and needs a smaller LR
- you want more stable final convergence

Now the code below is easier to read: each scheduler is just a different rule for changing `lr`.

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# ── StepLR: multiply LR by gamma every step_size epochs ───────
# Simple and predictable: halve the LR every 10 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# After 10 epochs: LR=5e-4. After 20: LR=2.5e-4. After 30: LR=1.25e-4.

# ── CosineAnnealingLR: smooth cosine decay ─────────────────────
# LR follows cosine curve from initial LR down to eta_min over T_max epochs
# Smooth transitions without sudden drops
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# ── ReduceLROnPlateau: reduce LR when metric stops improving ────
# Adaptive: only reduces LR when the model is stuck
# patience=5: wait 5 epochs with no improvement before reducing
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min',       # minimize val_loss
    patience=5,                  # wait this many epochs without improvement
    factor=0.5                   # multiply LR by this factor when triggered
)

# ── OneCycleLR: warmup then annealing, one cycle total ─────────
# Starts low, ramps up to max_lr, then decays — all in one training run
# Excellent for fine-tuning and rapid convergence
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-2,
    steps_per_epoch=len(train_loader),   # called every batch, not every epoch
    epochs=30
)

# ── ExponentialLR: multiply LR by gamma every epoch ───────────
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# ── How to use in training loop ────────────────────────────────
for epoch in range(epochs):
    train(...)
    scheduler.step()                  # Most schedulers: call once per epoch
# scheduler.step(val_loss)          # ReduceLROnPlateau: needs a metric
    print(f"LR: {scheduler.get_last_lr()}")
```

**When to call `scheduler.step()`:**
- `StepLR`, `CosineAnnealingLR`, `ExponentialLR` → once per **epoch**
- `OneCycleLR` → once per **batch** (designed around total training steps)
- `ReduceLROnPlateau` → after validation, because it needs to see the metric

---

## 7. Complete Training Pipeline from Scratch

Now let's build a full production-quality training pipeline. Every piece from the previous sections comes together here:

### How should a beginner read this large example?

Not as one giant block to memorize.

Read it as seven smaller stages:

1. create data
2. wrap data into datasets and loaders
3. define the model
4. choose loss/optimizer/scheduler
5. write the training loop
6. write the validation loop
7. run epochs and save the best model

The important idea is that a training pipeline is mostly plumbing: each piece has one job, and the full program works because those jobs are connected in the right order.

### What problem is this example solving?

This is a regression example.

That means:

- input: a vector of features
- output: one continuous number

So the final layer has one output neuron, and the loss is `MSELoss` rather than a classification loss like `CrossEntropyLoss`.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ============================================================
# STEP 1: Generate data with a real learnable pattern
# ============================================================
torch.manual_seed(42)

# y depends on x[:,0] and x[:,1] in a known way — the model should learn this
X = torch.rand(500, 10)
y = (X[:, 0] * 3 + X[:, 1] - 1          # true relationship (hidden from model)
     + torch.randn(500) * 0.1)            # + small gaussian noise
y = y.unsqueeze(1)                         # shape (500,) → (500, 1) for MSELoss

# Train/validation split manually
X_train, X_val = X[:400], X[400:]
y_train, y_val = y[:400], y[400:]

# ============================================================
# STEP 2: Create DataLoaders
# ============================================================
# TensorDataset wraps tensors into a dataset that returns (X[i], y[i]) pairs
train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val,   y_val)

# DataLoader handles batching, shuffling, and (optionally) parallel loading
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)   # Shuffle training data each epoch
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)  # Never shuffle validation

# ============================================================
# STEP 3: Define the Model
# ============================================================
class RegressionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()    # ALWAYS call this — initializes PyTorch internals
        # Sequential: layers applied one after another
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),   # Input layer → 64 hidden neurons
            nn.ReLU(),                   # Non-linearity: max(0, x)
            nn.Linear(64, 32),           # 64 → 32 hidden neurons
            nn.ReLU(),
            nn.Linear(32, 1)             # 32 → 1 output (single regression target)
            # No activation on last layer for regression!
        )

    def forward(self, x):
        return self.net(x)   # Pass input through all layers in sequence

# ============================================================
# STEP 4: Setup (model, loss, optimizer, scheduler)
# ============================================================
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = RegressionNet(input_dim=10).to(device)    # Move model to GPU if available
criterion = nn.MSELoss()                               # Regression loss
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# ============================================================
# STEP 5: Training Loop (one epoch)
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()   # Switches model to training mode: enables Dropout, BatchNorm updates
    total_loss = 0.0

    for X_batch, y_batch in loader:
        # Move data to the same device as the model
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()               # Clear gradients from previous batch
        preds = model(X_batch)              # Forward pass: model predicts
        loss  = criterion(preds, y_batch)   # Compute loss: measure prediction error
        loss.backward()                     # Backward pass: compute gradients
        optimizer.step()                    # Update weights using gradients

        total_loss += loss.item()          # Accumulate scalar loss (not tensor!)
    return total_loss / len(loader)        # Average loss across batches

# ============================================================
# STEP 6: Validation Loop
# ============================================================
def evaluate(model, loader, criterion, device):
    model.eval()   # Switches to eval mode: disables Dropout, freezes BatchNorm stats
    total_loss = 0.0
    with torch.no_grad():   # No gradient computation needed — saves memory and time
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds   = model(X_batch)
            total_loss += criterion(preds, y_batch).item()
    return total_loss / len(loader)

# ============================================================
# STEP 7: Run Training
# ============================================================
num_epochs    = 50
best_val_loss = float('inf')             # Track the best validation loss seen so far
history       = {"train_loss": [], "val_loss": []}

for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss   = evaluate(model, val_loader, criterion, device)
    scheduler.step()   # Update learning rate after each epoch

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    # Save the best model checkpoint (best validation loss, not train loss!)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")   # Save only weights

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
```

### Reading the complete example as a story

Follow one epoch to understand the flow:

1. The outer loop calls `train_one_epoch(...)` with the current model state
2. `model.train()` enables train-time behaviors (Dropout randomness, BatchNorm batch statistics)
3. The inner loop reads one mini-batch from `train_loader`
4. That batch is moved to the device (GPU if available)
5. `optimizer.zero_grad()` clears any leftover gradients
6. `model(X_batch)` runs the forward pass with current weights
7. `criterion(preds, y_batch)` computes a single error scalar for this batch
8. `loss.backward()` computes `∂loss/∂w` for every weight
9. `optimizer.step()` adjusts every weight by its gradient
10. The inner loop repeats for each batch until the epoch ends
11. `evaluate(...)` runs validation: no gradients, just forward passes and loss accumulation
12. `scheduler.step()` updates the learning rate for the next epoch
13. If validation improved, a checkpoint is saved

### What runs once vs every epoch vs every batch

```text
Run once at the start:
  create data, DataLoaders, model, loss, optimizer, scheduler

Repeat once per epoch:
  train_one_epoch (runs the batch loop)
  evaluate (validation)
  scheduler.step()
  save checkpoint if improved

Inside train_one_epoch, repeat once per batch:
  zero_grad → forward → loss → backward → optimizer.step()
```

---

## 8. A Minimal Pipeline (No DataLoader)

Sometimes the raw tensor version makes the loop easier to understand. This tiny example has the same 4 essential steps:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simple linear relationship: y = 2x
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])   # input
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])   # target

model     = nn.Linear(1, 1)        # One input, one output — learns y ≈ w*x + b
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    optimizer.zero_grad()    # 1. Clear old gradients
    preds = model(X)         # 2. Forward: predict using current weight
    loss  = criterion(preds, y)  # 3. Loss: how far are predictions from targets?
    loss.backward()          # 4. Backward: compute d(loss)/d(weight)
    optimizer.step()         # 5. Update: weight = weight - lr * gradient

# After 100 epochs, model.weight should be ≈ 2.0 and model.bias ≈ 0.0
print(f"Learned weight: {model.weight.item():.4f}")
print(f"Learned bias:   {model.bias.item():.4f}")
```

This tiny loop contains the same four core actions (forward, loss, backward, step) that every PyTorch model uses. The larger DataLoader-based version is just this loop with batches, validation, and scheduling layered on top.

---

## 9. Common Training Mistakes

```python
# ── Mistake 1: Forgetting model.train() before training ───────
# Without model.train(), Dropout stays in eval mode (all neurons active)
# → model over-fits because regularization is off during training
model.train()  # ← call this before the training loop

# ── Mistake 2: Forgetting model.eval() during validation ──────
# Without model.eval(), BatchNorm uses batch statistics even during evaluation
# → noisy, inconsistent predictions that vary with batch size
model.eval()   # ← call this before the validation loop

# ── Mistake 3: Not zeroing gradients ──────────────────────────
# Gradients accumulate by default — previous step's gradients ADD to current ones
optimizer.zero_grad()  # ← call this BEFORE loss.backward() every step

# ── Mistake 4: Wrong target dtype for CrossEntropyLoss ─────────
# CrossEntropyLoss expects labels as torch.long (int64)
targets = torch.tensor([0, 1, 2], dtype=torch.long)  # ✅
# targets = torch.tensor([0, 1, 2])  # ← int64 is the default for integer literals ✅
# targets = torch.tensor([0.0, 1.0, 2.0])  # ← float32 → RuntimeError ❌

# ── Mistake 5: Applying softmax before CrossEntropyLoss ────────
# CrossEntropyLoss applies LogSoftmax internally
# Applying softmax first → double-softmax → incorrect gradients
# model output should be RAW LOGITS, not probabilities

# ── Mistake 6: Computing loss inside no_grad() for training ────
# with torch.no_grad():
#     loss = criterion(model(X), y)
#     loss.backward()   # ❌ RuntimeError: no graph was built
```

---

## 10. Model Save & Load

```python
# ── SAVE ────────────────────────────────────────────────────
# Method 1: Save only weights (RECOMMENDED)
# state_dict() is a Python dict mapping layer names to parameter tensors
torch.save(model.state_dict(), "model_weights.pth")

# Method 2: Save entire model object (brittle — tied to class definition location)
torch.save(model, "full_model.pth")

# Method 3: Full training checkpoint (for resuming interrupted training)
torch.save({
    "epoch":           epoch,
    "model_state":     model.state_dict(),
    "optimizer_state": optimizer.state_dict(),   # Adam's momentum buffers, etc.
    "scheduler_state": scheduler.state_dict(),
    "val_loss":        val_loss,
}, "checkpoint.pth")

# ── LOAD ────────────────────────────────────────────────────
# Load weights into a freshly instantiated model
model = RegressionNet(input_dim=10)             # Build the same architecture first
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.to(device)     # Move to device after loading
model.eval()         # Always set eval mode for inference after loading

# Load full checkpoint (to resume training exactly where it stopped)
checkpoint = torch.load("checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
scheduler.load_state_dict(checkpoint["scheduler_state"])
start_epoch = checkpoint["epoch"]      # Resume from this epoch
best_val    = checkpoint["val_loss"]   # Resume best metric tracking
```

For long-running jobs, always save a full checkpoint (weights + optimizer state + epoch). The optimizer state contains Adam's momentum buffers — without it, the first few epochs after resuming will have incorrect update steps.

---

## 11. Plotting Training Curves

Training curves are one of the most important diagnostic tools. They tell you immediately whether the model is learning, overfitting, or underfitting:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"],   label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# How to interpret the curves:
# train loss >> val loss    → Underfitting (model too simple; train longer or bigger)
# val loss >> train loss    → Overfitting  (add regularization or more data)
# both losses high          → Underfitting
# val loss rises while train falls → Classic overfitting — use early stopping
# both losses low and close → Good fit ✅
```

---

## 12. Interview Questions

<details>
<summary><strong>Q1: What are the 7 steps of a PyTorch training loop?</strong></summary>

1. Move data to device (`.to(device)`)
2. `optimizer.zero_grad()` — reset accumulated gradients from previous step
3. Forward pass: `output = model(input)` — compute predictions
4. Compute loss: `loss = criterion(output, target)` — measure error
5. Backward pass: `loss.backward()` — compute gradients via chain rule
6. Update weights: `optimizer.step()` — adjust weights using gradients
7. Log metrics: `loss.item()`, accuracy, etc. — track training progress
</details>

<details>
<summary><strong>Q2: Why do we use BCEWithLogitsLoss instead of BCELoss?</strong></summary>

`BCEWithLogitsLoss` combines sigmoid + BCE loss into one numerically stable operation using the log-sum-exp trick. The problem with separate sigmoid → BCELoss: when logits are very large (e.g., 100), `sigmoid(100) = 1.0` exactly in float32, and `log(1 - 1.0) = log(0) = -inf` → NaN loss. The fused version avoids this by computing `log(sigmoid(x))` in a numerically stable way. Always pass raw logits to `BCEWithLogitsLoss`.
</details>

<details>
<summary><strong>Q3: What does model.train() vs model.eval() affect?</strong></summary>

- `model.train()`: **Dropout** randomly zeros activations (regularization); **BatchNorm** normalizes using current batch statistics and updates its running mean/var.
- `model.eval()`: **Dropout** is disabled (all neurons active, full capacity); **BatchNorm** uses stored running statistics (not batch-dependent → consistent predictions).

Forgetting `model.eval()` during validation causes noisy, non-deterministic predictions (Dropout is random) and incorrect BatchNorm behavior. This is one of the most common bugs in PyTorch code.
</details>

<details>
<summary><strong>Q4: What is a learning rate scheduler and why use one?</strong></summary>

A scheduler adjusts the learning rate during training automatically. A fixed LR is often suboptimal: too high → training diverges or oscillates; too low → slow convergence. Common strategies: start high and decay over time (StepLR, CosineAnnealing), reduce when a metric plateaus (ReduceLROnPlateau), warmup then decay (OneCycleLR). Cosine annealing is a safe default — it smoothly reduces LR without sudden drops and often finds better final accuracy than a fixed LR.
</details>

<details>
<summary><strong>Q5: Why save model.state_dict() instead of the full model?</strong></summary>

`state_dict()` saves only the parameter tensors (weights and biases) as a Python dict. When loading, you instantiate the class first, then load weights — this is portable across code changes. Saving the full model with `torch.save(model, ...)` pickles the entire class definition, which breaks if you rename the class, move the file, change method names, or upgrade PyTorch. `state_dict()` is the recommended approach for anything that will be used beyond a single session.
</details>

---

## 🔗 References
- [PyTorch Training Tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [CampusX Video 4](https://www.youtube.com/watch?v=MKxEbbKpL5Q)
