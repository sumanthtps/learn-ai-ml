---
id: dropout-code-examples
title: "Dropout code examples for regression and classification"
sidebar_label: "25 · Dropout Code"
sidebar_position: 25
slug: /theory/dnn/dropout-code-examples-for-regression-and-classification
description: "Complete PyTorch code examples showing how to apply dropout correctly for regression and classification tasks, including the critical train/eval mode distinction."
tags: [dropout, pytorch, regularization, deep-learning]
---

# Dropout code examples for regression and classification

Note 24 explained the theory of dropout — the ensemble interpretation, inverted dropout scaling, and why it prevents co-adaptation. This note focuses on the practical implementation: complete PyTorch examples for regression and classification, including the most common pitfall (forgetting to switch between train and eval mode).

## One-line definition

Dropout in code is implemented as a `nn.Dropout(p)` layer that randomly zeros activations with probability `p` during training and passes all activations unchanged during evaluation.

## Why this topic matters

Dropout is theoretically elegant, but correctly using it in PyTorch requires understanding one critical detail: the module has two modes that must be switched explicitly. Forgetting to call `model.eval()` before inference is one of the most common bugs in PyTorch code — it produces stochastic, unreliable predictions.

## The train/eval mode distinction

During **training** (`model.train()`):
- Each activation is independently zeroed with probability `p`
- Surviving activations are scaled by `1/(1-p)` (inverted dropout — keeps expected values correct)

During **evaluation/inference** (`model.eval()`):
- No activations are zeroed
- No scaling is applied (the 1/(1-p) scaling already compensates during training)

![Dropout in train and eval modes — during training, neurons are randomly dropped out (zeroed) with probability p; during evaluation, all neurons are active](https://commons.wikimedia.org/wiki/Special:Redirect/file/MultiLayerPerceptron.svg)
*Source: [Wikimedia Commons — MultiLayerPerceptron](https://commons.wikimedia.org/wiki/File:MultiLayerPerceptron.svg) (CC BY-SA 4.0)*

```python
import torch
import torch.nn as nn

layer = nn.Dropout(p=0.5)
x = torch.ones(1, 10)

layer.train()
print("Training mode:", layer(x))     # ~half values are 0, rest are 2.0

layer.eval()
print("Eval mode:", layer(x))         # all values are 1.0 (no dropout)
```

The factor 2.0 during training (when p=0.5) is inverted dropout: surviving activations are scaled by 1/(1-0.5) = 2 to keep the expected value unchanged.

## Dropout for binary classification

```python
import torch
import torch.nn as nn
import torch.optim as optim

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, dropout_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),       # after activation, before next linear
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 1)            # no sigmoid — use BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.net(x)

# Setup
model = BinaryClassifier(input_dim=64, dropout_p=0.3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# Training loop
model.train()                            # MUST be in train mode for dropout to activate
for epoch in range(3):
    x = torch.randn(128, 64)
    y = torch.randint(0, 2, (128, 1)).float()

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

# Evaluation — CRITICAL: switch to eval mode
model.eval()
with torch.no_grad():
    x_test = torch.randn(32, 64)
    logits_test = model(x_test)
    probs = torch.sigmoid(logits_test)
    preds = (probs > 0.5).int()
    print("Predictions:", preds[:5].flatten())
```

## Dropout for multi-class classification

```python
class MultiClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_p=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),         # BN + Dropout can be combined
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes)  # no softmax — CrossEntropyLoss handles it
        )

    def forward(self, x):
        return self.net(x)

model = MultiClassifier(input_dim=128, num_classes=10, dropout_p=0.4)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Training
model.train()
x = torch.randn(64, 128)
y = torch.randint(0, 10, (64,))

optimizer.zero_grad()
logits = model(x)
loss = criterion(logits, y)
loss.backward()
optimizer.step()

# Validation
model.eval()
with torch.no_grad():
    val_x = torch.randn(32, 128)
    val_y = torch.randint(0, 10, (32,))
    val_logits = model(val_x)
    val_loss = criterion(val_logits, val_y)
    preds = val_logits.argmax(dim=1)
    accuracy = (preds == val_y).float().mean()
    print(f"Val loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
```

## Dropout for regression

```python
class RegressionModel(nn.Module):
    def __init__(self, input_dim, dropout_p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),       # Lower dropout for regression (0.1-0.3)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 1)             # scalar output, no activation
        )

    def forward(self, x):
        return self.net(x)

model = RegressionModel(input_dim=32, dropout_p=0.2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

model.train()
x = torch.randn(64, 32)
y = torch.randn(64, 1)

optimizer.zero_grad()
preds = model(x)
loss = criterion(preds, y)
loss.backward()
optimizer.step()

# Monte Carlo dropout for prediction uncertainty (optional)
# Keep dropout ON during inference to get stochastic predictions
model.train()   # intentionally in train mode for MC dropout
mc_preds = torch.stack([model(x) for _ in range(50)])  # 50 stochastic forward passes
mean_pred = mc_preds.mean(dim=0)
uncertainty = mc_preds.std(dim=0)
print("Prediction mean shape:", mean_pred.shape)
print("Uncertainty shape:", uncertainty.shape)
```

## Dropout placement guidelines

| Where | Recommendation |
|---|---|
| After activation functions | Standard and most common |
| Before the final linear layer | Common for regularization |
| Between conv and linear in CNNs | Yes — reduces overfitting in dense head |
| Inside recurrent cells | Use `nn.RNN(dropout=p)` between layers only |
| Before batch normalization | Generally not recommended (BN + Dropout interaction is complex) |
| After batch normalization | More common — BN first, then Dropout |

## Interview questions

<details>
<summary>What is inverted dropout and why is it necessary?</summary>

Inverted dropout scales surviving activations by 1/(1-p) during training so that their expected value equals the original value: E[dropped_output] = (1-p)·(1/(1-p))·x = x. Without this scaling, the expected activation during training would be (1-p)·x, but during evaluation it would be x — a scale mismatch that would require dividing by (1-p) at evaluation time. Inverted dropout puts all the scaling at training time, so evaluation can run without any modification.
</details>

<details>
<summary>What happens if you forget to call model.eval() before inference?</summary>

The model will still be in training mode, so dropout continues to randomly zero activations. Each forward pass produces a different, stochastic output for the same input. Predictions are noisy and unreliable. Loss metrics computed in this mode are also noisy. This is one of the most common silent bugs in PyTorch — the model appears to work but produces inconsistent results.
</details>

<details>
<summary>What is Monte Carlo dropout and when is it useful?</summary>

MC dropout intentionally keeps dropout active during inference and runs multiple stochastic forward passes. The mean of the predictions is the point estimate; the variance across passes is an uncertainty estimate. It provides approximate Bayesian uncertainty quantification for free, using a model that was only trained with standard dropout. It is useful in high-stakes applications (medical, autonomous systems) where uncertainty estimates help identify out-of-distribution inputs.
</details>

## Common mistakes

- Forgetting `model.eval()` before validation or test — this is the #1 dropout bug.
- Using `torch.no_grad()` but forgetting `model.eval()` — `no_grad()` only stops gradient computation; it does not change the dropout mode.
- Using dropout rates > 0.5 for regression tasks — high dropout rates can underfit continuous outputs; 0.1–0.3 is typical for regression.
- Not using dropout at all because batch normalization is present — they serve different purposes and can be used together.

## Final takeaway

Dropout implementation in PyTorch is two lines: `nn.Dropout(p)` in the model, and `model.train()` / `model.eval()` to switch modes. The most important rule is never forget `model.eval()` before inference. The inverted dropout scaling ensures the expected values are consistent between training and evaluation without any manual adjustment.

## References

- Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. ICML.
