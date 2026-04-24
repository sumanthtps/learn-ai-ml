---
id: handwritten-digit-classification-ann
title: "Handwritten digit classification with an ANN in PyTorch"
sidebar_label: "12 · MNIST Digit Classification"
sidebar_position: 12
slug: /theory/dnn/handwritten-digit-classification-with-an-ann-in-pytorch
description: "Multi-class classification on MNIST using a dense ANN: image flattening, CrossEntropyLoss, softmax, and accuracy evaluation."
tags: [mnist, multiclass-classification, pytorch, ann, image-classification, deep-learning]
---

# Handwritten digit classification with an ANN in PyTorch

The churn project covered binary classification on tabular data. This project extends the same pipeline to ten-class classification on images, introducing `torchvision`, pixel normalization, and `CrossEntropyLoss`.

## One-line definition

MNIST digit classification is a ten-class problem where a dense neural network reads a flattened 784-pixel grayscale image and predicts which digit (0–9) it depicts.

## Why this topic matters

MNIST is the canonical benchmark for verifying that a deep learning pipeline works correctly. The jump from binary to multi-class introduces a new loss (`CrossEntropyLoss` instead of `BCEWithLogitsLoss`) and requires understanding the relationship between raw logits, softmax, and class probabilities. The image-flattening step makes explicit a fundamental limitation that motivates the later shift to CNNs.

## Dataset and preprocessing

MNIST contains 70,000 grayscale images of handwritten digits:

- 60,000 training images
- 10,000 test images
- Each image: 28 × 28 = 784 pixels, values in [0, 255]

Standard preprocessing:

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),          # converts PIL image to [0,1] float tensor
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST global mean and std
])
```

The global mean 0.1307 and standard deviation 0.3081 are precomputed on the training set. Normalizing pixel values to zero mean and unit variance stabilizes gradient flow through the first layer.

## Architecture: flattened image as input vector

```mermaid
flowchart LR
    A["Input image 28×28"] --> B["Flatten → 784"]
    B --> C["Linear(784→512)"]
    C --> D["ReLU"]
    D --> E["Dropout(0.2)"]
    E --> F["Linear(512→256)"]
    F --> G["ReLU"]
    G --> H["Linear(256→10)"]
    H --> I["10 logits (one per class)"]
```

The key difference from binary classification: the output layer has 10 neurons, producing one unnormalized score (logit) per class.

## The loss function for multi-class classification

`CrossEntropyLoss` in PyTorch combines `LogSoftmax` + `NLLLoss` in a single numerically stable operation:

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{z_{i,y_i}}}{\sum_{j=1}^{C}e^{z_{i,j}}}
$$

where $z_{i,j}$ is the logit for sample $i$ and class $j$, $y_i$ is the true class label, and $C = 10$ for MNIST.

This is equivalent to:

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{i=1}^{N}\left(z_{i,y_i} - \log\sum_{j=1}^{C}e^{z_{i,j}}\right)
$$

Important: **pass raw logits to `CrossEntropyLoss`**. Do not apply softmax before this loss.

## Converting logits to predictions

At inference time, the predicted class is the argmax of the logits:

$$
\hat{y} = \arg\max_j \; z_j
$$

$$
p(y = j \mid x) = \frac{e^{z_j}}{\sum_{k=1}^{C} e^{z_k}}
$$

The argmax of logits equals the argmax of softmax probabilities because softmax is monotone, so you do not need to compute softmax to get the predicted label.

## PyTorch example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Data ──────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

# ── Architecture ──────────────────────────────────────────────────────────────
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()          # 28×28 → 784
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10),              # 10 output logits
        )

    def forward(self, x):
        return self.net(self.flatten(x))

model     = DigitNet()
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── Training loop ─────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            preds  = logits.argmax(dim=1)    # predicted class = argmax logit
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total

for epoch in range(1, 11):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        logits = model(images)               # shape: (128, 10)
        loss   = loss_fn(logits, labels)     # labels are integer class indices
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if epoch % 2 == 0:
        val_acc = evaluate(model, test_loader)
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch:2d} | loss={avg_loss:.4f} | test_acc={val_acc:.4f}")
```

## Expected results

A well-tuned dense network on MNIST typically achieves:

| Model depth | Test accuracy |
|---|---|
| 1 hidden layer (512 units) | ~97.5% |
| 2 hidden layers (512, 256) | ~98.0% |
| Same architecture + BatchNorm | ~98.2% |
| CNN (LeNet-5) | ~99.0%+ |

The gap between the best dense network and a CNN (~1%) motivates the CNN module.

## The flattening limitation

Flattening a 28×28 image into a 784-vector destroys all spatial structure. A pixel at position (3, 4) has no positional relationship to its neighbor (3, 5) after flattening. This means:

- the network cannot exploit local patterns (edges, curves)
- the same digit shifted by one pixel requires the network to re-learn the pattern
- the parameter count scales poorly: a 32×32 image needs 1024 inputs, while a 256×256 image needs 65,536

Convolutional neural networks solve this by operating locally on spatial patches.

## Interview questions

<details>
<summary>Why are raw logits passed to CrossEntropyLoss instead of softmax probabilities?</summary>

PyTorch's `CrossEntropyLoss` internally computes `log(softmax(logits))` using the log-sum-exp trick for numerical stability. Passing pre-applied softmax probabilities leads to computing `log(softmax(softmax(logits)))`, which is incorrect and numerically unstable.
</details>

<details>
<summary>Why is the output layer size 10, not 1, for MNIST?</summary>

Multi-class classification with $C$ classes requires $C$ output logits, one per class. The predicted class is the argmax. This is different from binary classification, which uses a single logit and sigmoid.
</details>

<details>
<summary>What is the mathematical relationship between CrossEntropyLoss and NLLLoss in PyTorch?</summary>

`CrossEntropyLoss = LogSoftmax + NLLLoss`. `NLLLoss` computes the negative log-likelihood from log-probabilities. `CrossEntropyLoss` accepts raw logits and handles the log-softmax internally.
</details>

<details>
<summary>Why does flattening images hurt performance on ANN relative to CNN?</summary>

Flattening destroys the 2D spatial structure. Nearby pixels are no longer neighbors in the vector. CNNs exploit local spatial correlations through shared convolutional filters, which is far more efficient and effective for image data.
</details>

<details>
<summary>Why does MNIST normalization use (0.1307, 0.3081) specifically?</summary>

These are the empirically computed mean and standard deviation of the MNIST training set pixel values after the ToTensor transform (which rescales to [0, 1]). Using them ensures the input distribution has zero mean and unit variance, which helps gradient flow.
</details>

<details>
<summary>How does dropout placement affect the network during training vs evaluation?</summary>

During training, dropout randomly zeroes activations with probability p, scaling the remaining ones by 1/(1-p). During evaluation (`model.eval()`), dropout is disabled — all neurons are active. This means calling `model.eval()` before evaluation is mandatory for correct test accuracy.
</details>

## Common mistakes

- Applying softmax before `CrossEntropyLoss` (double-softmax)
- Using `labels` as float tensors instead of long integer tensors for `CrossEntropyLoss`
- Forgetting `model.eval()` before computing test accuracy (dropout active during evaluation inflates error)
- Using one-hot encoded labels instead of integer class indices with `CrossEntropyLoss`
- Not normalizing pixel values (or using wrong normalization statistics)
- Evaluating accuracy as `(preds == labels).mean()` on float tensors — always compare on the same dtype

## Advanced perspective

MNIST is almost "solved" for dense networks; the main constraint is the spatial ignorance of MLPs. The interesting engineering question is: how does accuracy scale with parameter count? A linear classifier achieves ~92%, a shallow MLP achieves ~97.5%, and a deep MLP with dropout achieves ~98.2%. The next 0.8% requires spatial inductive bias (convolutions), highlighting that architecture encodes assumptions about the data structure.

## Final takeaway

MNIST digit classification is the standard sanity check for a PyTorch pipeline. The critical concepts are: flatten → linear layers → 10-way logits → CrossEntropyLoss → argmax prediction. The same pattern scales to any C-class problem. The gap between dense network accuracy and CNN accuracy motivates why spatial architectures are needed for image tasks.

## References

- CampusX YouTube: Handwritten Digit Classification using ANN in PyTorch
- LeCun et al. (1998): Gradient-Based Learning Applied to Document Recognition
- MNIST database: http://yann.lecun.com/exdb/mnist/
