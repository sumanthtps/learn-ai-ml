---
id: 12-transfer-learning
title: "Video 12: Transfer Learning using PyTorch"
sidebar_label: "12 · Transfer Learning"
sidebar_position: 12
description: Fine-tuning pretrained models (ResNet, EfficientNet, VGG) for custom image classification with minimal data.
tags: [pytorch, transfer-learning, fine-tuning, resnet, efficientnet, pretrained, campusx]
---

# Transfer Learning using PyTorch
**📺 CampusX — Practical Deep Learning using PyTorch | Video 12**

> **What you'll learn:** How to reuse powerful pretrained models and adapt them to your own image classification task — saving weeks of training and requiring only a fraction of the data.

---

## 1. What is Transfer Learning?

Training a deep CNN from scratch needs:
- 1M+ labeled images
- Weeks of GPU training
- Deep expertise in architecture design

**Transfer learning** uses a model already trained on a large dataset (ImageNet — 1.2M images, 1000 classes) and adapts it to your task (e.g., 5 dog breeds with 500 images).

**Why does it work?** CNNs learn a hierarchy of features:
```
Early layers  → Edges, corners, textures    (universal — any image)
Middle layers → Shapes, parts               (somewhat general)
Late layers   → Object-specific features    (task-specific)
```

Early features are **universal** — they transfer perfectly to any image task. Only the final layers need to be task-specific.

## Visual Reference

![Feature extraction vs fine-tuning transfer learning strategies](https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/05-transfer-learning-feature-extraction-vs-fine-tuning.png)

*Feature extraction (left) freezes all pretrained backbone weights and trains only a new classification head — best when your dataset is small. Fine-tuning (right) unfreezes some or all backbone layers and trains end-to-end with a low learning rate — better when you have more data. Both strategies reuse the universal low-level features (edges, textures) learned on ImageNet.*

### Industry perspective

For most teams, transfer learning is the default starting point for vision projects because it reduces:

- data requirements
- training time
- hyperparameter search cost
- the risk of badly undertrained models

---

## 2. Three Strategies

| Strategy | When to use | What to train |
|---|---|---|
| **Feature Extraction** | Small dataset (< 5K), similar to ImageNet | Freeze all backbone, train new head only |
| **Fine-tuning** | Medium dataset (5K–50K) or different domain | Unfreeze backbone with low LR + train head |
| **From scratch** | Very large dataset, very different domain | Train all weights |

**Decision guide:**
```
Dataset SIZE:  Small ──────────────────────────── Large
               ↓                                   ↓
               Feature extraction             Fine-tuning / From scratch

Domain:   Similar to ImageNet ──────────────── Very different
               ↓                                   ↓
               Feature extraction             More fine-tuning needed
```

---

## 3. Loading Pretrained Models

```python
import torchvision.models as models

# ResNet family
resnet18  = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet50  = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)  # V2 = better
resnet101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)

# EfficientNet (state-of-the-art accuracy/efficiency)
eff_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
eff_b4 = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)

# VGG
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# MobileNet (fast, lightweight)
mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

# Vision Transformer
vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

# List all available
print(models.list_models()[:10])

# No pretrained weights (random init — for from-scratch training)
resnet50_scratch = models.resnet50(weights=None)
```

---

## 4. Strategy 1: Feature Extraction

Before the code, here is the key beginner idea:

### What does "feature extraction" mean?

A pretrained CNN already knows how to detect useful visual patterns such as:

- edges
- corners
- textures
- object parts

Those internal patterns are called **features**.

When we do feature extraction, we keep that pretrained feature detector mostly unchanged and only train a new classifier on top of it for our task.

So the model is split into two conceptual parts:

- **backbone**: the pretrained feature extractor
- **head**: the final task-specific classifier

In ResNet, the backbone is most of the network and the head is the final `fc` layer.

### What does "freeze the backbone" mean?

Freezing means:

- gradients are not used to update those parameters
- the pretrained weights stay fixed during training
- only the new head learns

This is usually the safest starting point when:

- your dataset is small
- your task is similar to natural images
- you want a strong baseline quickly

Now the code below is easier to read: we are keeping the pretrained feature extractor and swapping in a new classifier.

```python
import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 5   # Your task

# Load pretrained ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Step 1: Freeze ALL backbone parameters
for param in model.parameters():
    param.requires_grad = False

# Step 2: Replace the final classification head
# ResNet50's final layer: model.fc = Linear(2048, 1000)
in_features = model.fc.in_features   # 2048
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, NUM_CLASSES)
)
# model.fc is trainable by default (requires_grad=True)

# Verify
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"TRAINABLE: {name}")
# Only: fc.0.weight, fc.0.bias, fc.1.weight, fc.1.bias

# Count params
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}, Trainable: {trainable:,}")
# Total: ~25M, Trainable: ~10K  (only the head!)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Only pass trainable parameters to optimizer
optimizer = torch.optim.AdamW(
    model.fc.parameters(),   # Only the head
    lr=1e-3
)
```

### Code walkthrough

- freezing parameters means backpropagation skips weight updates for the backbone
- only `model.fc` is replaced, because the backbone already knows useful visual features
- the optimizer receives only the head parameters, which avoids wasting compute on frozen tensors

### CampusX project note: VGG16 as the concrete example

In the transcript, transfer learning is introduced through a very practical
vision-project lens: after pushing ANN and CNN performance manually, the next
step is to use a **pretrained model** such as `VGG16`.

That is a valuable teaching move because it highlights the main reason transfer
learning is so popular in industry:

- you get a much stronger starting point than a hand-built small CNN
- you often need only a new classification head
- you can try multiple pretrained families like VGG, ResNet, MobileNet, or Inception with the same overall workflow

---

## 5. Strategy 2: Fine-tuning

Fine-tuning goes further than feature extraction: instead of freezing the backbone, we **unfreeze it** and let gradients update the pretrained weights. This lets the model adapt its internal representations to your specific domain.

### What changes when we move from feature extraction to fine-tuning?

Feature extraction says:

- "keep the backbone fixed"
- "train only the new head"

Fine-tuning says:

- "start from pretrained weights"
- "but allow some or all of them to change"

This is more powerful, but also more risky. Once the backbone is trainable, your dataset can push those good pretrained weights in the wrong direction if training is too aggressive.

The challenge: if you unfreeze the backbone and use a large learning rate, the pretrained weights get overwritten by gradients from your small dataset. The model **"forgets"** the rich ImageNet features it spent weeks learning — this is called **catastrophic forgetting**. After a few steps, your model is effectively trained from scratch, but on less data.

The fix is **differential learning rates**: use a very small learning rate for the backbone (so pretrained features change only slightly) and a normal learning rate for the new head (which needs to learn from scratch):

### What are differential learning rates?

Instead of giving one learning rate to the whole model, we give different parts of the model different learning rates.

Why?

- early layers learn very general features, so they need tiny updates
- later layers are more task-specific, so they can change a bit more
- the new head starts random, so it needs the biggest updates

This is why the optimizer below receives a **list of parameter groups** instead of a single `model.parameters()` call. Each group is one part of the model with its own learning rate.

```python
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Replace the classification head with one for your task
in_features = model.classifier[1].in_features   # 1280 for EfficientNet-B0
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, NUM_CLASSES)
)

# Unfreeze ALL parameters — backbone + head — for end-to-end fine-tuning
for param in model.parameters():
    param.requires_grad = True

# CRITICAL: Use differential learning rates to prevent catastrophic forgetting.
# The intuition: early backbone layers already learned universal features
# (edges, textures) that need minimal change. Late backbone layers learned
# ImageNet-specific features that may need moderate adjustment. The new
# classification head knows nothing — it needs to learn fast.
optimizer = torch.optim.AdamW([
    # Early backbone blocks: very low LR (1e-5).
    # These are the "universal" layers — gentle nudges only.
    {"params": [p for n, p in model.features.named_parameters()
                if int(n.split('.')[0]) < 4],    # First 4 EfficientNet feature blocks
     "lr": 1e-5},

    # Later backbone blocks: medium LR (1e-4).
    # These are more task-specific — can be updated a bit more aggressively.
    {"params": [p for n, p in model.features.named_parameters()
                if int(n.split('.')[0]) >= 4],
     "lr": 1e-4},

    # New classification head: full LR (1e-3).
    # Starts from random initialization — needs large steps to converge.
    {"params": model.classifier.parameters(),
     "lr": 1e-3},
], weight_decay=1e-4)
# Result: three "speed lanes" in one optimizer — PyTorch handles them simultaneously.
```

---

## 6. Architecture-Specific Head Replacement

This section is here because "replace the last layer" sounds simple, but every architecture stores that last layer in a different place.

Beginner rule of thumb:

- the idea is always the same: keep the pretrained body, replace the final classifier
- the code path is different because model classes use different attribute names

So you are not learning four different transfer-learning techniques here. You are learning one technique with four different model-specific APIs.

```python
# Different architectures have different final layer names:

# ResNet18/34/50/101/152
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# VGG16/19
in_features = model.classifier[6].in_features   # 4096
model.classifier[6] = nn.Linear(in_features, num_classes)

# EfficientNet-B0 to B7
in_features = model.classifier[1].in_features   # 1280
model.classifier[1] = nn.Linear(in_features, num_classes)

# MobileNetV3-Large
in_features = model.classifier[3].in_features   # 1280
model.classifier[3] = nn.Linear(in_features, num_classes)

# DenseNet121
in_features = model.classifier.in_features      # 1024
model.classifier = nn.Linear(in_features, num_classes)

# ViT-B/16
in_features = model.heads.head.in_features      # 768
model.heads.head = nn.Linear(in_features, num_classes)
```

---

## 7. Full Transfer Learning Pipeline

This section puts the earlier ideas into one end-to-end workflow.

### How should a beginner read this pipeline?

As five separate jobs:

1. prepare image transforms
2. load datasets and DataLoaders
3. load a pretrained model and replace the head
4. train the new head first
5. then fine-tune the full network carefully

The code is long because transfer learning has several moving parts, not because each individual part is conceptually complicated.

### Why are there separate train and validation transforms?

Because they serve different purposes:

- training transforms add randomness and augmentation so the model generalizes better
- validation transforms should be stable and repeatable so evaluation is fair

That is why the training pipeline includes random crop/flip/jitter, while validation uses deterministic resize + center crop.

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# ── 1. ImageNet normalization (MUST use for pretrained models!) ──
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# ── 2. Data (folder structure: data/train/class_name/img.jpg) ──
# Expects: data/train/cats/..., data/train/dogs/...
train_data = ImageFolder('data/train', transform=train_transform)
val_data   = ImageFolder('data/val',   transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_data,   batch_size=64, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f"Classes: {train_data.classes}")
print(f"Samples: {len(train_data)} train, {len(val_data)} val")
NUM_CLASSES = len(train_data.classes)

# ── 3. Model ─────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Phase 1: Feature extraction
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model = model.to(device)

# ── 4. Phase 1: Train head only (5 epochs) ───────────────────
# Why phase 1? The backbone is frozen — only the new head trains.
# This quickly calibrates the head to produce reasonable class scores.
# If we jumped straight to phase 2 (unfreezing backbone), the head's
# random gradients would propagate into the well-trained backbone and damage it.
# Phase 1 "warms up" the head before we risk touching the backbone.
#
# label_smoothing=0.1: instead of hard targets (0 or 1), use 0.1 and 0.9.
# Prevents the model from becoming overconfident on training data.
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Only optimize the new head — backbone is frozen, so its params have no gradients anyway
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
# CosineAnnealingLR: smoothly decays LR from 1e-3 to near zero over T_max=5 epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

best_val_acc, best_state = 0, None

for epoch in range(5):
    model.train()   # Even in phase 1, set train mode so BatchNorm updates running stats
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        criterion(model(X), y).backward()   # Gradient flows only to model.fc (head)
        optimizer.step()                    # Only head parameters are updated
    scheduler.step()   # Reduce LR after each epoch

    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in val_loader:
            correct += (model(X.to(device)).argmax(1) == y.to(device)).sum().item()
    val_acc = correct / len(val_data)
    print(f"Phase 1 Epoch {epoch+1}: val_acc={val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # Save entire model state (backbone + head) so we can restore best result later
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

# ── 5. Phase 2: Fine-tune full model (10 epochs) ─────────────
# Now that the head is calibrated, carefully unfreeze the backbone.
# The head's gradients will be reasonable (not random), so they won't
# destroy the backbone's pretrained features during backpropagation.
for param in model.parameters():
    param.requires_grad = True   # Unfreeze backbone — all parameters now train

# Differential LRs: backbone gets 1e-5 (careful nudges),
# head gets 1e-3 (still learning). See section 5 for the full explanation.
optimizer = torch.optim.AdamW([
    {"params": list(model.parameters())[:-4], "lr": 1e-5},  # Backbone: very low LR
    {"params": list(model.parameters())[-4:], "lr": 1e-3},  # Head: higher LR
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        criterion(model(X), y).backward()
        # Gradient clipping: fine-tuning can sometimes produce larger gradients
        # because both backbone and head are updating simultaneously
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in val_loader:
            correct += (model(X.to(device)).argmax(1) == y.to(device)).sum().item()
    val_acc = correct / len(val_data)
    print(f"Phase 2 Epoch {epoch+1}: val_acc={val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_transfer.pth")

print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
```

### Why two-phase fine-tuning is common

- phase 1 trains the new head quickly so the classifier adapts to your label space
- phase 2 unfreezes more of the network with a smaller learning rate
- lower LR on the backbone protects pretrained features from being destroyed too quickly

---

## 8. Interview Questions

<details>
<summary><strong>Q1: What is transfer learning and why is it effective?</strong></summary>

Transfer learning uses a model pretrained on a large dataset (e.g., ImageNet) as a starting point for a different task. It's effective because: (1) early CNN layers learn universal features (edges, textures) that are useful for any image task; (2) requires far less labeled data (100s vs millions); (3) trains much faster (minutes vs weeks); (4) acts as strong regularization — starting from a good local minimum near the optimum.
</details>

<details>
<summary><strong>Q2: Why must you use ImageNet normalization for pretrained models?</strong></summary>

The pretrained model's weights are calibrated for inputs normalized with `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`. The internal weight distributions, BatchNorm statistics, and decision boundaries are all computed assuming this input scale. Using different normalization shifts the distribution the model expects, causing poor feature extraction and unstable fine-tuning.
</details>

<details>
<summary><strong>Q3: What is catastrophic forgetting?</strong></summary>

When fine-tuning all layers simultaneously with a high learning rate, the model's pretrained weights are completely overwritten by gradients from your new task. The model "forgets" the rich ImageNet features it learned. Fixes: (1) use very low learning rate for backbone (1e-5); (2) gradual unfreezing — train head first, then unfreeze backbone layers one by one; (3) layer-wise learning rate decay.
</details>

<details>
<summary><strong>Q4: What is differential learning rate and when to use it?</strong></summary>

Using different learning rates for different parts of the model. Early layers (already have good general features): very low LR (1e-5 or 1e-6). Later backbone layers: medium LR (1e-4). New classification head: high LR (1e-3). Reason: early layers need minimal updates (universal features), the new head needs to learn from scratch (needs large LR). In PyTorch, pass a list of `{"params": ..., "lr": ...}` dicts to the optimizer.
</details>

---

## 🔗 References
- [torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [CampusX Video 12](https://www.youtube.com/watch?v=aPu6a5htRXM)
