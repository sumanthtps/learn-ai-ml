---
id: transfer-learning-cnn
title: "Transfer learning: feature extraction vs fine-tuning"
sidebar_label: "53 · Transfer Learning"
sidebar_position: 53
slug: /theory/dnn/transfer-learning-feature-extraction-vs-fine-tuning
description: "How to adapt pretrained CNNs to new tasks: feature extraction (freeze backbone, train head) vs fine-tuning (train all layers), with guidance on when to use each."
tags: [transfer-learning, fine-tuning, feature-extraction, pretrained, cnn, deep-learning]
---

# Transfer learning: feature extraction vs fine-tuning

Training a deep CNN from scratch requires millions of labeled images, weeks of GPU time, and careful hyperparameter tuning. Transfer learning bypasses this by starting from a pretrained model — typically trained on ImageNet — and adapting it to a new task. The key decision is how much to adapt: feature extraction (freeze the backbone, train only the head) or fine-tuning (unfreeze and retrain all layers).

## One-line definition

Transfer learning applies knowledge from a pretrained model (source domain) to a new task (target domain), either by using frozen pretrained features as-is or by continuing to train the entire model on new data.

![Transfer learning — step 1: unsupervised pre-training on a large dataset; step 2: fine-tuning on a specific downstream task](https://jalammar.github.io/images/bert-transfer-learning.png)
*Source: [Jay Alammar — The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) — the same pattern applies to CNN transfer learning*

## The two strategies

### Feature extraction

Freeze all pretrained layers. Add a new classification head. Train only the head.

```
Pretrained backbone (frozen) → [new head] → task output
    ↑ weights frozen                ↑ weights trained
```

**When to use:**
- Small target dataset (< 1000 images per class)
- Target task is similar to ImageNet (natural images)
- Limited compute budget

**Why it works:** The backbone encodes generic visual features (edges, textures, object parts) that transfer to most image tasks. Freezing it prevents overfitting the rich pretrained representations on a tiny dataset.

### Fine-tuning

Unfreeze some or all pretrained layers. Continue training with a small learning rate.

```
Pretrained backbone (unfrozen) → [new head]
    ↑ all weights updated (small lr)
```

**When to use:**
- Larger target dataset (> 5000 images per class)
- Target task differs significantly from ImageNet (medical images, satellite imagery)
- Need maximum accuracy

**Why it works:** Fine-tuning adjusts the pretrained features to better match the target domain. The small learning rate is critical — large updates would destroy the pretrained knowledge.

## Decision framework

| Dataset size | Similarity to ImageNet | Strategy |
|---|---|---|
| Small (< 1K/class) | Similar (natural images) | Feature extraction: freeze all, train head |
| Small (< 1K/class) | Different (X-rays, satellite) | Feature extraction: freeze early layers, fine-tune late layers |
| Medium (1K–10K/class) | Similar | Fine-tune last 1–2 blocks + head |
| Medium (1K–10K/class) | Different | Fine-tune entire network with small lr |
| Large (> 10K/class) | Any | Full fine-tuning or even train from scratch |

**Rule of thumb**: more data + more domain shift → more unfreezing.

## Progressive unfreezing

A robust strategy: start frozen, gradually unfreeze from the top (near the head) toward the bottom (near the input), progressively lowering the learning rate for older layers.

```
Step 1: Train head only (lr=1e-3)
Step 2: Unfreeze last block, train (lr=1e-4)
Step 3: Unfreeze last 2 blocks (lr=1e-5)
Step 4: Unfreeze all (lr=1e-6)
```

This is the "discriminative fine-tuning" approach from ULMFiT, adapted to CNNs.

## PyTorch implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader


# ============================================================
# Strategy 1: Feature extraction — freeze backbone, train head
# ============================================================
def build_feature_extractor(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze ALL pretrained parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final FC layer — always trainable
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Feature Extraction] Trainable: {trainable:,} / {total:,} params")
    return model


# ============================================================
# Strategy 2: Fine-tuning — different LR for different layers
# ============================================================
def build_fine_tuned_model(num_classes: int) -> tuple[nn.Module, list]:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Layer groups for different learning rates
    param_groups = [
        # Earlier layers: smaller LR (preserve general features)
        {"params": list(model.layer1.parameters()) +
                   list(model.layer2.parameters()),
         "lr": 1e-5},
        # Later layers: medium LR (adapt task-specific features)
        {"params": list(model.layer3.parameters()) +
                   list(model.layer4.parameters()),
         "lr": 1e-4},
        # New head: largest LR (train from scratch)
        {"params": model.fc.parameters(),
         "lr": 1e-3},
    ]
    return model, param_groups


# ============================================================
# Strategy 3: Progressive unfreezing
# ============================================================
class ProgressiveFinetuner:
    """
    Gradually unfreeze ResNet-50 layers during training.
    Call .unfreeze_next() each few epochs.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.stages = [
            [model.fc],
            [model.layer4],
            [model.layer3],
            [model.layer2],
            [model.layer1, model.conv1, model.bn1],
        ]
        self.current_stage = 0
        # Start fully frozen
        for param in model.parameters():
            param.requires_grad = False
        # Always unfreeze the head
        for param in model.fc.parameters():
            param.requires_grad = True

    def unfreeze_next(self):
        """Unfreeze the next layer group (call every N epochs)."""
        if self.current_stage < len(self.stages):
            for module in self.stages[self.current_stage]:
                for param in module.parameters():
                    param.requires_grad = True
            print(f"Unfrozen stage {self.current_stage}: "
                  f"{[m.__class__.__name__ for m in self.stages[self.current_stage]]}")
            self.current_stage += 1

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ============================================================
# Complete fine-tuning training loop
# ============================================================
def fine_tune(num_classes: int, data_dir: str,
              strategy: str = "feature_extraction",
              num_epochs: int = 20, batch_size: int = 32):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Datasets (ImageFolder expects: data_dir/class_name/img.jpg)
    from torchvision.datasets import ImageFolder
    train_ds = ImageFolder(f"{data_dir}/train", transform=train_tf)
    val_ds = ImageFolder(f"{data_dir}/val", transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if strategy == "feature_extraction":
        model = build_feature_extractor(num_classes).to(device)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3, weight_decay=1e-4,
        )
    elif strategy == "fine_tuning":
        model, param_groups = build_fine_tuned_model(num_classes)
        model = model.to(device)
        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_correct += (model(images).argmax(1) == labels).sum().item()

        # Validate
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                val_correct += (model(images).argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_ds)
        scheduler.step()

        print(f"Epoch {epoch:2d}: val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_transfer.pt")

    return model


# ============================================================
# Loading a fine-tuned model for inference
# ============================================================
def load_finetuned(checkpoint_path: str, num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)   # no pretrained weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model
```

## What happens to the features during fine-tuning

| Layer | Feature extraction | Fine-tuning |
|---|---|---|
| Layer 1 (edges) | Frozen — unchanged | Minimal update (very small LR) |
| Layer 2 (textures) | Frozen | Small update |
| Layer 3 (parts) | Frozen | Moderate update |
| Layer 4 (semantics) | Frozen | Larger update — adapts to new classes |
| FC head | Trained from scratch | Trained from scratch |

In feature extraction, the backbone is a fixed feature function. In fine-tuning, even early layers shift slightly, but the magnitude of update decreases with depth — this is why smaller LRs for early layers is correct.

## Learning rate schedule for fine-tuning

A cosine annealing schedule that starts from a warm-up:

```
LR
1e-3 |      /\
     |     /  \
     |    /    \______
1e-4 |   /
     |__/
     0    warmup  cosine decay   end
```

The warmup prevents unstable updates when first training the new head from random init. The cosine decay finds a smooth minimum rather than oscillating at the bottom.

## Interview questions

<details>
<summary>Why does fine-tuning with a large learning rate destroy pretrained knowledge?</summary>

Pretrained weights represent a highly non-trivial point in weight space found after training on millions of images. A large gradient update can move weights far from this point — into a region that works well for the new task (low training loss) but loses the carefully learned low-level features. This is catastrophic forgetting: the model quickly adapts to new data but forgets the general representations. A small learning rate allows gradual adjustment: the features shift from "generic ImageNet" toward "generic + task-specific" rather than abandoning the generic features entirely.
</details>

<details>
<summary>When would you use feature extraction over fine-tuning?</summary>

Feature extraction is better when: (1) the target dataset is very small (< 1000 images per class) — fine-tuning would overfit; (2) the target domain is similar to ImageNet — pretrained features transfer directly; (3) you have limited compute — training only the final layer is much faster. Fine-tuning is better when: (1) the target domain differs from ImageNet (e.g., medical imaging, satellite imagery) — the pretrained features need to adapt; (2) you have a sufficiently large target dataset to avoid overfitting.
</details>

<details>
<summary>What is label smoothing and why is it often used with fine-tuning?</summary>

Label smoothing replaces the hard one-hot target $y = [0, 0, 1, 0, ...]$ with a softened version: $\tilde{y} = [(1 - \epsilon)/K, ..., (1 - \epsilon + \epsilon), ...]$ where $\epsilon$ is a small value (e.g., 0.1). The loss penalizes the model for being too confident — it cannot push the class logit to infinity without being penalized for the soft probability mass on other classes. During fine-tuning, label smoothing prevents the new head from collapsing to overconfident predictions on the small training set, acting as a regularizer on the output distribution.
</details>

## Common mistakes

- Using the same LR for all layers during fine-tuning — early layers need smaller LR than the head
- Fine-tuning with `Adam` without `weight_decay` — AdamW with `weight_decay=1e-4` is the correct default
- Forgetting `model.train()` / `model.eval()` transitions — batch norm behaves differently; when the backbone is frozen and `model.eval()` is called, frozen BN layers use their population statistics, which is correct

## Final takeaway

Transfer learning with pretrained CNNs is the standard approach for almost every computer vision task. Feature extraction (frozen backbone, trained head) is the right choice for small datasets and similar domains. Fine-tuning (all layers trained with differential learning rates) is the right choice for larger datasets or different domains. In both cases, the ImageNet pretrained weights provide a starting point that is vastly better than random initialization.
