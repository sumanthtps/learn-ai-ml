---
id: 06-dataset-and-dataloader
title: "Video 6: Dataset & DataLoader Class in PyTorch"
sidebar_label: "06 · Dataset & DataLoader"
sidebar_position: 6
description: Building scalable data pipelines with PyTorch's Dataset and DataLoader — custom datasets, transforms, sampling, and performance tips.
tags: [pytorch, dataset, dataloader, transforms, campusx, data-pipeline]
---

# Dataset & DataLoader Class in PyTorch
**📺 CampusX — Practical Deep Learning using PyTorch | Video 6**

> **What you'll learn:** How to feed data efficiently to your model using PyTorch's `Dataset` and `DataLoader` abstractions — from simple in-memory data to large disk-based image datasets.

---

## 1. Why Not Just Load All Data at Once?

So far, the examples used small tensors loaded entirely into memory. Real deep learning datasets are orders of magnitude larger:

```python
# ❌ Naive approach — loads ALL data at once
X = load_all_data()   # What if it's 200GB of images?
model(X)              # Won't fit in RAM or GPU memory
```

Real datasets are enormous: ImageNet is 150 GB, Common Crawl is terabytes, medical imaging datasets can be hundreds of gigabytes. We need a smarter approach that:
- Loads data **on-demand in small batches** (only the current batch occupies memory)
- **Shuffles** each epoch so the model sees samples in a different order every time
- Applies **transforms/augmentations** on-the-fly without pre-storing all variants
- Loads data **in parallel** using multiple CPU workers while the GPU trains on the previous batch

PyTorch solves this elegantly with two complementary classes: `Dataset` and `DataLoader`.

### Why this matters for gradient descent

One important point: `Dataset` and `DataLoader` are what make the jump from **batch gradient descent** to **mini-batch gradient descent** practical in code.

```text
Full batch gradient descent:
  One forward/backward pass over the ENTIRE dataset → then one weight update
  Problem: impossible if dataset is larger than GPU memory

Mini-batch gradient descent:
  Split data into smaller batches (e.g., 32 images)
  Forward/backward pass + weight update for each batch
  Typical choice in practice: batch sizes of 32, 64, 128, 256
```

Mini-batch training is the standard because it:
- Fits in memory regardless of dataset size
- Updates weights more frequently (every batch instead of every epoch)
- Benefits from GPU parallelism (matrix operations on 32+ samples at once)
- Adds beneficial gradient noise that helps escape local minima

## 1.1 The Data Pipeline Mental Model

```text
Dataset    → knows how to get one sample at a given index
DataLoader → wraps Dataset, creates mini-batches, shuffles, runs workers
GPU        → consumes the next batch while workers prepare the one after that
```

In practice, bad model performance is sometimes a **data pipeline problem**, not a model problem. Slow loading, broken labels, missing transforms, and class imbalance often cause issues before the neural network is ever the bottleneck. Understanding `Dataset` and `DataLoader` deeply helps debug both.

## Visual Reference

![FashionMNIST dataset samples loaded via PyTorch Dataset](https://pytorch.org/docs/stable/_images/sphx_glr_data_tutorial_001.png)

*Nine sample images from the FashionMNIST dataset loaded through a PyTorch `Dataset`. Each call to `dataset[idx]` returns one `(image_tensor, label)` pair; `DataLoader` wraps this into shuffled mini-batches, loading them in parallel across multiple CPU workers while the GPU trains on the previous batch.*

---

## 2. The Dataset Class

A `Dataset` answers exactly one question: **"Give me sample number `idx`."** Everything else — batching, shuffling, parallelism — is handled by `DataLoader`.

There are two kinds of datasets:
- **Map-style**: you can access any sample by index (`dataset[5]`) — this is what you'll use 95% of the time
- **Iterable-style**: you consume samples one by one (useful for streaming large datasets from disk/network)

### 2.1 Map-Style Dataset — The Three Required Methods

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, ...):
        # Load metadata, file paths, labels — anything you need to FIND samples.
        # Do NOT load all images/data here — just enough info to load one on demand.
        # Heavy loading belongs in __getitem__.
        pass

    def __len__(self):
        # Return the total number of samples in the dataset.
        # DataLoader uses this to know when one epoch is complete.
        return ...

    def __getitem__(self, idx):
        # Return ONE sample: typically a (features, label) tuple.
        # DataLoader will call this for each index in a batch,
        # then stack the results into a batch tensor.
        return ...
```

### 2.2 Simple In-Memory Dataset

When your data fits entirely in RAM (e.g., a preprocessed CSV with thousands of rows), the simplest approach is to convert it to tensors once in `__init__`:

```python
import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    """Dataset for CSV/tabular data that fits comfortably in memory."""

    def __init__(self, X, y):
        # Convert to tensors ONCE in __init__ — not in __getitem__!
        # If you converted in __getitem__, you'd re-convert for every sample,
        # every epoch — extremely wasteful.
        self.X = torch.tensor(X, dtype=torch.float32)   # Features: float for NN math
        self.y = torch.tensor(y, dtype=torch.long)      # Labels: long for loss fns

    def __len__(self):
        return len(self.X)   # Total number of samples

    def __getitem__(self, idx):
        # Return exactly one (features, label) pair
        # DataLoader will call this for each index in a batch and stack results
        return self.X[idx], self.y[idx]

# Usage
import numpy as np
X = np.random.randn(1000, 20)      # 1000 samples, 20 features
y = np.random.randint(0, 5, 1000)  # 5 class labels
dataset = TabularDataset(X, y)

print(len(dataset))      # 1000
print(dataset[0])        # (tensor of shape [20], tensor with single label)
print(dataset[0][0].shape)  # torch.Size([20])
```

### 2.3 Image Dataset from Disk

When images don't fit in memory, load each one on demand in `__getitem__`. The key design: store only the file paths in `__init__`, load the actual image only when requested:

```python
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class ImageDataset(Dataset):
    """Load images from disk using a CSV mapping filename → label."""

    def __init__(self, csv_path, img_dir, transform=None):
        # Load only the CSV metadata — file paths and labels — not the actual images
        self.df      = pd.read_csv(csv_path)   # columns: filename, label
        self.img_dir = img_dir
        self.transform = transform              # Transform to apply to each image

        # Convert string labels to integers once
        self.classes       = sorted(self.df['label'].unique())
        self.class_to_idx  = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)   # One row per sample

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])

        # Load image from disk — this is the expensive step
        # Always convert to RGB! Some images are grayscale or RGBA by default
        image = Image.open(img_path).convert('RGB')

        # Convert string label to integer
        label = self.class_to_idx[row['label']]

        # Apply transforms (resize, normalize, augment) if provided
        if self.transform:
            image = self.transform(image)   # Returns a tensor after transforms

        return image, torch.tensor(label, dtype=torch.long)
```

### 2.4 Using torchvision's Built-in Datasets

For standard benchmark datasets (MNIST, CIFAR-10, ImageNet), torchvision provides ready-made Dataset classes that download and cache the data automatically:

```python
from torchvision import datasets, transforms

# Define the transform: convert PIL image to tensor, then normalize
transform = transforms.Compose([
    transforms.ToTensor(),                          # PIL (H,W,C) uint8 → tensor (C,H,W) float in [0,1]
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # Normalize to roughly [-1, 1]
])

# Downloads to root/ if not already present, returns a Dataset object
train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Other popular built-in datasets:
# datasets.CIFAR10      — 60K images, 10 classes, 32×32 color
# datasets.CIFAR100     — 60K images, 100 classes
# datasets.FashionMNIST — 70K fashion item images, 10 classes
# datasets.ImageNet     — 1.2M images, 1000 classes (requires manual download)

# ImageFolder: for custom datasets organized as folder-per-class
# data/
#   train/
#     cat/  img001.jpg img002.jpg ...
#     dog/  img001.jpg img002.jpg ...
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
print(train_dataset.classes)         # ['cat', 'dog']
print(train_dataset.class_to_idx)    # {'cat': 0, 'dog': 1}
```

---

## 3. TensorDataset — Simplest Dataset

When you already have tensors and just want to wrap them for DataLoader, `TensorDataset` is the simplest option — no custom class needed:

```python
from torch.utils.data import TensorDataset

X = torch.rand(1000, 20)           # Features tensor
y = torch.randint(0, 5, (1000,))   # Labels tensor

# TensorDataset zips the tensors: dataset[i] returns (X[i], y[i])
dataset = TensorDataset(X, y)
print(dataset[0])        # (X[0], y[0])
print(dataset[42])       # (X[42], y[42])

# Then use as normal with DataLoader
loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

`TensorDataset` is ideal for prototype experiments and the earlier video examples. For production datasets that come from disk, use a custom `Dataset` class.

---

## 4. The DataLoader

`DataLoader` wraps a `Dataset` and handles all the logistics: batching, shuffling, parallel loading, and prefetching:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,          # How many samples per batch
    shuffle=True,           # Shuffle at each epoch start (True for train, False for val/test)
    num_workers=4,          # Subprocesses for parallel data loading (0 = main process)
    pin_memory=True,        # Allocate in page-locked memory for faster CPU→GPU transfer
    drop_last=False,        # Drop last batch if it's smaller than batch_size
    prefetch_factor=2,      # How many batches to prefetch per worker (requires num_workers > 0)
    persistent_workers=True # Keep workers alive between epochs (avoids startup cost)
)

# Iterate: DataLoader calls __getitem__ for each index in a batch, stacks results
for X_batch, y_batch in loader:
    print(X_batch.shape)   # (32, ...) — 32 stacked samples
    print(y_batch.shape)   # (32,)

# Get just one batch for debugging (without iterating through all)
X, y = next(iter(loader))
print(X.shape, y.shape)

# Number of batches per epoch
print(len(loader))    # ceil(len(dataset) / batch_size)
```

### num_workers Guidelines

`num_workers` controls the number of parallel data loading processes:

```python
# num_workers = 0: Load in main process (single-threaded, synchronous)
# → GPU waits for CPU to load each batch before it can train
# → Simple, safe, best for in-memory data or debugging

# num_workers > 0: Load asynchronously in separate subprocesses
# → CPU loads next batch while GPU trains on current batch
# → Eliminates data loading bottleneck for large disk-based datasets

# Rules of thumb:
# SSD storage:       4–8 workers
# HDD storage:       2–4 workers (disk I/O is the bottleneck anyway)
# In-memory data:    0 (spawning processes costs more than the benefit)
# Windows platform:  must wrap training code in if __name__ == '__main__': guard
# Jupyter notebook:  start with 0 or 2 (can cause issues with higher values)
```

---

## 5. Transforms (Image Preprocessing & Augmentation)

Transforms convert raw data (PIL images, NumPy arrays) into model-ready tensors and optionally apply random augmentations. The key rule: **augmentation is applied only to training data, never to validation or test data.**

```python
from torchvision import transforms

# ── TRAINING transforms: augmentation + normalization ───────────
# Random augmentations are applied EVERY TIME a sample is loaded
# This means the model sees a slightly different version each epoch → regularization
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),       # Random crop at random scale + resize to 224×224
    transforms.RandomHorizontalFlip(p=0.5), # 50% chance of flipping left-right
    transforms.RandomVerticalFlip(p=0.1),   # 10% chance of flipping up-down
    transforms.RandomRotation(degrees=15),   # Rotate randomly within ±15 degrees
    transforms.ColorJitter(                  # Random color changes (simulates lighting conditions)
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.1
    ),
    transforms.RandomGrayscale(p=0.05),     # 5% chance of converting to grayscale
    transforms.RandomErasing(p=0.1),        # 10% chance of randomly masking a patch

    # Required: convert PIL image to tensor
    # Also scales pixel values from [0, 255] to [0.0, 1.0]
    transforms.ToTensor(),

    # Normalize with ImageNet mean/std — REQUIRED when using pretrained models
    # These specific values were computed from the ImageNet training set
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── VALIDATION/TEST transforms: only normalization ──────────────
# No random augmentations! We want consistent, reproducible predictions.
# The same image should produce the same tensor every time during evaluation.
val_transform = transforms.Compose([
    transforms.Resize(256),              # Resize shorter edge to 256
    transforms.CenterCrop(224),          # Take center 224×224 crop
    transforms.ToTensor(),               # PIL → tensor, scale to [0, 1]
    transforms.Normalize(               # Same normalization as training
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Apply to datasets
train_ds = datasets.ImageFolder('data/train', transform=train_transform)
val_ds   = datasets.ImageFolder('data/val',   transform=val_transform)
```

:::tip
Augmentation makes each epoch show slightly different versions of training images. This is the regularization effect — the model can't memorize exact pixel values, so it must learn more general features. Always apply augmentation **only to training data**, never to validation or test data.
:::

---

## 6. Dataset Splitting

You need separate splits for training, validation, and testing. The split must happen **before any preprocessing** to prevent data leakage:

```python
from torch.utils.data import random_split, Subset
import torch

# ── Simple random split ───────────────────────────────────────
dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
n_total = len(dataset)    # 60000 samples
n_train = int(0.8 * n_total)
n_val   = n_total - n_train   # 20%

# random_split: randomly assigns indices to each split
# With a fixed generator: reproducible across runs
train_data, val_data = random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)  # ← same result every run
)

# ── Subset: use explicit indices ──────────────────────────────
# Useful when you need specific samples (e.g., time-ordered splits)
train_idx = list(range(0, 50000))
val_idx   = list(range(50000, 60000))
train_data = Subset(dataset, train_idx)
val_data   = Subset(dataset, val_idx)

# ── Stratified split: preserves class distribution ────────────
# random_split doesn't guarantee equal class proportions — use sklearn for this
from sklearn.model_selection import train_test_split
labels    = [dataset[i][1] for i in range(len(dataset))]   # Collect all labels
train_idx, val_idx = train_test_split(
    range(len(dataset)),
    test_size=0.2,
    stratify=labels,        # ← Ensures each class gets 80/20 split
    random_state=42
)
train_data = Subset(dataset, train_idx)
val_data   = Subset(dataset, val_idx)
```

### The right order for preprocessing (preventing data leakage)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Split FIRST — before touching any data values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Fit preprocessing on TRAINING data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # Fit (compute mean/std) on train, then transform
X_test  = scaler.transform(X_test)         # Transform test using TRAIN's mean/std

# 3. Encode labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test  = encoder.transform(y_test)

# Why this order? If you scaled the entire dataset before splitting:
# - The scaler sees test data's statistics when computing mean/std
# - This is information leakage: the model indirectly "sees" test data during training
# - Reported test accuracy will be optimistically biased
```

---

## 7. Handling Class Imbalance with WeightedRandomSampler

Before using `WeightedRandomSampler`, we need one extra PyTorch idea:

### What is a sampler?

When you create a `DataLoader`, PyTorch has to answer this question:

> "In what order should I pick dataset items?"

That ordering logic is handled by a **sampler**.

- If you use `shuffle=False`, the loader usually reads samples in order: `0, 1, 2, 3, ...`
- If you use `shuffle=True`, the loader internally uses random ordering
- If you pass `sampler=...`, you are telling PyTorch: "Don't decide the order yourself. Use this custom rule instead."

So a sampler does **not** change your dataset's contents. It changes **which indices are drawn, and how often**.

That matters for **class imbalance**. Suppose your dataset has:

- 900 cat images
- 90 dog images
- 10 bird images

If you sample uniformly from the dataset, most mini-batches will be dominated by cats. The model will keep seeing cats over and over, and may learn a lazy strategy like "predict cat for everything."

`WeightedRandomSampler` fixes this by giving each sample a sampling probability. Rare-class samples get larger probability, so they appear more often during training.

### Important intuition

- The weights are assigned **per sample**, not per class
- Higher weight means "pick this sample more often"
- This changes the training distribution, but it does **not** create new data
- It is mainly a training-time trick to make learning more balanced

Now the code will make more sense:

```python
from torch.utils.data import WeightedRandomSampler

# Example: 900 cats, 90 dogs, 10 birds
class_counts = [900, 90, 10]

# Weight per class = inverse of frequency
# Rare classes get higher probability of being sampled
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
# class_weights: [0.0011, 0.011, 0.1]  — birds are 90× more likely than cats

# Assign each SAMPLE a weight based on its class
labels         = [0]*900 + [1]*90 + [2]*10   # All sample labels
sample_weights = class_weights[torch.tensor(labels)]

# Create sampler: randomly selects samples according to their weights
sampler = WeightedRandomSampler(
    weights=sample_weights,     # Per-sample sampling probability
    num_samples=len(sample_weights),  # Sample as many samples as the dataset size
    replacement=True            # With replacement: can sample same sample multiple times
)

# Use sampler — mutually exclusive with shuffle=True!
# The sampler IS the shuffling mechanism for imbalanced datasets
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
# Each batch will have roughly equal class representation ✅
```

### Why `replacement=True`?

This is another detail that is easy to gloss over.

With replacement means after a sample is drawn, it can be drawn again later in the same epoch. That is useful here because the minority class may have very few examples.

Example:

- only 10 bird images exist
- but you want birds to appear much more often during training

Without replacement, each bird image could be used only once per epoch, which limits oversampling. With replacement, PyTorch can re-draw those rare examples multiple times.

### When should you use a sampler?

Use a sampler when the default "just shuffle the dataset" behavior is not enough.

Common cases:

- class imbalance: `WeightedRandomSampler`
- distributed training: `DistributedSampler`
- special ordering rules: curriculum learning, grouped batches, bucketing by sequence length

If your dataset is balanced and ordinary, `shuffle=True` is usually enough. You only reach for a sampler when you need explicit control over how samples are chosen.

---

## 8. Custom Collate Function

There is one more hidden step inside `DataLoader` that beginners often do not see at first.

### What is `collate_fn`?

Your dataset returns **one sample at a time**.

For example, `dataset[7]` might return:

```python
(text_tensor, label)
```

But training does not happen one sample at a time. We usually want a mini-batch like:

```python
(batch_of_texts, batch_of_labels)
```

So after `DataLoader` fetches, say, 32 individual samples, it must combine them into one batch. That combining step is called **collation**, and the function that performs it is `collate_fn`.

In plain English:

- `Dataset.__getitem__` says: "Here is one sample."
- `collate_fn` says: "Here is how to merge many samples into one batch."

### What does the default collate function do?

PyTorch provides a default collate function automatically. It works well when every sample has the same shape.

Example:

- image 1 shape: `(3, 32, 32)`
- image 2 shape: `(3, 32, 32)`
- image 3 shape: `(3, 32, 32)`

Since the shapes match, PyTorch can stack them into a tensor of shape:

```python
(batch_size, 3, 32, 32)
```

### Why does the default collate fail for text?

Text sequences often have different lengths.

Example batch:

- `"hi"` -> `[5, 9]`
- `"hello"` -> `[5, 2, 12, 12, 15]`
- `"hey"` -> `[5, 2, 25]`

These tensors have lengths 2, 5, and 3. PyTorch cannot stack them directly into one rectangular tensor, because tensors in a batch must have the same size along each dimension.

That is why we write a custom `collate_fn`: it tells PyTorch how to make unequal-length samples batchable. The most common strategy is **padding** the shorter sequences to match the longest sequence in that batch.

### Mental model

Think of `collate_fn` as the "batch assembly" step:

1. DataLoader asks the dataset for sample A
2. DataLoader asks the dataset for sample B
3. DataLoader asks the dataset for sample C
4. `collate_fn` merges A, B, C into one batch object

For variable-length sequences (NLP tasks), you need custom collation that pads sequences to the same length within each batch:

```python
from torch.nn.utils.rnn import pad_sequence

def text_collate_fn(batch):
    """Collate function for variable-length text sequences."""
    # batch is a list of (text_tensor, label) tuples
    texts, labels = zip(*batch)   # Separate texts and labels into two tuples

    # texts is a tuple of 1D tensors with DIFFERENT lengths
    # Pad all sequences to the length of the longest one in this batch
    padded = pad_sequence(
        texts,
        batch_first=True,     # Output shape: (batch, max_seq_len)
        padding_value=0        # Use 0 as the padding token index
    )

    # Store original lengths (needed for packed sequences in RNNs)
    lengths = torch.tensor([len(t) for t in texts])
    labels  = torch.tensor(labels, dtype=torch.long)

    return padded, lengths, labels

# Pass custom collate function to DataLoader
loader = DataLoader(dataset, batch_size=32, collate_fn=text_collate_fn)
```

### Why return `lengths` too?

After padding, all sequences have the same visible length, but that length is partly fake because padding tokens were added.

Example:

- original lengths: `[2, 5, 3]`
- padded batch shape: `(3, 5)`

The model often needs the **true** sequence lengths so it can:

- ignore padding when computing sequence summaries
- use packed sequences in RNN/LSTM models
- build masks for attention-based models

So `collate_fn` often returns more than just `(inputs, labels)`. It can also return metadata such as original lengths, masks, IDs, or filenames.

### Rule of thumb

- fixed-size images or tensors: default collate is usually enough
- variable-length text/audio/sequences: custom `collate_fn` is often required
- weird nested sample structures: custom `collate_fn` may also be needed

---

## 9. Full Production Pipeline Example

Putting it all together: a complete data pipeline for CIFAR-10 classification:

```python
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ── 1. Define transforms ──────────────────────────────────────
# CIFAR-10 mean and std (computed from the training set)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)

train_tfm = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # Pad by 4 then take random 32×32 crop
    transforms.RandomHorizontalFlip(),          # 50% chance of horizontal flip
    transforms.ToTensor(),                      # PIL → float tensor in [0, 1]
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD) # Normalize with dataset-specific stats
])

val_tfm = transforms.Compose([
    transforms.ToTensor(),                       # No augmentation — just convert
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)  # Same normalization as training
])

# ── 2. Load datasets ──────────────────────────────────────────
# CIFAR-10 provides a standard train/test split
# We further split the training set into train + validation
full_train = datasets.CIFAR10('./data', train=True,  download=True, transform=train_tfm)
test_ds    = datasets.CIFAR10('./data', train=False, download=True, transform=val_tfm)

# Hold out 5000 samples from training for validation
# Using a fixed seed ensures the same 5000 samples are always the validation set
train_ds, val_ds = random_split(
    full_train, [45000, 5000],
    generator=torch.Generator().manual_seed(42)
)

# ── 3. Create DataLoaders ─────────────────────────────────────
train_loader = DataLoader(
    train_ds,
    batch_size=128,           # Larger batch = more GPU utilization
    shuffle=True,             # Randomize order every epoch
    num_workers=4,            # 4 parallel loading processes
    pin_memory=True,          # Page-lock memory for faster GPU transfer
    persistent_workers=True   # Keep workers alive between epochs (saves startup cost)
)

val_loader = DataLoader(
    val_ds,
    batch_size=256,           # Larger batch for eval (no gradients needed)
    shuffle=False,            # Never shuffle validation — consistent results
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} | Test: {len(test_loader)}")

# ── 4. Verify shapes — catch data bugs BEFORE training ───────
X, y = next(iter(train_loader))
print(f"X shape: {X.shape}, y shape: {y.shape}")
# Expected: X: torch.Size([128, 3, 32, 32]), y: torch.Size([128])
```

**Why this setup is production-quality:**
- Training and validation use different transforms (augmentation only for training)
- The split is reproducible via a seeded generator
- Training batches are shuffled; validation and test are not
- `pin_memory=True` enables faster CPU→GPU memory transfer
- `persistent_workers=True` avoids subprocess restart cost between epochs
- The shape check at the end catches dataset bugs before the first training epoch

---

## 10. Interview Questions

<details>
<summary><strong>Q1: What are the three methods required for a custom Dataset?</strong></summary>

For a map-style dataset: `__init__` (load metadata, set transforms, do NOT load actual data if it's large), `__len__` (return total number of samples — DataLoader uses this to know when one epoch is complete), `__getitem__` (return one `(input, label)` tuple for a given index — this is called per-sample for each batch). DataLoader calls `__getitem__` for each index in a batch, then calls the collate function to stack them into batch tensors.
</details>

<details>
<summary><strong>Q2: What does num_workers do? When should you set it to 0?</strong></summary>

`num_workers` controls how many subprocess workers load data in parallel. With `num_workers=0`, loading is synchronous in the main process — GPU waits idle while CPU loads each batch. With `num_workers=N`, N processes load asynchronously while the GPU trains on the previous batch, overlapping compute and I/O. Set to 0 when: data is tiny and fits in memory (process spawning overhead exceeds benefit), on Windows with multiprocessing issues, in Jupyter notebooks (can cause deadlocks), or when debugging data loading bugs.
</details>

<details>
<summary><strong>Q3: What is pin_memory and when should you use it?</strong></summary>

`pin_memory=True` allocates DataLoader output tensors in **page-locked (pinned) CPU memory** instead of pageable memory. Pinned memory can be directly DMA-transferred to GPU without an intermediate copy step, enabling asynchronous transfers. Combine with `non_blocking=True` in `.to(device, non_blocking=True)` for maximum throughput: GPU starts computing batch N while CPU simultaneously transfers batch N+1. Always use it when training on GPU with real datasets.
</details>

<details>
<summary><strong>Q4: Why is shuffle=True important for training but wrong for validation?</strong></summary>

Shuffling prevents the model from learning the **order** of data — if batches always appear in the same sequence, the model might unconsciously adapt to that order. Shuffling also ensures each batch contains samples from all classes, giving more representative gradient estimates. Validation doesn't shuffle because: (1) the order doesn't affect accuracy computation; (2) reproducibility — same order → same metric every run; (3) some evaluation patterns (like sliding window metrics) depend on the original order.
</details>

<details>
<summary><strong>Q5: What is WeightedRandomSampler and when do you need it?</strong></summary>

`WeightedRandomSampler` assigns a sampling probability to each sample. Rare classes get higher weights so they appear more often in batches. Used when your dataset has significant class imbalance (e.g., 95% negative, 5% positive in a disease detection task). Without it, the model sees mostly majority-class samples and learns to predict the majority class always — achieving high accuracy but poor recall on the rare class. Note: mutually exclusive with `shuffle=True` — the sampler replaces the shuffle mechanism.
</details>

---

## 🔗 References
- [DataLoader Docs](https://pytorch.org/docs/stable/data.html)
- [CampusX Video 6](https://www.youtube.com/watch?v=RH6DeE3bY6I)
