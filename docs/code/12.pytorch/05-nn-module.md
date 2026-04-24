---
id: 05-nn-module
title: "Video 5: PyTorch NN Module"
sidebar_label: "05 · NN Module"
sidebar_position: 5
description: Deep dive into torch.nn.Module — the base class for all neural networks in PyTorch.
tags: [pytorch, nn-module, layers, sequential, parameters, campusx]
---

# PyTorch NN Module
**📺 CampusX — Practical Deep Learning using PyTorch | Video 5**

> **What you'll learn:** How `torch.nn.Module` works under the hood — parameter registration, forward pass, state dict, and all built-in layers.

---

## 1. What is `nn.Module`?

In the previous video, you saw the manual training pipeline: weights and biases created by hand, forward function written from scratch, gradients zeroed manually, and parameters updated inside `torch.no_grad()`. That approach works, but it scales poorly. Every new model requires rewriting the same boilerplate.

`torch.nn.Module` solves this by being the **base class for every neural network component in PyTorch** — not just full models, but also individual layers, loss functions, and even non-trainable preprocessing steps. By subclassing `nn.Module`, your class automatically gets a large set of capabilities for free:

| Feature | What it gives you |
|---|---|
| Parameter registration | `model.parameters()` automatically discovers all trainable weights |
| Device migration | `.to(device)` moves ALL tensors (weights, buffers) to GPU in one call |
| State dict | Save/load weights with `.state_dict()` and `load_state_dict()` |
| Train/eval mode | `.train()` / `.eval()` toggles Dropout and BatchNorm behavior recursively |
| Hooks | Forward/backward hooks for debugging, feature extraction, gradient monitoring |
| Composability | Modules can contain other Modules, forming deep hierarchies |

### What `nn.Module` replaces from the manual pipeline

In the previous video, you did everything manually. `nn.Module` replaces those pieces cleanly:

| Manual pipeline piece | Cleaner `nn.Module` replacement |
|---|---|
| Manually created `weights = torch.randn(3, requires_grad=True)` | `nn.Linear(...)` creates and registers weights automatically |
| Manually applied `torch.sigmoid(...)` | `nn.Sigmoid()` or a fused loss function |
| Manually written loss formula | Built-in loss from `torch.nn` |
| Manually updated `weights -= lr * weights.grad` | Optimizer from `torch.optim` |

That is the real value: describe what the model computes at a high level, and PyTorch handles the infrastructure.

### What lives inside `torch.nn`

The `torch.nn` module is a bundle of practical building blocks for every deep learning task:

- **Layers**: `nn.Linear`, `nn.Conv2d`, `nn.RNN`, `nn.LSTM`, `nn.Transformer`
- **Activations**: `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.GELU`, `nn.Softmax`
- **Losses**: `nn.MSELoss`, `nn.BCEWithLogitsLoss`, `nn.CrossEntropyLoss`
- **Containers**: `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict`
- **Regularization**: `nn.Dropout`, `nn.BatchNorm1d`, `nn.LayerNorm`

`torch.optim` is the natural companion: `torch.nn` defines **what** the network is (its structure and computation), and `torch.optim` defines **how** its parameters get updated (the optimization algorithm).

## Visual Reference

![Three-hidden-layer multilayer perceptron](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/images/nn3.png)

*Every rectangle in this diagram corresponds to an `nn.Module`: each linear layer, each activation, and the whole network itself. Nesting modules inside modules is how PyTorch builds composable hierarchies — call `.to(device)` once on the root module and all parameters move; call `.parameters()` and every trainable weight is returned.*

---

## 2. The Minimum nn.Module

Every PyTorch model follows the same two-requirement structure: a class that inherits from `nn.Module`, with `__init__` to define layers and `forward` to define computation:

```python
import torch
import torch.nn as nn

class MyFirstModule(nn.Module):
    def __init__(self):
        # ALWAYS call super().__init__() first!
        # This initializes PyTorch's internal bookkeeping:
        # the _parameters, _modules, _buffers dicts that track everything
        super().__init__()

        # Assigning a layer as an attribute registers it automatically.
        # PyTorch sees "self.linear = nn.Linear(...)" and records it
        # in the internal _modules dict — so model.parameters() will find it.
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        # Define the computation that turns input x into output
        # This is the forward pass — autograd builds the graph here
        return self.linear(x)

# Create and use the model
model = MyFirstModule()
x     = torch.rand(8, 4)     # Batch of 8 samples, 4 features each
out   = model(x)             # Calls __call__ → forward() → returns output
print(out.shape)             # torch.Size([8, 2])
```

**Three critical points:**

1. `super().__init__()` — **never skip this.** It sets up the internal dicts that make everything else work. Without it, assigning layers won't register them.

2. `model(x)` vs `model.forward(x)` — always use `model(x)`. PyTorch's `__call__` method wraps `forward()` with hooks, gradient tracking setup, and other bookkeeping. Calling `forward()` directly skips all of that.

3. The batch dimension is preserved automatically: input `(8, 4)` → output `(8, 2)`. `nn.Linear` applies the same weight matrix to every row in the batch.

:::info Important
Never call `model.forward(x)` directly. Always call `model(x)`. PyTorch's `__call__` method wraps `forward()` and handles hooks, gradient tracking, and bookkeeping that are critical for correct behavior.
:::

---

## 3. Parameters and Buffers

Understanding the difference between parameters and buffers is essential for understanding what gets saved, what gets trained, and what gets moved to GPU.

### Parameters — learnable, updated by the optimizer

A **parameter** is a tensor that:
- Has `requires_grad=True` automatically
- Is registered in the module's `_parameters` dict
- Shows up in `model.parameters()` → the optimizer updates it
- Is saved and loaded with `state_dict()`
- Moves with `.to(device)`

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # nn.Parameter: marks a tensor as a learnable parameter
        # Equivalent to torch.tensor(x, requires_grad=True) + registration
        self.weight = nn.Parameter(torch.randn(4, 4))   # Registered as parameter
        self.bias   = nn.Parameter(torch.zeros(4))       # Registered as parameter

        # Sub-modules also contribute their parameters recursively
        # model.parameters() will discover linear.weight and linear.bias too
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        x = x @ self.weight + self.bias   # Use our custom parameter
        return self.linear(x)             # Then through the registered sub-module

model = MyModel()

# named_parameters() shows every parameter with its full name path
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")
# weight: shape=torch.Size([4, 4]), requires_grad=True
# bias:   shape=torch.Size([4]), requires_grad=True
# linear.weight: shape=torch.Size([2, 4]), requires_grad=True
# linear.bias:   shape=torch.Size([2]), requires_grad=True

# Count total trainable parameters — useful for model complexity assessment
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")
```

### Buffers — non-learnable tensors that are part of the model state

A **buffer** is a tensor that:
- Is NOT a learnable parameter (`requires_grad=False`)
- Is part of the model state (saved/loaded with `state_dict()`)
- Moves with `.to(device)` — crucial for keeping it on the same device as parameters

Buffers are used for running statistics, masks, or any persistent non-trainable state:

```python
class RunningMeanTracker(nn.Module):
    def __init__(self):
        super().__init__()
        # register_buffer: saved in state_dict and moved with .to(device),
        # but NOT included in model.parameters() → optimizer ignores it
        self.register_buffer('running_mean', torch.zeros(10))
        self.register_buffer('num_batches',  torch.tensor(0))

    def forward(self, x):
        # Update the running mean in-place during forward pass
        self.running_mean = (
            (self.running_mean * self.num_batches + x.mean(0))
            / (self.num_batches + 1)
        )
        self.num_batches += 1
        return x

# The most common use of buffers in practice: BatchNorm layers
# BatchNorm stores running_mean and running_var as buffers, not parameters
bn = nn.BatchNorm1d(10)
print(dict(bn.named_buffers()).keys())
# dict_keys(['running_mean', 'running_var', 'num_batches_tracked'])
# These are saved in state_dict() and move with .to(device),
# but the optimizer does not update them — BatchNorm updates them internally
```

---

## 4. Built-in Layers Reference

PyTorch provides a comprehensive set of layers for all common neural network operations. You build models by combining these as building blocks.

Before the layer list, here is the beginner-friendly way to read this section:

### How should you think about a layer?

A layer is just a reusable transformation.

It takes an input tensor, applies some rule, and returns an output tensor.

Different layers solve different jobs:

- `Linear`: mix input features
- activation: add non-linearity
- normalization: stabilize training
- dropout: regularize by adding noise during training

So when you read `nn.Linear(...)` or `nn.BatchNorm1d(...)`, do not think "new syntax to memorize." Think "new type of transformation in the network pipeline."

### Linear (Fully Connected)

The most fundamental layer: applies a linear transformation `y = x @ W.T + b`:

```python
# in_features: number of input neurons
# out_features: number of output neurons
# bias: whether to include a learnable bias term (default: True)
layer = nn.Linear(in_features=10, out_features=5, bias=True)

# Weight matrix shape: (out_features, in_features) — note: transposed relative to math
print(layer.weight.shape)  # (5, 10) — 5 output neurons, each connected to 10 inputs
print(layer.bias.shape)    # (5,)    — one bias per output neuron

x   = torch.rand(32, 10)   # 32 samples, each with 10 features
out = layer(x)             # (32, 5) — the batch dimension is preserved
```

### Activation Functions

Activations introduce non-linearity, allowing the network to learn curved decision boundaries:

### Why do we need activation functions at all?

Without activations, a stack of linear layers is still just one big linear transformation.

That means:

- no matter how many layers you add
- the model can only learn straight-line decision boundaries

Activations break that limitation by adding non-linearity, which is why deep networks can learn complex patterns.

```python
x = torch.tensor([-3., -1., 0., 1., 3.])

# ReLU: max(0, x) — most commonly used for hidden layers
# Fast, simple, avoids vanishing gradients for positive inputs
nn.ReLU()(x)           # [0, 0, 0, 1, 3]

# Leaky ReLU: small slope for negative values — prevents dead neurons
# Dead neuron: a ReLU neuron that always outputs 0 (gradient = 0, never learns)
nn.LeakyReLU(0.01)(x)  # [-0.03, -0.01, 0, 1, 3]

# ELU: Exponential Linear Unit — smooth transition into negative region
nn.ELU(alpha=1.0)(x)   # [~-0.95, ~-0.63, 0, 1, 3]

# Sigmoid: squashes to (0, 1) — use for binary output probability
# Note: NOT recommended for hidden layers (vanishing gradients)
nn.Sigmoid()(x)        # [0.047, 0.269, 0.5, 0.731, 0.952]

# Tanh: squashes to (-1, 1) — centered, used in RNNs
nn.Tanh()(x)           # [-0.995, -0.762, 0, 0.762, 0.995]

# GELU: smooth ReLU variant used in BERT, GPT, and most modern Transformers
nn.GELU()(x)

# Softmax: converts logits to probabilities that sum to 1
# Use ONLY at the output layer for multi-class classification
# (Don't use before CrossEntropyLoss — it applies this internally)
nn.Softmax(dim=-1)(x)

# Functional API: stateless activations (no learnable parameters)
# Preferred inside forward() to avoid creating unnecessary Module objects
import torch.nn.functional as F
F.relu(x)
F.sigmoid(x)
F.softmax(x, dim=-1)
F.gelu(x)
```

### Normalization Layers

Normalization layers stabilize training by keeping activations in reasonable ranges:

### What problem do normalization layers solve?

As data flows through many layers, the scale of activations can drift.

That can make training harder because:

- gradients may become unstable
- later layers keep seeing changing input distributions
- learning becomes more sensitive to initialization and learning rate

Normalization layers reduce that instability by rescaling activations in a more controlled way.

```python
# ── BatchNorm: normalize over the batch dimension ─────────────
# For each feature, normalize across all samples in the batch
# Learns scale (γ) and shift (β) parameters that can undo normalization if needed
nn.BatchNorm1d(num_features=64)   # For (N, C) or (N, C, L) tensors
nn.BatchNorm2d(num_features=64)   # For (N, C, H, W) images — normalizes per channel

x   = torch.rand(32, 64)   # 32 samples, 64 features
out = nn.BatchNorm1d(64)(x)
# Each of the 64 features is normalized: mean≈0, std≈1 across the 32 samples in the batch

# ── LayerNorm: normalize over the feature dimension of ONE sample ─
# Does NOT depend on batch size — same behavior in train and eval modes
# Essential for Transformers where batch statistics are meaningless (variable-length sequences)
nn.LayerNorm(normalized_shape=64)   # Normalize the last 64 dimensions

# ── GroupNorm: divide channels into groups, normalize within each group ─
# Compromise between BatchNorm (good for CNNs) and LayerNorm (good for sequences)
nn.GroupNorm(num_groups=8, num_channels=64)   # 64 channels split into 8 groups

# ── InstanceNorm: per sample, per channel — used in style transfer ─
nn.InstanceNorm2d(num_features=64)
```

### Dropout

Dropout randomly zeros a fraction of neurons during each training forward pass:

### Why would we intentionally drop neurons?

Because overfitting often happens when the model relies too heavily on a few specific activation paths.

Dropout makes training noisier on purpose:

- on each training step, some neurons are temporarily removed
- the network cannot depend on any one neuron too much
- this usually improves generalization

It is best thought of as a regularization trick, not a new model type.

```python
# p=0.5: 50% of neurons are randomly zeroed during training
# During model.eval(): Dropout is disabled automatically (all neurons active)
nn.Dropout(p=0.5)

# Typical dropout rates:
# After first hidden layer: 0.3-0.5 (more aggressive)
# After deeper layers: 0.1-0.3 (less aggressive deeper in the network)
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Dropout(0.3),      # Active during train(), disabled during eval()
    nn.Linear(64, 1)
)
```

---

## 5. Ways to Build Models

PyTorch offers multiple approaches to structuring a model. Each has different tradeoffs in flexibility vs conciseness.

### Why are there multiple ways to define a model?

Because not all models have the same shape of computation.

Some models are simple straight pipelines:

```text
input -> layer1 -> layer2 -> layer3 -> output
```

Others need:

- loops
- branching
- multiple heads
- dynamic numbers of layers

PyTorch gives you different containers so the code structure can match the model structure.

### Beginner rule of thumb

- use custom `nn.Module` by default
- use `nn.Sequential` when the model is just a straight stack
- use `nn.ModuleList` when you need a loop over layers
- use `nn.ModuleDict` when you need named selectable submodules

That is the main idea behind the examples below.

### Method 1: Custom nn.Module (most flexible, recommended)

Use this when the model has complex control flow, multiple input/output paths, skip connections, or custom logic:

```python
class ClassifierNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        # Define all sub-layers as attributes in __init__
        # PyTorch will register them automatically
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, num_classes)
        self.bn1  = nn.BatchNorm1d(hidden_dim)          # Normalize after layer1
        self.bn2  = nn.BatchNorm1d(hidden_dim // 2)     # Normalize after layer2
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply layers in order: Linear → BN → ReLU → Dropout → repeat
        # BN before activation is the standard ordering
        x = self.relu(self.bn1(self.layer1(x)))   # Layer 1 block
        x = self.drop(x)
        x = self.relu(self.bn2(self.layer2(x)))   # Layer 2 block
        x = self.drop(x)
        return self.output(x)   # Output layer (no activation — loss handles it)

model = ClassifierNet(input_dim=20, hidden_dim=128, num_classes=5)
```

### Method 2: nn.Sequential (linear stack, no branching)

Best for models where data flows straight through layers in order, with no skip connections, branches, or conditional logic:

### What does `nn.Sequential` really do?

It is just a container that says:

> "Take the output of layer 1 and feed it into layer 2, then layer 3, and so on."

So `nn.Sequential` is convenient, but only when the model really is a simple chain.

```python
# All layers applied in sequence: output of each is input to the next
model = nn.Sequential(
    nn.Linear(20, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 5)
)

# Named Sequential: gives each layer an accessible name
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
    ('fc1',     nn.Linear(20, 128)),
    ('bn1',     nn.BatchNorm1d(128)),
    ('relu1',   nn.ReLU()),
    ('dropout', nn.Dropout(0.3)),
    ('fc2',     nn.Linear(128, 5))
]))
print(model.fc1)   # Access layer by name: Linear(in_features=20, out_features=128)
```

### Method 3: nn.ModuleList (dynamic, loop-based)

Use when the number of layers is not fixed at class definition time, or when you need to loop over layers in `forward()`:

### Why not use a plain Python list?

Because PyTorch needs to **register** layers to manage them properly.

Registered layers are the ones that:

- appear in `model.parameters()`
- move with `model.to(device)`
- get saved in `state_dict()`

A plain Python list can hold layers, but PyTorch will not automatically treat them as part of the model. `ModuleList` fixes that while still letting you iterate dynamically.

```python
class DynamicNet(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        # MUST use ModuleList, not a plain Python list!
        # A Python list is NOT registered as a sub-module:
        # → model.parameters() won't find layers in a plain list
        # → .to(device) won't move them
        # → state_dict() won't save them
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x):
        # Apply ReLU after all layers except the last one
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)   # Last layer: no activation (loss handles it)

# Creates a network with layers: 20→128→64→32→5
model = DynamicNet([20, 128, 64, 32, 5])

# The broken version — NEVER do this:
class BrokenNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Python list — layers are NOT registered as sub-modules
        self.layers = [nn.Linear(10, 5), nn.Linear(5, 1)]

broken = BrokenNet()
print(list(broken.parameters()))   # [] ← empty! Parameters invisible to optimizer!
```

**Registration is everything.** PyTorch only optimizes, saves, and moves what it can **find** in the internal module registry. If a layer is stored in a plain Python list or as a raw attribute that isn't an `nn.Module` or `nn.Parameter`, it is invisible to the framework.

### Method 4: nn.ModuleDict (named dynamic modules)

Use when you have a collection of named modules that you select by key at runtime:

This is useful when the model has several possible branches or heads and you want to choose one by name instead of hard-coding a single path.

```python
class MultiTaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(20, 64)   # Shared feature extractor
        # ModuleDict for named task heads — select by string key at runtime
        self.heads = nn.ModuleDict({
            'classification': nn.Linear(64, 5),
            'regression':     nn.Linear(64, 1)
        })

    def forward(self, x, task):
        features = torch.relu(self.backbone(x))    # Shared computation
        return self.heads[task](features)          # Task-specific head

model = MultiTaskNet()
out_cls = model(torch.rand(8, 20), task='classification')  # (8, 5) logits
out_reg = model(torch.rand(8, 20), task='regression')      # (8, 1) value
```

---

## 6. Inspecting a Model

After building a model, always inspect it to verify the architecture is what you intended:

```python
model = ClassifierNet(input_dim=20, hidden_dim=128, num_classes=5)

# Print full architecture — shows all registered sub-modules with their types
print(model)
# ClassifierNet(
#   (layer1): Linear(in_features=20, out_features=128, bias=True)
#   (layer2): Linear(in_features=128, out_features=64, bias=True)
#   (output): Linear(in_features=64, out_features=5, bias=True)
#   (bn1):   BatchNorm1d(128, ...)
#   ...
# )

# Count parameters — useful for comparing model sizes
def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total, trainable = count_params(model)
print(f"Total: {total:,}, Trainable: {trainable:,}")

# Named parameters with shapes and trainability status
for name, param in model.named_parameters():
    status = 'trainable' if param.requires_grad else 'frozen'
    print(f"  {name:30s} {str(param.shape):20s} {status}")

# State dict — all parameters as a dict (used for saving/loading)
sd = model.state_dict()
for key in sd:
    print(f"  {key}: {sd[key].shape}")
```

---

## 7. Freezing and Unfreezing Parameters

Freezing parameters is the mechanism behind transfer learning. A frozen parameter's gradient is never computed, so the optimizer never updates it:

```python
model = ClassifierNet(input_dim=20, hidden_dim=128, num_classes=5)

# ── Freeze ALL parameters ──────────────────────────────────────
# After this, backward() still runs but skips gradient computation for these params
for param in model.parameters():
    param.requires_grad = False

# ── Unfreeze specific layers ───────────────────────────────────
# Common pattern: freeze a pretrained backbone, unfreeze only the output head
for param in model.output.parameters():
    param.requires_grad = True

# ── Only pass trainable params to optimizer ────────────────────
# If you pass ALL params, the optimizer will still try to update frozen ones
# Using filter ensures only trainable params get update steps
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

# ── Verify what's trainable ────────────────────────────────────
for name, param in model.named_parameters():
    print(f"{name}: trainable={param.requires_grad}")
```

---

## 8. Weight Initialization

When you create layers, PyTorch applies sensible default initialization (Kaiming uniform for linear layers). But for specific architectures or tasks, custom initialization can improve training:

```python
def init_weights(module):
    """Apply to all sub-modules using model.apply(init_weights)"""
    if isinstance(module, nn.Linear):
        # Kaiming (He) initialization: calibrated for ReLU activations
        # Avoids vanishing/exploding activations in the first few forward passes
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        nn.init.zeros_(module.bias)          # Zero-init bias (common)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)         # gamma = 1 (no initial scaling)
        nn.init.zeros_(module.bias)          # beta = 0 (no initial shift)

# model.apply recursively calls init_weights on every sub-module
model.apply(init_weights)

# Common init methods and when to use them:
nn.init.zeros_(tensor)          # All zeros — for bias terms
nn.init.ones_(tensor)           # All ones — for BatchNorm gamma
nn.init.uniform_(tensor, a=-0.1, b=0.1)   # Uniform between a and b
nn.init.normal_(tensor, mean=0, std=0.01) # Gaussian — small random init
nn.init.xavier_uniform_(tensor)           # Xavier: calibrated for tanh/sigmoid
nn.init.kaiming_uniform_(tensor, nonlinearity='relu')  # He: calibrated for ReLU
nn.init.kaiming_normal_(tensor, nonlinearity='relu')   # Same but from normal dist
```

**Why initialization matters:**
- Bad initialization causes **vanishing gradients** (weights too small → activations → 0 → gradients → 0)
- Or **exploding gradients** (weights too large → activations → ∞)
- Or **symmetry breaking failure** (all weights equal → all neurons learn the same feature)
- Xavier is calibrated for tanh/sigmoid; Kaiming accounts for ReLU zeroing half its inputs

---

## 9. Interview Questions

<details>
<summary><strong>Q1: What does super().__init__() do in nn.Module?</strong></summary>

It calls the `__init__` of `nn.Module`, which sets up the internal dictionaries `_parameters`, `_modules`, `_buffers`, `_forward_hooks`, etc. that PyTorch uses to track the model's state. Without this call, assigning `self.layer = nn.Linear(...)` won't be registered — the layer's weights will be invisible to `model.parameters()`, `model.to(device)`, and `state_dict()`. The model appears to work but the optimizer won't update the layer's weights.
</details>

<details>
<summary><strong>Q2: What is the difference between nn.Parameter and a regular tensor?</strong></summary>

`nn.Parameter` is a wrapper that: (1) sets `requires_grad=True` automatically; (2) registers the tensor in the module's `_parameters` dict when assigned as an attribute. This makes it visible to `model.parameters()` and thus to the optimizer. A regular tensor assigned as `self.x = torch.tensor(...)` is NOT registered — it won't appear in `model.parameters()`, won't be moved by `.to(device)`, and won't be saved in `state_dict()`.
</details>

<details>
<summary><strong>Q3: Why must you use nn.ModuleList instead of a Python list?</strong></summary>

When you store layers in a Python `list` inside an `nn.Module`, PyTorch cannot discover them through its registration system. They won't be included in `model.parameters()` (optimizer ignores them), won't move with `.to(device)`, and won't be saved in `state_dict()`. `nn.ModuleList` registers all contained modules as proper sub-modules so they participate in all Module operations.
</details>

<details>
<summary><strong>Q4: What is the difference between BatchNorm and LayerNorm?</strong></summary>

- **BatchNorm**: normalizes **across the batch dimension**. Computes mean/variance from the current batch. Requires batch size > 1. Behaves differently in train vs eval (uses running stats in eval). Good for CNNs.
- **LayerNorm**: normalizes **across the feature dimension** of a single sample. Independent of batch size; same behavior in train and eval. Preferred in Transformers and RNNs where sequence positions have their own statistics and batch-wise normalization would be meaningless.
</details>

<details>
<summary><strong>Q5: What is weight initialization and why does it matter?</strong></summary>

Weight initialization sets the starting values of parameters before training. Bad initialization causes: (1) **vanishing gradients** — weights too small → layer outputs converge to 0 → zero gradient → no learning in early layers; (2) **exploding gradients** — weights too large → outputs diverge → NaN; (3) **symmetry** — all weights equal → all neurons in a layer compute the same function and receive the same gradient → layer is effectively size 1. Xavier/Glorot init is calibrated for tanh; Kaiming/He init accounts for ReLU's zero-output region.
</details>

---

## 🔗 References
- [nn.Module Docs](https://pytorch.org/docs/stable/nn.html)
- [CampusX Video 5](https://www.youtube.com/watch?v=CAgWNxlmYsc)
