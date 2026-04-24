---
id: 02-tensors
title: "Video 2: Tensors in PyTorch"
sidebar_label: "02 · Tensors"
sidebar_position: 2
description: Deep dive into PyTorch tensors — creation, shapes, dtypes, operations, broadcasting, and GPU.
tags: [pytorch, tensors, campusx, numpy, gpu, operations]
---

# Tensors in PyTorch
**📺 CampusX — Practical Deep Learning using PyTorch | Video 2**

> **What you'll learn:** Everything about tensors — the fundamental building block of all PyTorch computations.

---

## 1. What is a Tensor?

A **tensor** is an n-dimensional array — a generalization of scalars, vectors, and matrices to any number of dimensions. The word "tensor" sounds intimidating, but it's really just a container for numbers arranged in a grid of any number of axes.

Think of tensors as containers for numerical data at different levels of nesting:

```
Rank 0 → Scalar      →  42
Rank 1 → Vector      →  [1, 2, 3]
Rank 2 → Matrix      →  [[1, 2],
                          [3, 4]]
Rank 3 → 3D Tensor   →  Image: (Height, Width, Channels)
Rank 4 → 4D Tensor   →  Batch of images: (N, C, H, W)
```

### A visual way to think about dimension

A tensor's dimension tells you **in how many directions the data extends**:
- A scalar does not extend in any direction → `0D`
- A vector extends in one direction (along its length) → `1D`
- A matrix extends in two directions (rows and columns) → `2D`
- An RGB image extends across channels, height, and width → `3D`
- A batch of RGB images adds one more axis (the batch index) → `4D`

This is a helpful intuition because shape debugging often becomes easier when you ask: "what new axis am I adding or removing?"

### Deep-learning examples

Almost every object in a neural network is a tensor:

| Tensor rank | Deep-learning example | Why it matters |
|---|---|---|
| `0D` | Loss value | Training reduces everything to one scalar objective |
| `1D` | Word embedding vector | One token represented as a vector of learned numbers |
| `2D` | Grayscale image OR batch of feature vectors | Height × Width, or batch × features |
| `3D` | RGB image | Channels × Height × Width |
| `4D` | Batch of RGB images | Batch × Channels × Height × Width |
| `5D+` | Video or sequence batches | Extra axes for time or sequence structure |

### Real-world tensor shapes

| Data type | Tensor shape | How to read it |
|---|---|---|
| Grayscale image | `(28, 28)` | H × W |
| Color image | `(3, 224, 224)` | C × H × W |
| Batch of images | `(32, 3, 224, 224)` | N × C × H × W |
| Text token sequence | `(50,)` | sequence length |
| Batch of sequences | `(16, 50, 128)` | N × seq_len × embedding_dim |
| Video | `(T, C, H, W)` | Frames × channels × H × W |

### Shape reading rule of thumb

When you see a tensor shape, read it left to right as:
- `batch` dimension first (how many samples)
- then structural dimensions (channels, sequence length, height, width)
- finally feature or embedding dimensions

Examples:
- `(32, 10)` → 32 samples, each with 10 features
- `(64, 3, 224, 224)` → 64 RGB images
- `(16, 128, 768)` → 16 sequences, each 128 tokens long, each token in 768-dim space

---

## 2. Creating Tensors

There are several ways to create tensors in PyTorch. Choosing the right creation method matters for performance and correctness.

### 2.1 From Python data

The most basic way: pass Python scalars, lists, or nested lists to `torch.tensor()`:

```python
import torch

# ── Scalar (0D) ───────────────────────────────────────────────
t0 = torch.tensor(7)
print(t0)          # tensor(7)
print(t0.ndim)     # 0  — zero axes
print(t0.shape)    # torch.Size([]) — shape is empty (no dimensions)
print(t0.item())   # 7  — extract as Python int/float (useful for logging)

# ── 1D Vector ──────────────────────────────────────────────────
t1 = torch.tensor([1.0, 2.0, 3.0])
print(t1.shape)    # torch.Size([3]) — one axis with 3 elements

# ── 2D Matrix ──────────────────────────────────────────────────
t2 = torch.tensor([[1, 2, 3],
                    [4, 5, 6]])
print(t2.shape)    # torch.Size([2, 3]) — 2 rows, 3 columns

# ── 3D Tensor ──────────────────────────────────────────────────
# Think of this as a stack of 2 (2×2) matrices
t3 = torch.tensor([[[1, 2], [3, 4]],
                    [[5, 6], [7, 8]]])
print(t3.shape)    # torch.Size([2, 2, 2]) — depth=2, rows=2, cols=2
```

**Walkthrough:**
- `torch.tensor(7)` creates a scalar with an empty shape `torch.Size([])` — no axes at all.
- `ndim` tells you how many axes exist; `shape` gives you the length of each axis.
- `.item()` extracts a Python scalar from a 0D or single-element tensor. Use this when logging loss values — it avoids holding a reference to the computation graph.
- As soon as values are nested inside lists of lists, the tensor rank increases from vector → matrix → 3D → and so on.

### 2.2 Factory functions (pre-filled tensors)

Rather than always providing exact values, factory functions create tensors filled with specific patterns. These are used constantly for initializing weights, creating masks, and setting up inputs:

```python
# ── Constant-fill tensors ─────────────────────────────────────
torch.zeros(3, 4)              # 3×4 tensor of 0.0  — used for zero-init
torch.ones(2, 3)               # 2×3 tensor of 1.0  — useful for masks
torch.full((3, 3), 5.0)        # 3×3 tensor of 5.0  — any constant value

# ── Identity matrix ─────────────────────────────────────────────
# Diagonal of 1s, zeros elsewhere — used in linear algebra and residual connections
torch.eye(4)                   # 4×4 identity matrix

# ── Range-based tensors ──────────────────────────────────────────
# Like Python range() or numpy arange(), but returns a tensor
torch.arange(0, 10, 2)         # tensor([0, 2, 4, 6, 8]) — start, stop, step
torch.linspace(0, 1, 5)        # tensor([0.00, 0.25, 0.50, 0.75, 1.00])
                                # evenly spaced between 0 and 1, 5 points

# ── Random tensors ─────────────────────────────────────────────
# Uniform distribution: values drawn from [0, 1)
torch.rand(3, 4)
# Standard normal: values drawn from N(0, 1) — mean=0, std=1
# Used for weight initialization (values centered around 0)
torch.randn(3, 4)
# Random integers in [0, 10) — used for random class labels, indices
torch.randint(0, 10, (3, 4))

# ── Shape-matching creation ──────────────────────────────────────
# Create a tensor with the same shape as an existing one
# Very useful when you want a zero-filled tensor of the same shape as your input
x = torch.rand(2, 3)
torch.zeros_like(x)            # Same shape as x, all zeros
torch.ones_like(x)             # Same shape as x, all ones
torch.rand_like(x)             # Same shape as x, random values

# ── Reproducibility with manual seed ────────────────────────────
# Always set a seed before generating random tensors in experiments
# This guarantees the same "random" values every run
torch.manual_seed(42)
x = torch.rand(3, 3)
torch.manual_seed(42)          # Reset to the same seed
y = torch.rand(3, 3)
print(torch.equal(x, y))       # True — same seed produces same values
```

### 2.3 Uninitialized tensors

Sometimes performance matters more than initialization:

```python
# torch.empty() allocates memory without initializing it
# The values printed will be whatever happens to be in that memory location
x = torch.empty(2, 3)
print(x)   # "random-looking" garbage values — this is NOT random, just uninitialized

# This is faster than zeros/ones because it skips the fill step
# Only use it when you immediately overwrite every value afterward
# If you're not sure, use zeros() or randn() instead
```

`torch.empty()` is useful in performance-critical code where you plan to immediately write every element (e.g., you're filling it inside a loop). For almost all other cases, use an initialized factory function.

### 2.4 From NumPy

Since NumPy is the standard Python array library, PyTorch makes it easy to convert to and from NumPy arrays. This bridge is essential in real projects that mix traditional ML (scikit-learn, NumPy) with deep learning (PyTorch):

```python
import numpy as np

arr = np.array([1.0, 2.0, 3.0])

# ── Option 1: from_numpy — shares memory with the NumPy array (zero-copy)
# Changing one changes the other! They point to the same memory.
t = torch.from_numpy(arr)
arr[0] = 99.0                  # Modify the NumPy array
print(t)   # tensor([99.,  2.,  3.])  ← t also changed! (same memory)

# ── Option 2: torch.tensor — copies data into a new tensor (safe)
# t2 is independent of arr — modifying arr does not affect t2
t2 = torch.tensor(arr)
arr[0] = 0.0                   # Modify arr again
print(t2)  # tensor([99.,  2.,  3.])  ← t2 is NOT affected

# ── Back to NumPy from PyTorch (CPU only)
back = t2.numpy()              # Shares memory again (like from_numpy)
```

:::warning
`torch.from_numpy()` shares memory with the NumPy array. Modifying one silently modifies the other. Always use `torch.tensor(arr)` if you want an independent copy that won't surprise you with unexpected mutations.
:::

**Why the NumPy bridge matters in practice:**
- Many real projects start with NumPy arrays from CSV files or scikit-learn preprocessing
- Matplotlib and other visualization libraries require NumPy
- Conversion between them happens constantly at data pipeline boundaries
- The big differences remain: PyTorch tensors participate in **autograd** and can live on **GPU**; NumPy arrays cannot do either

---

## 3. Tensor Attributes

Every tensor carries metadata that describes its content. Knowing these attributes is the first debugging step when something goes wrong:

```python
x = torch.rand(3, 4, dtype=torch.float32)

x.shape        # torch.Size([3, 4])   — same as x.size()
x.ndim         # 2                     — number of axes/dimensions
x.numel()      # 12                    — total number of elements (3×4)
x.dtype        # torch.float32         — numeric data type
x.device       # device(type='cpu')    — where the tensor lives
x.requires_grad  # False              — is gradient tracking enabled?
x.is_contiguous()  # True             — is memory layout contiguous?
```

### Data Types (dtypes)

The dtype controls how each number is stored in memory. Using the wrong dtype is one of the most common causes of PyTorch errors:

| dtype | Alias | Bytes per element | Use case |
|---|---|---|---|
| `torch.float32` | `torch.float` | 4 | **Default for training** — best balance of precision and speed |
| `torch.float64` | `torch.double` | 8 | Scientific computation needing extra precision |
| `torch.float16` | `torch.half` | 2 | Mixed precision training on GPU (saves memory) |
| `torch.bfloat16` | — | 2 | Modern GPU/TPU training (better numerical range than float16) |
| `torch.int32` | `torch.int` | 4 | General integers |
| `torch.int64` | `torch.long` | 8 | **Indices, class labels** — CrossEntropyLoss requires this |
| `torch.bool` | — | 1 | Masks, boolean operations |

```python
# dtype follows the literal values you provide:
torch.tensor([1, 2, 3]).dtype        # int64  — Python ints → int64
torch.tensor([1.0, 2.0, 3.0]).dtype  # float32 — Python floats → float32
torch.tensor([True, False]).dtype    # bool

# Cast between dtypes when needed:
x = torch.tensor([1.0, 2.0, 3.0])  # float32
x_half = x.to(torch.float16)       # Method 1: .to(dtype) — most general
x_int  = x.int()                   # Method 2: shorthand (.int, .long, .float)
x_long = x.long()                  # .long() is equivalent to .to(torch.int64)
x_fp32 = x.float()                 # .float() is equivalent to .to(torch.float32)

# IMPORTANT: class labels MUST be torch.long (int64) for CrossEntropyLoss
# CrossEntropyLoss uses labels as array indices internally — indices must be int
labels = torch.tensor([0, 1, 2], dtype=torch.long)  # correct
# labels = torch.tensor([0, 1, 2], dtype=torch.float32)  # WRONG → RuntimeError
```

---

## 4. Indexing and Slicing

Tensor indexing follows the same rules as NumPy, so if you know NumPy indexing, you already know this. The key is understanding that the same rules extend to higher dimensions:

```python
x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
# x is:
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [12., 13., 14., 15.]])

# ── Basic indexing ─────────────────────────────────────────────
x[0]           # First row:  tensor([0., 1., 2., 3.])
x[-1]          # Last row:   tensor([12., 13., 14., 15.])
x[0, 1]        # Element at row 0, col 1: tensor(1.)
x[0][1]        # Same thing — two separate index operations

# ── Slicing ──────────────────────────────────────────────────────
x[:, 0]        # ALL rows, column 0: tensor([0., 4., 8., 12.])
x[0:2, 1:3]    # Rows 0-1, columns 1-2: tensor([[1., 2.], [5., 6.]])
x[::2]         # Every other row (rows 0 and 2)

# ── Boolean (mask) indexing ──────────────────────────────────────
# Create a boolean mask — True where condition holds
mask = x > 7
# Select all elements where mask is True — result is always 1D
x[mask]        # tensor([8., 9., 10., 11., 12., 13., 14., 15.])

# ── Fancy indexing — select specific rows by index ────────────────
idx = torch.tensor([0, 2, 3])   # Select rows 0, 2, and 3
x[idx]         # tensor([[0., 1., 2., 3.], [8., 9., 10., 11.], [12., 13., 14., 15.]])

# ── In-place modification ────────────────────────────────────────
x[0, 0] = 100  # Changes the actual tensor value in memory
```

---

## 5. Reshaping Operations

Reshaping is one of the most frequently used tensor operations — you'll reshape tensors constantly when moving data between layers that expect different shapes.

```python
x = torch.arange(12, dtype=torch.float32)   # 1D tensor: [0, 1, 2, ..., 11]

# ── reshape — most common, handles contiguity automatically ──────
x.reshape(3, 4)        # (3, 4): 3 rows, 4 columns
x.reshape(3, -1)       # -1 means "infer this dimension" → (3, 4)
x.reshape(-1, 4)       # → (3, 4): PyTorch infers 3
x.reshape(-1)          # Flatten to 1D: (12,)

# ── view — like reshape but REQUIRES contiguous memory ───────────
# Faster than reshape when memory is already contiguous (no copy needed)
x.view(3, 4)           # Same as reshape when memory is contiguous

# ── Flatten — reduce to 1D ───────────────────────────────────────
x.reshape(3, 4).flatten()                      # (12,)
x.reshape(3, 4).flatten(start_dim=0)           # same as above
# Used inside neural network forward() to bridge conv layers and linear layers:
# import torch.nn as nn
# nn.Flatten()(x.reshape(1, 3, 4))             # keeps batch dim, flattens rest

# ── squeeze / unsqueeze — remove/add size-1 dimensions ──────────
x = torch.rand(3, 1, 4)
x.squeeze(1)      # Remove dim 1 (which has size 1): (3, 4)
x.squeeze()       # Remove ALL size-1 dimensions: (3, 4)
x.unsqueeze(0)    # Insert a new dim at position 0: (1, 3, 1, 4)
# unsqueeze is very common for adding a batch dimension:
# sample = torch.rand(3, 224, 224)         # single image
# batch  = sample.unsqueeze(0)             # (1, 3, 224, 224) — batch of 1

# ── permute — reorder dimensions ────────────────────────────────
# PyTorch stores images as (C, H, W) internally
img = torch.rand(3, 224, 224)   # PyTorch format: channels first
# Matplotlib expects (H, W, C) — channels last
img.permute(1, 2, 0).shape      # (224, 224, 3) — now matplotlib-compatible
```

### view() vs reshape() — important distinction

This is a common source of confusion. The difference is about **memory layout**:

### Why do beginners get confused here?

Because both `view()` and `reshape()` seem to do the same thing in simple examples: change the tensor's shape.

The difference is not about the visible values. It is about how the tensor is stored in memory underneath.

Beginner rule of thumb:

- if you just want a new shape, use `reshape()`
- if you specifically care that no memory copy happens, then think about `view()`

So you do not need to master tensor memory internals on day one. You mainly need to know why `view()` sometimes fails unexpectedly.

```python
x = torch.rand(4, 3)
y = x.T                      # Transpose: swaps axes — produces a NON-contiguous tensor
                              # Non-contiguous means the underlying memory is not in row-major order

# view() works ONLY on contiguous tensors
# y.view(12)                 # ❌ RuntimeError: non-contiguous tensor
y.reshape(12)                # ✅ reshape() always works — copies memory if needed
y.contiguous().view(12)      # ✅ Explicitly make it contiguous, then view

# Check contiguity:
print(x.is_contiguous())     # True — x was created normally
print(y.is_contiguous())     # False — y is a transpose view
```

**Rule of thumb:**
- Use `reshape()` for everyday code — it handles everything
- Use `view()` only when you need to guarantee no memory copy happened (performance-critical code)
- Use `transpose()` / `.T` to swap exactly two dimensions
- Use `permute()` to reorder many dimensions (essential for image format conversions)

---

## 6. Mathematical Operations

### Element-wise operations

Element-wise operations apply a function to each element independently — the shapes must match (or be broadcastable):

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# ── Arithmetic — all element-wise ─────────────────────────────
a + b       # tensor([5., 7., 9.])   — addition
a - b       # tensor([-3., -3., -3.]) — subtraction
a * b       # tensor([4., 10., 18.]) — element-wise multiply (NOT matrix multiply!)
a / b       # tensor([0.25, 0.40, 0.50])
a ** 2      # tensor([1., 4., 9.])   — squaring each element
a % 2       # tensor([1., 0., 1.])   — remainder

# ── Math functions ─────────────────────────────────────────────
torch.sqrt(a)                       # √each element
torch.exp(a)                        # e^each element — used in softmax
torch.log(a)                        # natural log — used in cross-entropy
torch.abs(torch.tensor([-1., 2., -3.]))  # absolute value
torch.clamp(a, min=1.5, max=2.5)    # clip values to [1.5, 2.5]

# ── In-place operations (trailing underscore = in-place) ────────
# In-place modifies the tensor directly — no new tensor is created
# Saves memory but can break autograd if used on tensors that need gradients
a.add_(b)      # a = a + b (modifies a in place)
a.mul_(2)      # a = a * 2
a.zero_()      # Fill a with zeros

# All three forms are equivalent and interchangeable:
# torch.add(a, b)  ==  a + b  ==  a.add(b)
```

### Matrix operations

Matrix multiplication is the single most important operation in deep learning — every linear layer, every attention mechanism, every convolution is ultimately a matrix multiply:

### Why is matrix multiplication such a big deal?

Because a huge amount of neural-network computation can be reduced to:

- multiply learned weights by input activations
- produce a new representation

So even though the notation can look abstract, this is the core engine inside many layers.

If you understand the shape story:

```python
(batch, in_features) @ (in_features, out_features) -> (batch, out_features)
```

you can reason about a lot of PyTorch code much more confidently.

## Visual Reference

![Linear model weight matrix and tensor flow](https://datahacker.rs/wp-content/uploads/2021/01/Picture1.jpg)

*A linear layer is a weight matrix multiplied by an input tensor: `output = W @ input + b`. The input tensor shape `(batch, features)` is multiplied by `W` with shape `(features, out)` to produce `(batch, out)`. Understanding this matrix-tensor relationship is the foundation for reasoning about every `nn.Linear` layer shape in PyTorch.*

```python
A = torch.rand(3, 4)
B = torch.rand(4, 2)

# ── 2D matrix multiply ────────────────────────────────────────
# (3,4) × (4,2) = (3,2) — inner dimensions must match, outer dims are result
C = A @ B               # Operator form — most readable
C = torch.matmul(A, B)  # Functional form — same result, works for any rank
C = torch.mm(A, B)      # 2D-only form — slightly faster, but no broadcasting

# ── Batched matrix multiply ──────────────────────────────────
# In deep learning, you process batches — apply matrix multiply to each pair
A = torch.rand(8, 3, 4)   # 8 matrices of shape (3,4)
B = torch.rand(8, 4, 2)   # 8 matrices of shape (4,2)
C = torch.bmm(A, B)        # (8, 3, 2) — multiply each pair independently

# ── Dot product (1D only) ────────────────────────────────────
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
torch.dot(a, b)   # 1*4 + 2*5 + 3*6 = 32  — sum of element-wise products

# ── Linear algebra ───────────────────────────────────────────
# Less common in standard training, more common in research and custom layers
A_square = torch.rand(3, 3)
torch.linalg.det(A_square)    # Determinant
torch.linalg.inv(A_square)    # Matrix inverse
torch.linalg.norm(A)           # Frobenius norm (size of matrix)
torch.linalg.eig(A_square)    # Eigenvalues and eigenvectors
```

### Reduction operations

Reductions collapse dimensions by computing a summary statistic. The `dim` argument controls **which axis gets collapsed**:

```python
x = torch.tensor([[1., 2., 3.],
                   [4., 5., 6.]])
# x has shape (2, 3): 2 rows, 3 columns

x.sum()              # tensor(21.)   — sum ALL elements, result is scalar
x.sum(dim=0)         # tensor([5., 7., 9.]) — collapse rows: sum each column
                     # shape goes from (2,3) → (3,)
x.sum(dim=1)         # tensor([ 6., 15.]) — collapse columns: sum each row
                     # shape goes from (2,3) → (2,)

# keepdim=True preserves the collapsed dimension as size 1
# This is crucial for broadcasting (adding the result back to the original)
x.sum(dim=1, keepdim=True)    # tensor([[ 6.], [15.]]) — shape (2,1) not (2,)

# Other reductions follow the same dim= logic
x.mean()             # Overall average
x.mean(dim=0)        # Column-wise average
x.max()              # Maximum element (scalar)
x.max(dim=0)         # Max in each column — returns (values, indices)
values, indices = x.max(dim=0)  # Unpack the named tuple

x.argmax()           # Index of the global maximum (flattened)
x.argmax(dim=1)      # Index of max in each row — the "predicted class" pattern!

x.std()              # Standard deviation of all elements
x.var()              # Variance
torch.cumsum(x, dim=1)  # Cumulative sum along columns
```

---

## 7. Broadcasting

Broadcasting lets you perform operations between tensors of **different shapes** without physically copying data. It's how you can add a bias vector to a whole batch without writing a loop.

### Why does broadcasting matter so much?

Because beginners often imagine that tensor shapes must always match exactly.

In practice, PyTorch can often "stretch" size-1 dimensions automatically so common math works naturally.

That is why code like this works:

```python
x + bias
```

even when `x` is a whole batch and `bias` is just one feature vector.

Broadcasting is one of the reasons tensor code stays compact instead of being full of Python loops.

**Rules (applied dimension by dimension, right-to-left):**
1. If tensors have different numbers of dims, prepend 1s to the smaller shape
2. Two dimensions are compatible if they are equal OR one of them is 1
3. The result shape = the maximum along each dimension pair

```python
# ── Example 1: Adding a row vector to a column vector ────────
# (3, 1) + (1, 4) → (3, 4)
# Rule: both are made (3,4) — the 1s expand to fill
a = torch.ones(3, 1)    # 3 rows, 1 column
b = torch.ones(1, 4)    # 1 row,  4 columns
(a + b).shape           # torch.Size([3, 4])

# ── Example 2: Adding a bias vector to a batch ──────────────
# This is exactly what nn.Linear does internally for the bias term
x    = torch.rand(64, 10)   # 64 samples, 10 features each
bias = torch.rand(10)        # one bias per feature — shape (10,)
# bias is broadcast: (10,) → (1, 10) → (64, 10)
(x + bias).shape             # (64, 10) — correct!

# ── Example 3: Per-channel image scaling ─────────────────────
img   = torch.rand(3, 224, 224)               # (C, H, W) image
scale = torch.tensor([0.5, 1.0, 0.8]).view(3, 1, 1)  # (3, 1, 1)
# scale broadcasts to (3, 224, 224) — each channel gets a different scale factor
(img * scale).shape   # (3, 224, 224)

# ── Visualizing how broadcasting expands shapes ──────────────
# shape (3, 1) + shape (1, 4)
# step 1: make same ndim → (3, 1) and (1, 4) already match
# step 2: expand 1s → (3, 4) and (3, 4)
# step 3: add element-wise → (3, 4)
```

**Key insight:** Broadcasting does not physically copy the smaller tensor across memory first. PyTorch behaves *as if* it expanded the smaller tensor, performs the operation, and returns the larger result. This means:
- Code stays compact (no explicit loops)
- Memory is not wasted (no copies)
- It's the reason bias vectors can be added to whole batches in one line

### `expand()` vs `repeat()`

Sometimes you need explicit expansion rather than implicit broadcasting:

```python
x = torch.tensor([[1], [2], [3]])    # (3, 1)

# expand() — no data copy, just changes how the tensor is viewed
# Only works on size-1 dimensions (can't "unexpand" a dim of size 3 to 4)
x.expand(3, 4)   # (3, 4) — view the data as if repeated 4 times along dim 1

# repeat() — actually copies the data
x.repeat(1, 4)   # (3, 4) — physically repeats data 4 times along dim 1
```

- `expand()` is memory-efficient but only works on size-1 dimensions
- `repeat()` physically duplicates values — useful when you need an actual independent copy
- Broadcasting gives you `expand()` behavior automatically during arithmetic

---

## 8. GPU Operations

Moving tensors to GPU is one line. Once on GPU, all operations on those tensors automatically run on the GPU — you don't need to change your math code at all:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Create directly on GPU ────────────────────────────────────
x = torch.rand(3, 3, device=device)  # Never touches CPU memory

# ── Move existing tensor to GPU ───────────────────────────────
x_cpu = torch.rand(3, 3)             # Created on CPU
x_gpu = x_cpu.to(device)            # Copy to GPU (.cuda() is shorthand on CUDA)

# ── Move back to CPU (required before converting to NumPy) ────
x_back = x_gpu.to("cpu")            # Copy back to CPU
arr    = x_back.numpy()             # NumPy can only read CPU memory

# ── Operations on GPU tensors run on GPU automatically ────────
y = x_gpu + x_gpu      # Element-wise addition — runs on GPU
z = x_gpu @ x_gpu.T   # Matrix multiply — runs on GPU

# ── IMPORTANT: both operands must be on the same device ───────
# This will crash with a RuntimeError about device mismatch:
# x_gpu + x_cpu   # ❌ RuntimeError: Expected all tensors on same device

# Fix by moving the CPU tensor to GPU first:
# x_gpu + x_cpu.to(device)   # ✅
```

---

## 9. Useful Tensor Utilities

### Comparison operations

```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([1., 0., 3.])

torch.equal(a, b)         # False — are ALL elements exactly equal?
torch.allclose(a, b)      # False — are all elements close within tolerance?
(a == b)                  # tensor([True, False, True]) — element-wise comparison
(a > b)                   # element-wise greater-than
```

### Stacking and concatenating

These operations are used constantly to combine tensors from different batches, branches, or models:

```python
a = torch.rand(2, 3)
b = torch.rand(2, 3)

# cat: join along an existing dimension (no new dimension is added)
torch.cat([a, b], dim=0)    # (4, 3) — stack along rows (more samples)
torch.cat([a, b], dim=1)    # (2, 6) — stack along columns (more features)

# stack: join along a NEW dimension (a new axis is created)
torch.stack([a, b], dim=0)  # (2, 2, 3) — new axis at position 0
# Use this to build a batch from individual samples:
# samples = [model(x_i) for x_i in inputs]
# batch   = torch.stack(samples, dim=0)

# Splitting (inverse of cat)
x     = torch.rand(6, 3)
parts = torch.chunk(x, 3, dim=0)         # Split into 3 equal chunks: each (2,3)
a, b, c = torch.split(x, [2, 2, 2], dim=0)  # Split by specified sizes

# Sorting
x = torch.tensor([3., 1., 4., 1., 5., 9.])
sorted_x, indices = torch.sort(x)
# sorted_x: tensor([1., 1., 3., 4., 5., 9.])
# indices: tells you where each element came from in the original tensor
```

---

## 10. Copying Tensors Safely

Understanding copying vs sharing is critical for avoiding silent bugs in training code:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

y = x                 # Same object — y IS x, not a copy
z = x.clone()         # New tensor with same values, own memory, same grad status
w = x.detach()        # Shares data memory, but has NO gradient connection

print(x.data_ptr() == z.data_ptr())   # False — z has its own memory
print(x.data_ptr() == w.data_ptr())   # True  — w shares x's memory

# When to use each:
# clone()         — when you need an independent copy for safe mutation
# detach()        — when you want the value but NOT the gradient history
# clone().detach() — when you want both: independent copy AND no gradient
```

This comes up constantly in:
- Logging (use `.item()` or `.detach()` before storing loss values)
- Debugging (use `.clone()` to snapshot a tensor before operations that modify it)
- Training tricks that require stopping gradient flow (e.g., target networks in RL)

---

## 11. Common Mistakes and How to Avoid Them

### Wrong dtype for loss function

```python
# CrossEntropyLoss uses labels as array indices internally
# Array indexing requires integers — specifically int64 (long) in PyTorch
labels_wrong = torch.tensor([0, 1, 2], dtype=torch.float32)   # ❌ RuntimeError
labels_right = torch.tensor([0, 1, 2], dtype=torch.long)       # ✅
```

### Mixing CPU and GPU tensors

```python
a = torch.rand(3).cuda()   # on GPU
b = torch.rand(3)          # on CPU
# c = a + b               # ❌ RuntimeError: tensors on different devices

c = a + b.to("cuda")       # ✅ move b to GPU first
```

### In-place operations break autograd

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
# x.add_(1)   # ❌ RuntimeError — in-place op on a leaf tensor that needs grad
x = x + 1     # ✅ Creates new tensor, computation graph is preserved
```

### NumPy conversion from GPU tensor

```python
x = torch.rand(3).cuda()
# x.numpy()              # ❌ RuntimeError: NumPy can't access GPU memory
x.cpu().numpy()          # ✅ Move to CPU first
x.detach().cpu().numpy() # ✅ Also detach if requires_grad=True
```

---

## 12. Tensor Patterns Inside Models

These patterns appear inside `forward()` methods constantly:

```python
x = torch.randn(32, 128)   # batch of 32 samples, each with 128 features

x.mean(dim=0)               # Feature-wise mean over the batch: shape (128,)
x.argmax(dim=1)             # Predicted class per sample: shape (32,)
x.unsqueeze(1).shape        # Add seq dim: (32, 1, 128) — for RNN input
x.flatten(start_dim=1)      # Keep batch dim, flatten the rest: (32, 128)
```

Mental model for the `dim=` argument:
- `dim=0` → "across the batch" (reducing batch dimension)
- `dim=1` → "across features" or "across classes" (reducing feature dimension)
- Flattening usually keeps `dim=0` and collapses everything else

---

## 13. Interview Questions

<details>
<summary><strong>Q1: What is a tensor? How is it different from a NumPy array?</strong></summary>

A tensor is an n-dimensional array — the fundamental data structure in PyTorch. Differences from NumPy:
1. **GPU support**: tensors can live on GPU (`.to("cuda")`), enabling massive parallelism. NumPy is CPU-only.
2. **Autograd**: tensors track operations for automatic differentiation (`requires_grad=True`). NumPy has no gradient concept.
3. **dtype system**: tensors have a richer type system including `float16`, `bfloat16` for mixed-precision training.
4. **Integration**: tensors plug directly into `nn.Module`, `torch.optim`, and `DataLoader`.
</details>

<details>
<summary><strong>Q2: Explain the difference between view() and reshape().</strong></summary>

Both change the shape without changing the underlying data. The difference is about memory contiguity:
- `view()`: requires the tensor to be **contiguous** in memory (elements laid out consecutively). Returns a view (zero copy, shares memory). Raises RuntimeError if not contiguous.
- `reshape()`: works always. Returns a view if memory is contiguous; otherwise silently makes a contiguous copy first.

Rule of thumb: use `reshape()` unless you specifically need to guarantee no copy was made (e.g., for performance benchmarking or memory constraints).
</details>

<details>
<summary><strong>Q3: What is broadcasting? Give an example.</strong></summary>

Broadcasting automatically expands tensors of smaller shapes to match larger shapes without copying data. PyTorch follows NumPy's rules: dimensions are compared right-to-left; a dimension of 1 can be broadcast to match any size. Example:
```python
x    = torch.rand(64, 10)  # batch of 64 samples, 10 features each
bias = torch.rand(10)      # one bias per feature — shape (10,)
result = x + bias          # bias broadcast: (10,) → (1,10) → (64,10)
```
Bias is effectively added to every row without copying it 64 times in memory.
</details>

<details>
<summary><strong>Q4: What does .item() do?</strong></summary>

`.item()` extracts a **Python scalar** from a single-element tensor. It's used when: (1) **logging loss values** — `loss.item()` returns a plain float and breaks the reference to the computation graph, preventing memory leaks; (2) **Python comparisons** — you can't use `if tensor > 0` but you can use `if tensor.item() > 0`.
</details>

<details>
<summary><strong>Q5: What is the significance of the dim argument in reduction operations?</strong></summary>

`dim` specifies which axis to collapse. For a `(2, 3)` tensor:
- `x.sum()` → scalar (collapse everything)
- `x.sum(dim=0)` → `(3,)` — each column summed across rows (collapse the row axis)
- `x.sum(dim=1)` → `(2,)` — each row summed across columns (collapse the column axis)
- `keepdim=True` preserves the collapsed dimension as size 1, which keeps the tensor broadcastable against the original.
</details>

<details>
<summary><strong>Q6: Why do class labels need to be torch.long (int64)?</strong></summary>

`CrossEntropyLoss` and `NLLLoss` use labels as **indices** into the output probability vector — they do `output[label_i]` internally for each sample. Array indexing requires integers. In PyTorch, the required integer type is `int64` (torch.long). Passing `float32` labels triggers a RuntimeError because floating-point values cannot be used as array indices.
</details>

---

## 🔗 References
- [PyTorch Tensor Docs](https://pytorch.org/docs/stable/tensors.html)
- [CampusX Video 2](https://www.youtube.com/watch?v=mDsFsnw3SK4)
