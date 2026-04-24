---
id: 08-gpu-training
title: "Video 8: Neural Network Training on GPU"
sidebar_label: "08 · GPU Training"
sidebar_position: 8
description: Moving PyTorch training to GPU — CUDA basics, device management, mixed precision, and multi-GPU.
tags: [pytorch, gpu, cuda, mixed-precision, training, campusx]
---

# Neural Network Training on GPU
**📺 CampusX — Practical Deep Learning using PyTorch | Video 8**

> **What you'll learn:** How to leverage GPU acceleration for drastically faster neural network training — device management, memory, mixed precision, and best practices.

---

## 1. Why GPU for Deep Learning?

In the previous video (video 7), you built and trained a complete ANN for breast cancer classification. The code worked, but training was slow and you had to limit the dataset size to keep it manageable. This video answers the obvious follow-up question: **what changes when we move training to a GPU?**

The short answer: almost nothing changes in your logic, but training becomes 10–100× faster, which lets you use more data, larger models, more epochs, and larger batches — all of which lead to better results.

Here is why GPUs are fundamentally faster for neural networks:

| | CPU | GPU |
|---|---|---|
| Cores | 4–64 powerful | 3,000–16,000 smaller |
| Memory | 16–128 GB RAM | 8–80 GB VRAM |
| Memory BW | ~50–100 GB/s | ~900–2,000 GB/s |
| Strength | Serial tasks, complex logic | Massively parallel math |
| Example | Intel Core i9 | NVIDIA A100 |

A matrix multiply `(1000×1000) @ (1000×1000)` = 2 billion FLOPs. On CPU: ~2 seconds. On GPU: ~0.002 seconds — **1000× speedup**.

Neural network forward and backward passes are almost entirely matrix multiplications and element-wise operations. Every single training step — for every batch — runs these same dense math operations. The GPU can execute thousands of those operations simultaneously in parallel, while the CPU does them one (or a few) at a time.

Neural network training is almost entirely matrix multiplications and element-wise ops → GPU wins by a huge margin.

### The practical motivation in the transcript

CampusX frames this lesson very concretely: CPU training forced the earlier ANN
project to use only a **subset** of the dataset, while GPU training makes it
reasonable to use the **full dataset**.

That shift matters because GPU training is not just about speed for its own
sake. It often changes what experiments become feasible:

- full dataset instead of a small subset
- more epochs
- larger batch sizes
- bigger models
- faster iteration during tuning

## Visual Reference

![GPU Streaming Multiprocessor — parallel CUDA cores](https://tvm.d2l.ai/_images/gpu_sm.svg)

*A GPU Streaming Multiprocessor (SM) contains hundreds of CUDA cores grouped into processing blocks. All cores in a warp execute the same instruction simultaneously on different data (SIMD). Matrix multiplications decompose perfectly into independent dot products, saturating all cores in parallel — this is why a single A100 can deliver 312 TFLOPS vs a few hundred GFLOPS on CPU.*

---

## 2. CUDA Basics

**CUDA** (Compute Unified Device Architecture) is NVIDIA's GPU programming framework. PyTorch uses it under the hood.

```python
import torch

# Is CUDA (NVIDIA GPU) available?
print(torch.cuda.is_available())          # True / False

# How many GPUs?
print(torch.cuda.device_count())          # e.g., 2

# Current device
print(torch.cuda.current_device())        # 0 (index of default GPU)

# GPU name
print(torch.cuda.get_device_name(0))      # e.g., "NVIDIA GeForce RTX 3090"

# Memory info (bytes)
print(torch.cuda.memory_allocated(0))     # Memory in use
print(torch.cuda.memory_reserved(0))      # Memory reserved by caching allocator
print(torch.cuda.max_memory_allocated(0)) # Peak usage

# CUDA version
print(torch.version.cuda)                 # e.g., "11.8"
```

### Notebook / Colab reality check

In many learning setups, especially notebooks, the missing step is not PyTorch
code but **enabling GPU runtime** first. A simple checklist is:

1. confirm the environment actually has a GPU
2. verify `torch.cuda.is_available()`
3. move both model and batches to the selected device
4. only then expect speedups

If step 2 is `False`, the rest of the GPU code will silently fall back to CPU
or fail with device errors.

### Transcript-backed practical speed tips

CampusX calls out two simple optimizations after moving the code to GPU:

- try a larger batch size like `64` or `128` if memory allows
- enable `pin_memory=True` in the `DataLoader`

Why those help:

- larger batches can keep the GPU busier
- `pin_memory=True` reduces some CPU to GPU transfer overhead for large datasets

That means the GPU lesson is really about two layers of speedup:

1. run the math on GPU
2. reduce data-transfer bottlenecks around that GPU

---

## 3. Moving Tensors and Models to GPU

```python
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# ── Tensors ────────────────────────────────────────────────
x_cpu = torch.rand(3, 3)            # Created on CPU
x_gpu = x_cpu.to(device)           # Move to GPU

# CUDA-only shorthand
if torch.cuda.is_available():
    x_gpu = x_cpu.cuda()
    x = torch.zeros(3, 3, device="cuda")

# Create directly on the selected device
x = torch.rand(3, 3, device=device)

# Back to CPU
x_cpu = x_gpu.to("cpu")
x_cpu = x_gpu.cpu()

# Convert to numpy (must be on CPU)
arr = x_gpu.cpu().numpy()           # ✅
# arr = x_gpu.numpy()              # ❌ RuntimeError

# ── Models ─────────────────────────────────────────────────
model = nn.Linear(3, 2)
model = model.to(device)           # Move ALL parameters to GPU
# or: model.cuda()

# Verify model is on GPU
print(next(model.parameters()).device)   # cuda:0
```

### Code walkthrough

- `torch.device(...)` gives your code one place to decide CPU vs GPU.
- Tensors and models do not move automatically together; you must move both.
- `.cpu().numpy()` is required because NumPy cannot read GPU memory directly.
- A device mismatch error almost always means one tensor stayed on CPU by accident.

---

## 4. The Golden Rule: Keep Everything on the Same Device

```python
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = nn.Linear(10, 2).to(device)

for X, y in train_loader:
    # DataLoader returns CPU tensors by default
    X = X.to(device)    # MUST move to GPU before passing to model
    y = y.to(device)

    output = model(X)   # ✅ model and X both on GPU
    loss   = criterion(output, y)  # ✅ output and y both on GPU
    loss.backward()
    optimizer.step()

# Common bugs:
# model on GPU, X on CPU → RuntimeError: Expected all tensors on the same device
# loss.item() works regardless of device (extracts Python float)
```

### Async Transfer with non_blocking=True

```python
# non_blocking=True: CPU→GPU transfer happens asynchronously
# GPU starts computing previous batch while CPU transfers next batch
X = X.to(device, non_blocking=True)
y = y.to(device, non_blocking=True)
# This is a performance optimization — requires pin_memory=True in DataLoader
```

---

## 5. Complete GPU Training Pipeline

This is the ANN training loop from video 7, rewritten with GPU support. Compare it side by side with the CPU version: the structure is identical, but we add `.to(device)` calls in three places (model, X_batch, y_batch) and set `pin_memory=True` in the DataLoader.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# ── 1. Device selection ─────────────────────────────────────
# This single line makes the code portable:
# - On a machine with a GPU: device = cuda
# - On a CPU-only machine: device = cpu
# The rest of the code references `device` everywhere — so nothing else changes.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# ── 2. Data ─────────────────────────────────────────────────
X = torch.randn(10000, 100)   # 10,000 samples, 100 features each
y = torch.randint(0, 10, (10000,))   # 10-class labels

train_ds, val_ds = random_split(TensorDataset(X, y), [8000, 2000])

# pin_memory=True: keeps CPU tensors in "pinned" (page-locked) memory
# This allows GPU to DMA-copy them directly, without an intermediate CPU buffer step.
# Rule: always use pin_memory=True when training on GPU.
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False,
                          num_workers=4, pin_memory=True)

# ── 3. Model ─────────────────────────────────────────────────
# .to(device) moves ALL model parameters and buffers to GPU memory.
# After this call, every matrix multiply inside the model runs on GPU.
model = nn.Sequential(
    nn.Linear(100, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(256, 10)   # 10 output logits, one per class
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── 4. Loss, optimizer, scheduler ───────────────────────────
criterion = nn.CrossEntropyLoss()   # Handles softmax + NLL loss together
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# OneCycleLR ramps LR up then down over the full training run.
# It needs to know total steps = epochs × batches_per_epoch.
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-2,
    steps_per_epoch=len(train_loader), epochs=30
)

# ── 5. Training loop ─────────────────────────────────────────
for epoch in range(1, 31):
    model.train()   # Enable BatchNorm + Dropout in training mode

    for X_batch, y_batch in train_loader:
        # DataLoader returns CPU tensors by default.
        # MUST move to same device as model before any computation.
        # non_blocking=True: starts the GPU transfer and continues Python code
        # immediately (asynchronous). The GPU buffers the incoming data.
        # This overlaps data loading and GPU computation — both happen at once.
        # Requires pin_memory=True in DataLoader to work.
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()               # Reset gradients from previous batch
        loss = criterion(model(X_batch), y_batch)  # Forward + loss on GPU
        loss.backward()                     # Backward pass on GPU
        # Gradient clipping: if gradient norm > 1.0, scale it down proportionally.
        # Prevents exploding gradients, especially with large models or LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()    # Update weights
        scheduler.step()    # Update LR (per batch, not per epoch for OneCycleLR)

    # ── Validation ────────────────────────────────────────────
    model.eval()    # Disable Dropout; BatchNorm uses running stats
    correct = 0
    with torch.no_grad():   # No gradients needed for validation
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            # argmax(1) picks the class with highest score for each sample
            correct += (model(X_batch).argmax(1) == y_batch).sum().item()

    print(f"Epoch {epoch:2d} | Val Acc: {correct / len(val_ds):.4f}")
```

### Why this example is close to real training code

The 5 changes vs CPU-only code:
1. `device = torch.device(...)` — one place to control CPU vs GPU
2. `model.to(device)` — moves model parameters to GPU memory
3. `X_batch.to(device, non_blocking=True)` — moves batch data to GPU
4. `pin_memory=True` in DataLoader — enables efficient async CPU→GPU transfers
5. `non_blocking=True` in `.to()` — overlaps data loading with GPU computation

Everything else — the training loop structure, the 7 steps (zero_grad → forward → loss → backward → clip → step → scheduler) — is identical to the CPU version.

---

## 6. Mixed Precision Training (AMP)

The GPU training loop we wrote in section 5 already runs on the GPU. But we can squeeze even more performance out by using **mixed precision**: running most computations in `float16` (half precision) instead of `float32` (single precision).

Before the code, there are two hidden ideas to make explicit:

### What does "precision" mean here?

Precision means the numeric format used to store and compute values.

Common formats:

- `float32`: the normal default in PyTorch training
- `float16`: uses less memory and can run faster on modern GPUs

So mixed precision means:

- use `float16` where it is safe and fast
- keep `float32` where extra numeric stability is needed

It is called "mixed" because the training loop uses a mixture of both formats, not only one.

Why does this help?
- `float16` uses 2 bytes per number vs 4 bytes for `float32` → fits twice as many values in VRAM
- Modern NVIDIA GPUs have dedicated **Tensor Cores** that run `float16` matrix multiplications at 2–8× the speed of `float32`
- The catch: `float16` has a limited range. Very small gradients (common in deep networks) can underflow to exactly 0 → `GradScaler` fixes this

### What are `autocast` and `GradScaler`?

AMP has two separate tools, and each solves a different problem:

- `autocast`: chooses lower precision automatically for operations that are usually safe
- `GradScaler`: protects the backward pass from tiny gradients disappearing

So if you ever feel confused by the AMP code, remember:

- forward-pass speed and memory savings come from `autocast`
- backward-pass safety comes from `GradScaler`

### Why not just convert the whole model to `float16` manually?

Because some operations become numerically fragile in pure `float16`. Beginners sometimes imagine AMP means:

```python
model.half()
```

everywhere, but that is not the recommended starting point. AMP is safer because PyTorch keeps sensitive operations in higher precision when needed.

```python
from torch.cuda.amp import autocast, GradScaler

# GradScaler keeps a "loss scale" (a large number, e.g. 65536).
# It multiplies the loss by this scale before backward pass,
# making gradients larger and preventing float16 underflow.
# It automatically reduces the scale if inf/nan gradients appear.
scaler = GradScaler()

for X, y in train_loader:
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()

    # ① autocast context: PyTorch automatically picks float16 for safe ops
    # (matrix multiplies, convolutions) and keeps float32 for sensitive ops
    # (softmax, loss computation) — you don't need to decide manually.
    with autocast():
        output = model(X)         # Matrix multiplies run in float16 on Tensor Cores
        loss   = criterion(output, y)  # Loss kept in float32 for numerical stability

    # ② Multiply loss by scale factor before backward.
    # This makes all gradients proportionally larger → they don't underflow in float16.
    scaler.scale(loss).backward()

    # ③ Before gradient clipping, remove the scale factor from gradients first.
    # clip_grad_norm_ must see the real gradient magnitudes, not the scaled ones.
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # ④ Optimizer step: if unscaled gradients contain inf/nan (overflow),
    # this step is skipped entirely — protects weights from corruption.
    scaler.step(optimizer)

    # ⑤ Adjust scale for next iteration: if step was skipped, reduce scale;
    # if step succeeded, gradually increase scale to maximize gradient precision.
    scaler.update()
```

### AMP in plain language

Think of AMP as a two-layer system:
1. **`autocast()`**: automatically runs math in float16 where safe, keeping float32 only where needed for numerical stability. You don't write float16 anywhere — PyTorch decides.
2. **`GradScaler`**: handles the gradient underflow problem by scaling the loss up before backward and scaling gradients back down before the optimizer step.

The 5-step replace `loss.backward() → optimizer.step()` with the 4-step scaler dance. That's the entire change from a standard training loop.

**Speedup numbers (RTX 3090):**
- ResNet-50 training: float32 → ~350 img/sec, float16 AMP → ~850 img/sec (2.4×)
- Memory: float32 → 12 GB VRAM, float16 AMP → 7 GB VRAM (1.7× saving)

These numbers matter because they translate directly to either training larger models or training the same model faster for the same cost.

---

## 7. GPU Memory Management

### Why do we need separate memory tools on GPU?

Because GPU memory is much smaller than system RAM, and training jobs can run out of it quickly.

So when training on GPU, we often care about three different questions:

- how much memory is being used right now?
- what was the peak usage?
- am I accidentally keeping tensors alive longer than needed?

The code below is mostly about answering those questions and debugging OOM problems.

```python
# Check memory
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
print(f"Peak:      {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# Free cached memory (rarely needed — PyTorch manages it)
torch.cuda.empty_cache()

# Reset peak memory stats (for benchmarking)
torch.cuda.reset_peak_memory_stats()

# Common OOM (Out of Memory) fixes:
# 1. Reduce batch size
# 2. Use mixed precision (AMP)
# 3. Use gradient checkpointing
# 4. Use gradient accumulation (accumulate N batches before optimizer.step)
# 5. Delete tensors you no longer need: del tensor; torch.cuda.empty_cache()

# Memory leak pattern — AVOID:
losses = []
for X, y in loader:
    loss = criterion(model(X), y)
    losses.append(loss)           # ❌ Keeps computation graph in memory!
    losses.append(loss.item())    # ✅ Python float, no graph
```

---

## 8. GPU Timing (Correctly)

### Why is GPU timing tricky?

Because CUDA operations are asynchronous.

That means when Python launches a GPU operation, it often continues immediately without waiting for the GPU to finish. So a normal wall-clock measurement can report only the launch time, not the real execution time.

That is why timing GPU code needs explicit synchronization or CUDA events.

```python
# ❌ WRONG: GPU is async, time.time() doesn't capture GPU time
import time
start = time.time()
output = model(x_gpu)
elapsed = time.time() - start   # This is meaningless!

# ✅ CORRECT: Use CUDA events (synchronizes with GPU)
start_event = torch.cuda.Event(enable_timing=True)
end_event   = torch.cuda.Event(enable_timing=True)

start_event.record()
output = model(x_gpu)
end_event.record()

torch.cuda.synchronize()  # Wait for GPU to finish
elapsed_ms = start_event.elapsed_time(end_event)
print(f"GPU time: {elapsed_ms:.2f} ms")
```

---

## 9. Interview Questions

<details>
<summary><strong>Q1: Why is GPU much faster than CPU for deep learning?</strong></summary>

Deep learning training is dominated by matrix multiplications and convolutions — massively parallelizable operations. A GPU has thousands of smaller cores designed to execute the same instruction on many data elements simultaneously (SIMD). A CPU has few powerful cores optimized for sequential, complex tasks. For a matrix multiply, the GPU executes thousands of multiply-add operations in parallel vs the CPU's sequential execution.
</details>

<details>
<summary><strong>Q2: What happens if you forget to move your data to GPU?</strong></summary>

You get a `RuntimeError: Expected all tensors to be on the same device`. PyTorch operations require all operands to be on the same device. The model's parameters are on GPU, but the input tensor is on CPU — the addition/multiply operations between them fail. Always move both model and data to the same device before any computation.
</details>

<details>
<summary><strong>Q3: What is mixed precision training? What is GradScaler for?</strong></summary>

Mixed precision training uses float16 (half precision) for most computations (reducing memory and increasing speed on Tensor Cores) while keeping float32 for numerically sensitive operations. `GradScaler` addresses float16's limited dynamic range: very small gradients underflow to 0 in fp16. GradScaler multiplies the loss by a scale factor before backward (making gradients larger → no underflow), then divides before the optimizer step (correcting back). If any inf/nan appears, it skips the step and reduces the scale.
</details>

<details>
<summary><strong>Q4: What is pin_memory and non_blocking for?</strong></summary>

`pin_memory=True` in DataLoader allocates tensors in page-locked (pinned) CPU memory, which can be directly DMA-transferred to GPU without an intermediate copy. `non_blocking=True` in `.to(device)` makes the CPU→GPU transfer asynchronous — Python code continues while transfer happens in the background. Together, they enable the GPU to start training on batch N while the CPU loads batch N+1, overlapping computation and data transfer.
</details>

---

## 🔗 References
- [CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [CampusX Video 8](https://www.youtube.com/watch?v=CabHrf9eOVs)
