---
id: 15-industry-standard-pytorch
title: "Video 15: Industry-Standard PyTorch"
sidebar_label: "15 · Industry-Standard PyTorch"
sidebar_position: 15
description: Production-minded PyTorch practices — reproducibility, AMP, torch.compile, checkpointing, DDP, profiling, export, and deployment readiness.
tags: [pytorch, production, amp, torch-compile, ddp, profiling, onnx, reproducibility]
---

# Industry-Standard PyTorch

> **What you'll learn:** How PyTorch code changes when you move from "a notebook that works" to "a training system a team can trust."

---

## 1. What Changes in Real Projects?

Tutorial code usually optimizes for learning. Production code optimizes for:

- reproducibility
- speed per GPU hour
- memory efficiency
- resumability after failures
- observability and debugging
- clean deployment paths

The big mental shift is this:

```text
Notebook success != production readiness
```

If a model trains once but cannot be resumed, profiled, exported, or reproduced, the project is fragile.

## Visual Reference

![Ring all-reduce in distributed training](https://raw.githubusercontent.com/ppadjin/torch-parallel-from-scratch/45d6c85e/media/ring_all_reduce.png)

*Production training systems often synchronize gradients across multiple GPU processes using ring all-reduce. In standard DDP, each GPU usually holds a full replica of the model, while the input batch is sharded across processes. Each process computes local gradients, then the gradients are aggregated efficiently before the optimizer step. Everything else in this chapter — AMP, `torch.compile`, checkpointing — makes that loop faster and more reliable.*

---

## 2. Reproducibility Baseline

PyTorch's official reproducibility notes make an important point: complete reproducibility is not guaranteed across every platform and release, but you can reduce nondeterminism a lot.

```python
import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # More deterministic behavior, often a bit slower.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Raise if a nondeterministic op is used.
    torch.use_deterministic_algorithms(True, warn_only=False)


seed_everything(42)
os.environ["PYTHONHASHSEED"] = "42"
```

### Code walkthrough

- Python, NumPy, and PyTorch each have separate randomness sources, so all three are seeded.
- `cudnn.benchmark=False` avoids algorithm auto-tuning that can change run-to-run behavior.
- deterministic settings are excellent for debugging, but may reduce training throughput.

:::tip
For research and debugging, prefer determinism. For final throughput benchmarking, you may relax determinism after you trust the pipeline.
:::

---

## 3. Mixed Precision Training with `torch.amp`

Automatic mixed precision is now a standard default on CUDA training jobs because it usually improves speed and lowers memory use.

### What is mixed precision?

PyTorch models normally train in `float32`. Mixed precision means we let some operations run in lower precision, usually `float16`, while keeping sensitive operations in higher precision when needed.

So the goal is:

- keep most of the speed and memory benefits of `float16`
- keep enough `float32` to avoid unstable training

### What are the two AMP pieces?

This code uses two helpers, each with a separate job:

- `autocast`: automatically chooses lower precision for many forward-pass operations
- `GradScaler`: scales the loss to protect small gradients during backward

That is the key idea to hold onto before reading the loop:

- `autocast` helps the forward pass
- `GradScaler` helps the backward pass

```python
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"

model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

for inputs, targets in train_loader:
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        logits = model(inputs)
        loss = criterion(logits, targets)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

### Why this is industry-standard

- `autocast` runs fast ops in lower precision where safe.
- `GradScaler` protects small gradients from underflow.
- `set_to_none=True` is a small but common optimization because it avoids writing zeros into every gradient tensor.

---

## 4. Compile the Model with `torch.compile`

In modern PyTorch, `torch.compile` is the first speedup tool to try before jumping into deeper optimization work.

### What does `torch.compile` actually do?

Normally, PyTorch eagerly runs your model line by line. `torch.compile` tries to turn parts of that Python-level execution into a more optimized compiled graph.

Beginner-friendly mental model:

- normal PyTorch: "run this operation now, then the next one, then the next one"
- compiled PyTorch: "analyze this chunk of work and generate a faster execution plan"

You still write normal PyTorch code. `torch.compile` is an optimization wrapper around the model, not a new modeling API.

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
).to(device)
model = torch.compile(model)

for step, (inputs, targets) in enumerate(train_loader):
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
```

### What to expect

- the first few iterations are usually slower because compilation has overhead
- later iterations may become much faster
- Python-heavy control flow can create graph breaks and reduce gains
- start by compiling the model only, then benchmark before and after

---

## 5. Gradient Accumulation, Clipping, and Memory-Saving Tricks

When the ideal batch size does not fit in memory, teams often simulate it with gradient accumulation.

### What is gradient accumulation?

Usually, one batch follows this pattern:

1. forward pass
2. backward pass
3. optimizer step

With gradient accumulation, we delay step 3.

Instead of updating weights after every mini-batch, we:

- run several small batches
- accumulate their gradients in `.grad`
- do one optimizer step after enough batches have been combined

This is useful when the batch size you want does not fit in GPU memory.

Example:

- desired batch size: 256
- fits in memory: only 64
- solution: run 4 batches of 64 and update once

That is why the code divides the loss by `accum_steps`: we want the accumulated gradient to behave like one larger batch, not four full-strength updates added together.

```python
accum_steps = 4
optimizer.zero_grad(set_to_none=True)

for step, (inputs, targets) in enumerate(train_loader, start=1):
    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        logits = model(inputs)
        loss = criterion(logits, targets) / accum_steps

    scaler.scale(loss).backward()

    if step % accum_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

### Activation checkpointing

If activation memory is the bottleneck, checkpoint part of the forward pass and recompute it during backward.

### What are activations, and why checkpoint them?

During the forward pass, PyTorch saves intermediate tensors because backward needs them later. Those saved intermediates are called **activations**.

In large models, activations can use more memory than the parameters themselves.

Checkpointing says:

- do not save every intermediate tensor
- save less memory now
- recompute some forward work later during backward

So checkpointing is a memory-for-compute tradeoff.

```python
from torch.utils.checkpoint import checkpoint


def forward(self, x):
    x = checkpoint(self.block1, x, use_reentrant=False)
    x = checkpoint(self.block2, x, use_reentrant=False)
    return self.head(x)
```

### Tradeoff

- gradient accumulation trades time for larger effective batch size
- activation checkpointing trades compute for lower memory use
- clipping protects training when gradients spike

---

## 6. Save Checkpoints You Can Actually Resume

Saving only weights is fine for inference. Training jobs need resumable checkpoints.

### What is a checkpoint?

A checkpoint is a saved snapshot of training state.

For inference-only use, saving model weights may be enough.

For resuming training, that is not enough because training also depends on things like:

- optimizer state
- scheduler state
- current epoch
- best validation metric so far

If you restore only model weights, the model remembers what it learned, but the training process forgets where it was.

```python
def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_loss):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt["epoch"], ckpt["best_val_loss"]
```

### What teams usually save

- current epoch or global step
- model weights
- optimizer and scheduler state
- best validation metric
- tokenizer, vocabulary, or label mapping if inference depends on them

---

## 7. Scale to Multiple GPUs with DDP

For multi-GPU training, prefer **DistributedDataParallel (DDP)** over the older `DataParallel`.

### Launch command

```bash
torchrun --standalone --nproc_per_node=4 train.py
```

### Minimal DDP skeleton

Before reading this code, there is one term worth slowing down on:

### What is `DistributedSampler`?

In normal single-GPU training, one `DataLoader` can iterate over the whole dataset in a shuffled order.

In Distributed Data Parallel (DDP), multiple GPU processes are training at the same time. If every process read the full dataset in the same order, then:

- GPU 0 would see sample 0
- GPU 1 would also see sample 0
- GPU 2 would also see sample 0

That would waste computation because all GPUs would be doing duplicate work on the same samples.

`DistributedSampler` solves that problem. Its job is to split the dataset across processes so each GPU gets a different slice of data for the current epoch.

In beginner-friendly terms:

- `sampler` decides which indices a `DataLoader` will read
- `DistributedSampler` makes each process read a different subset of those indices

Why call `sampler.set_epoch(epoch)`?

Because we still want shuffling to change from epoch to epoch. Calling `set_epoch(epoch)` gives each epoch a new shuffle order while keeping all processes coordinated correctly.

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    X = torch.randn(2048, 20)
    y = torch.randint(0, 2, (2048,))
    dataset = TensorDataset(X, y)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)

    model = nn.Sequential(
        nn.Linear(20, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    ).to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 5

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # important for proper shuffling across processes
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()
```

### Key idea

Each GPU runs its own process, sees a different shard of the batch, computes gradients locally, and DDP synchronizes those gradients before the optimizer step.

:::info
When a single model no longer fits comfortably on each GPU, the next topics to study are FSDP and tensor/model parallelism.
:::

---

## 8. Profile Before You Guess

Performance problems often come from the wrong place: data loading, CPU transforms, synchronization, or a few expensive operators.

### What is profiling?

Profiling means measuring where time and memory are actually going.

This matters because performance intuition is often wrong. A slow training job might look like "the model is too heavy," but the real bottleneck might be:

- slow data loading
- CPU preprocessing
- GPU idle time
- one unexpectedly expensive operator

So profiling is the difference between guessing and measuring.

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("train_step"):
        inputs, targets = next(iter(train_loader))
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### What profiling tells you

- which ops take the most time
- which ops allocate the most memory
- whether you are CPU-bound or GPU-bound
- whether `torch.compile`, AMP, or input pipeline changes are worth doing

---

## 9. Export for Inference

If deployment needs a framework-neutral format, ONNX is still a common path.

### What does "export" mean here?

Training usually happens inside PyTorch, but deployment may happen somewhere else:

- another runtime
- another programming language
- a production inference server

Exporting means converting the trained model into a portable representation that another runtime can execute.

So this section is not about improving training. It is about packaging the model for use after training.

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * 224 * 224, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
).eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

onnx_program = torch.onnx.export(
    model,
    example_inputs,
    dynamo=True,
)

onnx_program.save("model.onnx")
```

### Export checklist

- call `model.eval()` first
- use representative example inputs
- verify the exported model in the target runtime
- watch for dynamic shape and Python control-flow issues

---

## 10. A Practical Team Checklist

Before calling a PyTorch training pipeline "ready", ask:

- Can I rerun it with the same seed and get close to the same result?
- Can I resume from a checkpoint after interruption?
- Do I know whether training is data-bound, compute-bound, or memory-bound?
- Have I tried AMP and `torch.compile` and measured the result?
- If using multiple GPUs, am I using DDP correctly?
- Can I export or package the trained model for inference?
- Are label maps, tokenizers, and transforms versioned alongside the checkpoint?

---

## 11. Where to Go Next

If you want to keep leveling up after this module:

1. Learn DDP well.
2. Learn profiling well.
3. Add AMP and `torch.compile` to your default toolkit.
4. Study FSDP if your models are too large for per-GPU memory.
5. Learn export and inference validation, not just training.

---

## 🔗 References

- [PyTorch Reproducibility Notes](https://docs.pytorch.org/docs/stable/notes/randomness)
- [Automatic Mixed Precision (`torch.amp`)](https://docs.pytorch.org/docs/stable/amp.html)
- [Introduction to `torch.compile`](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Getting Started with Distributed Data Parallel](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [What is Distributed Data Parallel (DDP)](https://docs.pytorch.org/tutorials/beginner/ddp_series_theory.html)
- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [Export a PyTorch Model to ONNX](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
- [Activation Checkpointing](https://docs.pytorch.org/docs/stable/checkpoint)
