---
id: 01-introduction-to-pytorch
title: "Video 1: Introduction to PyTorch"
sidebar_label: "01 · Introduction to PyTorch"
sidebar_position: 1
description: What is PyTorch, why it matters, how it compares to other frameworks, and environment setup.
tags: [pytorch, introduction, campusx, deep-learning, setup]
---

# Introduction to PyTorch
**📺 CampusX — Practical Deep Learning using PyTorch | Video 1**

> **What you'll learn:** What PyTorch is, why the deep learning community loves it, how it compares to TensorFlow/NumPy, and how to install and verify your environment.

---

## 1. What is Deep Learning? (Quick Recap)

Before diving into PyTorch, it helps to understand exactly **where** it sits in the broader AI landscape — because this positioning explains why the framework was built the way it was.

```
Artificial Intelligence
    └── Machine Learning
            └── Deep Learning   ← PyTorch lives here
                    ├── Computer Vision
                    ├── NLP / LLMs
                    ├── Speech Recognition
                    └── Reinforcement Learning
```

Deep Learning uses **neural networks** with many layers to learn patterns from large amounts of data. Instead of hand-crafted features (which a human engineer designs), the model learns its own internal representations during training. This self-learning from raw data is what makes deep learning so powerful — and also what makes the engineering challenge substantial.

## Visual Reference

![Two-hidden-layer feedforward neural network](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/images/nn2.png)

*A deep neural network with two hidden layers. Each circle is a neuron; each line is a learned weight. The input layer receives raw features, hidden layers transform them into increasingly abstract representations, and the output layer produces the final prediction. PyTorch provides the building blocks — tensors, autograd, and `nn.Module` — to construct and train networks exactly like this.*

**Why do we need a framework like PyTorch?**

Building a neural network from scratch — without any framework — would require:
- Efficient tensor math (matrix multiplications running on GPU)
- Automatic gradient computation (applying the chain rule through thousands of operations)
- GPU memory management (allocating and freeing memory intelligently)
- Parallel data loading (feeding the GPU fast enough so it stays busy)

Doing all this in raw NumPy or plain C++ would be thousands of lines of error-prone, hard-to-debug code. PyTorch handles all of it cleanly, letting you focus on the model design rather than the infrastructure.

---

## 2. What is PyTorch?

PyTorch is an **open-source deep learning framework** created by **Meta AI (Facebook)**, released in 2016. It wraps the earlier Torch library (written in Lua) in a clean Python API that feels natural to anyone who already knows Python.

At its heart, PyTorch provides two foundational capabilities that everything else builds on:

| Feature | What it means |
|---|---|
| **Tensor computation** | Like NumPy, but runs on GPU |
| **Automatic differentiation (Autograd)** | Computes gradients automatically for any computation |

Everything else — CNNs, RNNs, Transformers, optimizers, data loaders — is built on top of these two primitives. Understanding tensors and autograd deeply means you can understand and debug any PyTorch model.

### Why PyTorch was created

The transcript explains PyTorch as the practical merger of two worlds:

- **Torch** gave powerful deep-learning capabilities (GPU-accelerated tensors, automatic differentiation)
- **Python** gave a simpler, expressive language and a rich scientific ecosystem (NumPy, SciPy, Matplotlib)

That is why PyTorch is often described as the "marriage" of Torch-style deep learning with Python usability. This also explains one of its early strengths: it fit naturally into existing Python workflows that researchers and data scientists already knew, without requiring them to learn an entirely new mental model.

### PyTorch matters right now

One point emphasized early is that PyTorch is not just "another ML library". It became the **default working language** for a huge part of modern deep learning, especially:

- Open-source LLM work (LLaMA, Mistral, Falcon, and most community fine-tunes)
- Research codebases (the majority of ML research papers on arXiv)
- Fine-tuning workflows (Hugging Face, PEFT, LoRA)
- Custom neural-network experimentation (architecture search, novel training tricks)

If your long-term goal is **Generative AI, LLMs, computer vision, or deep learning engineering**, PyTorch is one of the most useful foundations to build first — because later libraries often still expose PyTorch-style tensors, modules, losses, and training loops under the hood.

### PyTorch in the Real World

Models you interact with daily are trained with PyTorch:

| Model | Organization | Framework |
|---|---|---|
| GPT-4 | OpenAI | PyTorch |
| LLaMA 3 | Meta | PyTorch |
| Stable Diffusion | Stability AI | PyTorch |
| DALL-E | OpenAI | PyTorch |
| Claude | Anthropic | PyTorch |
| Gemini | Google | JAX / TF |

PyTorch dominates in research and open-source practice: most modern tutorials, research repos, and foundation-model training stacks are written with PyTorch-style APIs.

---

## 3. PyTorch vs TensorFlow vs NumPy

### PyTorch vs NumPy

NumPy is a CPU-only array computation library with no notion of gradients. PyTorch tensors are like NumPy arrays but with **two critical additions** that make neural network training possible:

```python
import numpy as np
import torch

# ── NumPy: CPU only, no gradients ─────────────────────────────
a = np.array([1.0, 2.0, 3.0])
b = a * 2
print(b)   # [2. 4. 6.]
# You can compute values, but you can't ask "what is db/da?"

# ── PyTorch: GPU capable, tracks gradients ─────────────────────
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# requires_grad=True tells PyTorch to record every operation on x
# so it can later compute derivatives through those operations

y = x * 2          # Operation is recorded in a computation graph
y.sum().backward() # Backpropagate: compute dy/dx for each element

print(x.grad)   # tensor([2., 2., 2.])
# dy/dx = d(2x)/dx = 2 for each element — computed automatically!
```

The key difference: NumPy is for general numerical computation; PyTorch is built specifically for gradient-based optimization (i.e., training neural networks).

### PyTorch vs TensorFlow

| Feature | PyTorch | TensorFlow 1.x | TensorFlow 2.x (Keras) |
|---|---|---|---|
| Graph type | **Dynamic** (define-by-run) | Static (define-then-run) | Eager + Static |
| Debugging | Normal Python debugger | Special TF tools | Improved |
| API feel | Pythonic, intuitive | Verbose, boilerplate-heavy | Better |
| Research use | **Dominant** | Declining | Present |
| Mobile deploy | TorchScript, ONNX | TFLite | TFLite |
| Learning curve | Gentler | Steep | Moderate |

**Key insight:** PyTorch's dynamic graph means you write Python naturally. Your model IS Python — no separate compilation step, no sessions, no placeholders. This matters enormously for experimentation speed.

### Why dynamic computation graphs mattered so much

A **computation graph** is a graph-style representation of mathematical operations. For a neural network, every matrix multiply, addition, activation function call, and loss computation can be stored as a node in this graph.

Older static-graph frameworks like TensorFlow 1.x forced you to **define the graph first** and **run it later**. This created a two-stage workflow that was hard to debug and couldn't easily support variable-length inputs or conditional computation:

```python
# TensorFlow 1.x style (old way — compile then run)
# Stage 1: Define the graph (no computation happens yet)
x = tf.placeholder(tf.float32)   # x is just a symbol, not a number
y = x * 2
# Stage 2: Run the graph in a session
with tf.Session() as sess:
    result = sess.run(y, feed_dict={x: 3.0})   # Now x=3.0 is substituted
print(result)   # 6.0

# PyTorch style — just Python! No stages, no sessions, no placeholders.
x = torch.tensor(3.0)
y = x * 2
print(y)   # tensor(6.) — computed immediately, like regular Python
```

PyTorch made the graph **dynamic**: it builds as your Python code runs, can change from one forward pass to the next, and is fully debuggable with standard Python tools like `pdb` and `print`. That flexibility is a big reason PyTorch became so popular for research and custom model development.

---

## 4. The PyTorch Ecosystem

PyTorch is not just one library — it is an ecosystem of tools that cover the full model lifecycle from data loading to deployment:

```
torch (core)
├── torch.Tensor          ← N-D arrays with GPU + gradient support
├── torch.autograd        ← Automatic differentiation engine
├── torch.nn              ← Neural network building blocks (layers, loss fns)
├── torch.optim           ← Optimization algorithms (SGD, Adam, AdamW)
├── torch.utils.data      ← Dataset and DataLoader
│
├── torchvision           ← Images: datasets, pretrained models, transforms
├── torchaudio            ← Audio processing
├── torchtext             ← NLP utilities
│
├── TorchScript           ← Export models for production (no Python needed)
├── ONNX export           ← Cross-framework model format
└── TorchServe            ← Model serving in production
```

Understanding the hierarchy is important: `torch.Tensor` and `torch.autograd` are the bedrock. Everything above them — layers, losses, optimizers, data loaders — is built on top of these two primitives. If you understand tensors and autograd, you can read and understand any PyTorch code.

## 4.1 A Mental Model for the Whole Library

When people start PyTorch, the API surface can feel large and overwhelming. A simpler way to think about it is as a pipeline of transformations, where each part of the library owns exactly one stage:

```text
Data    -> torch.Tensor -> nn.Module -> Loss -> autograd -> Optimizer
images     numbers         model        score    gradients   weight update
text       batches         layers       error    chain rule  learning step
```

This mental model maps directly to how you write training code:
- `torch` stores and moves data (the numbers the model learns from)
- `torch.nn` defines learnable computation (the model itself)
- `torch.autograd` computes gradients (how each weight contributed to the error)
- `torch.optim` updates parameters (adjusting weights to reduce the error)
- `torch.utils.data` feeds mini-batches efficiently (keeps GPU fed with data)

## 4.2 The Learning Roadmap

The playlist builds understanding in a sequence where each lesson unlocks the next:

```text
Intro -> Tensors -> Computation Graph / Autograd -> Training Pipeline
     -> nn.Module + torch.optim -> Dataset/DataLoader
     -> ANN -> GPU Training -> Optimization -> Hyperparameter Tuning
     -> CNN -> Transfer Learning -> RNN -> LSTM
```

This roadmap matters because PyTorch is easier to learn when you understand the dependency chain:
- Tensors are the data structure everything is built on
- Autograd explains where gradients come from
- The training pipeline combines forward pass, loss, backward, and weight update
- `nn.Module` and `torch.optim` automate the tedious manual pieces
- `Dataset` and `DataLoader` make the training loop scalable to real data

---

## 5. Installation & Environment Setup

### Step 1: Create a virtual environment (recommended)

Isolating your PyTorch environment prevents conflicts with other Python projects:

```bash
# Using conda (recommended for deep learning — handles GPU drivers cleanly)
conda create -n pytorch-env python=3.10
conda activate pytorch-env

# Or using venv (lighter weight)
python -m venv pytorch-env
source pytorch-env/bin/activate   # Linux/Mac
pytorch-env\Scripts\activate      # Windows
```

### Step 2: Install PyTorch

The exact install command depends on your hardware. Visit **https://pytorch.org/get-started/locally/** for the selector. Common cases:

```bash
# CPU only (works everywhere, slower for large models)
pip install torch torchvision torchaudio

# NVIDIA GPU — CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# NVIDIA GPU — CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (M1/M2/M3) — PyTorch uses MPS (Metal Performance Shaders) backend
pip install torch torchvision torchaudio
# MPS is included automatically when you install on macOS with Apple Silicon
```

### Step 3: Verify Installation

```python
import torch

# Check the PyTorch version you installed
print(torch.__version__)            # e.g., 2.2.0

# Check GPU availability
# This is the first thing to verify before any GPU training
print(torch.cuda.is_available())    # True if NVIDIA GPU + CUDA is set up
print(torch.cuda.device_count())    # Number of GPUs available
print(torch.cuda.get_device_name(0))  # e.g., "NVIDIA GeForce RTX 3080"

# Apple Silicon check
print(torch.backends.mps.is_available())  # True on M1/M2/M3 Mac
```

### Step 4: Choosing your device

The best practice is to write **device-agnostic code** — code that automatically uses a GPU if one is available and falls back to CPU otherwise. This means the same code runs correctly on a laptop (CPU), a cloud instance (CUDA GPU), or a MacBook (MPS):

```python
import torch
import torch.nn as nn

# Single line that picks the best available device
# Checks CUDA (NVIDIA GPU) first, then MPS (Apple Silicon), then falls back to CPU
device = (
    "cuda"  if torch.cuda.is_available()  else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

# Move both model and data to the same device before computation
model = nn.Linear(3, 1).to(device)  # Move all model parameters to device
x = x.to(device)                    # Move input tensor to device
```

This pattern — defining `device` once at the top of your script and using `.to(device)` everywhere — is considered the correct way to write portable PyTorch code.

---

## 6. First PyTorch Code

Let's write a minimal but complete PyTorch program that demonstrates tensors, operations, and device movement. Read each line carefully because these patterns appear in every PyTorch project:

### What should a beginner notice in this first example?

This tiny script is doing four core things:

1. creating a tensor
2. inspecting its metadata
3. doing basic math
4. moving it to a GPU if one exists

That may look too simple, but these are the exact building blocks used everywhere later:

- models are collections of tensors
- neural-network math is mostly tensor math
- debugging often starts with shape, dtype, and device
- GPU training is mostly "same code, different device"

So the goal of this section is not to solve a real ML problem yet. It is to make the basic PyTorch workflow feel normal before larger examples appear.

```python
import torch

# Create a 2D tensor (a matrix) from Python lists
# dtype=torch.float32 is the standard for neural network math
# Most PyTorch operations and model weights expect float32 by default
x = torch.tensor([[1, 2, 3],
                   [4, 5, 6]], dtype=torch.float32)

# Inspect the tensor — always check these three before using a tensor
print("Shape:", x.shape)       # torch.Size([2, 3]) — 2 rows, 3 columns
print("Dtype:", x.dtype)       # torch.float32 — the numeric type
print("Device:", x.device)     # cpu — where computations happen

# Element-wise operations — every arithmetic op applies to each element
print(x + 10)   # Adds 10 to every element
print(x * 2)    # Multiplies every element by 2

# Matrix multiplication — the fundamental operation of neural networks
# x has shape (2,3), x.T has shape (3,2), result is (2,2)
# This is what happens inside every nn.Linear layer!
print(x @ x.T)   # Matrix multiply: (2,3) × (3,2) = (2,2)

# Move to GPU if one is available
# All subsequent operations on x will run on the GPU
if torch.cuda.is_available():
    x = x.to("cuda")
    print("Now on:", x.device)   # cuda:0 — first GPU
```

### What each line teaches

- `torch.tensor(...)` creates a typed tensor from plain Python values. The `dtype` argument controls the numeric precision.
- `x.shape`, `x.dtype`, and `x.device` are the **first three things** to inspect whenever a tensor behaves unexpectedly — most bugs come down to a wrong shape, a wrong type, or data being on the wrong device.
- `x + 10` and `x * 2` are element-wise operations, while `x @ x.T` is matrix multiplication — this distinction matters constantly in neural network code.
- `x.to("cuda")` does not change the math — it changes **where** the math runs. The result is numerically identical but computed on the GPU.

---

## 7. How PyTorch Training Works (Big Picture)

Before diving into code across the next 13 videos, it helps to see the complete training loop in one place. Every single PyTorch training project — whether it's a simple regression model or a 70-billion-parameter LLM — follows exactly this 7-step cycle:

```
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING LOOP                             │
│                                                             │
│  for each epoch:                                            │
│    for each mini-batch from DataLoader:                     │
│      1. Move batch to device (GPU)                          │
│      2. optimizer.zero_grad()  ← reset accumulated grads   │
│      3. output = model(input)  ← FORWARD PASS              │
│      4. loss = criterion(output, target)  ← measure error  │
│      5. loss.backward()  ← BACKWARD PASS (autograd)        │
│      6. optimizer.step()  ← UPDATE WEIGHTS                 │
│      7. Log metrics (loss, accuracy, etc.)                  │
└─────────────────────────────────────────────────────────────┘
```

- Steps 1–2 are **setup** for the batch — move data to the right device and clear any leftover gradients from the previous step.
- Steps 3–4 are the **forward pass** — run the model and measure how wrong its predictions are.
- Step 5 is the **backward pass** — autograd traverses the computation graph and computes how each weight contributed to the error.
- Step 6 is the **weight update** — the optimizer uses the gradients to nudge weights in the direction that reduces the loss.
- Step 7 is **monitoring** — without logging, you can't tell if training is working.

You will implement each piece of this loop from scratch across the first 9 videos. Each video adds one piece of the puzzle until the whole picture is clear.

---

## 8. Key Concepts to Understand Before Next Video

| Concept | Simple explanation |
|---|---|
| **Tensor** | A multi-dimensional array (the data container of all PyTorch computations) |
| **Gradient** | The slope of the loss with respect to a weight — tells us which direction to adjust the weight |
| **Loss** | A single number measuring how wrong the model's predictions are |
| **Backward pass** | Computing all gradients by applying the chain rule through the computation graph |
| **Optimizer** | The algorithm that uses gradients to update weights (e.g., SGD, Adam) |
| **Epoch** | One full pass through the entire training dataset |
| **Batch** | A small subset of data processed in one step (e.g., 32 images at a time) |

---

## 9. Interview Questions

<details>
<summary><strong>Q1: What is PyTorch and who developed it?</strong></summary>

PyTorch is an open-source machine learning framework developed by **Meta AI (Facebook Research)**, released in 2016. It provides tensor computation with GPU acceleration and automatic differentiation for building and training neural networks. It is built on top of the earlier Torch library (Lua) with a Python-first API. The name reflects its heritage: Torch capabilities wrapped in a Python interface.
</details>

<details>
<summary><strong>Q2: What is the key difference between PyTorch and TensorFlow?</strong></summary>

The fundamental difference is the **computation graph style**:
- PyTorch uses **dynamic graphs** (define-by-run / eager execution): the graph is built on-the-fly as Python code executes. Each forward pass can be structurally different. You can use any Python control flow (if statements, loops, recursion) inside a model.
- TensorFlow 1.x used **static graphs** (define-then-run): the graph is fully defined first, then executed in a session. Python control flow doesn't exist inside the graph — you need TF ops like `tf.cond` and `tf.while_loop`.

Dynamic graphs make PyTorch much easier to debug (use standard Python debugger), support variable-length inputs naturally (critical for NLP), and enable complex architectures like recursive networks and dynamic computation.
</details>

<details>
<summary><strong>Q3: What are the two main capabilities of PyTorch?</strong></summary>

1. **Tensor computation**: Multi-dimensional array operations that can run on GPU — like NumPy but GPU-accelerated and integrated with gradients.
2. **Automatic differentiation (Autograd)**: Automatically computes gradients of any computation that involves `requires_grad=True` tensors, enabling backpropagation without manual derivative formulas. Every layer, loss, and optimizer in PyTorch is built on these two capabilities.
</details>

<details>
<summary><strong>Q4: Why is PyTorch preferred in research?</strong></summary>

- **Dynamic graphs** support any Python control flow inside models (loops, conditionals, recursion) — this is essential for architectures that vary per input, like tree-structured models or dynamic attention
- **Pythonic API** reduces boilerplate and is easy to read/debug with standard Python tools
- **Rich ecosystem** (Hugging Face Transformers, PyTorch Lightning, etc. all use PyTorch natively)
- **Speed of experimentation**: change architecture on-the-fly without recompiling graphs
- Over 70% of ML research papers use PyTorch as their implementation framework
</details>

<details>
<summary><strong>Q5: What is the difference between a CPU and GPU in the context of deep learning?</strong></summary>

- **CPU** (Central Processing Unit): few powerful cores (4–64), optimized for sequential tasks, complex branching, and mixed workloads. Good for data preprocessing, Python logic.
- **GPU** (Graphics Processing Unit): thousands of smaller cores (3000–16000+), optimized for massively parallel math operations. Training a neural network = billions of floating-point multiplications, mostly independent → GPU parallelizes them simultaneously → 10–100× faster than CPU.
- **In PyTorch**: move tensors and models to GPU with `.to("cuda")`. All operations on that tensor automatically run on the GPU. The math is identical — only the execution hardware changes.
</details>

---

## 🔗 References
- [PyTorch Official Site](https://pytorch.org)
- [PyTorch Docs](https://pytorch.org/docs/stable/)
- [CampusX Video 1](https://www.youtube.com/watch?v=QZsguRbcOBM)
