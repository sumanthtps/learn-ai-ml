---
id: 03-autograd
title: "Video 3: PyTorch Autograd"
sidebar_label: "03 · Autograd"
sidebar_position: 3
description:
  How PyTorch computes gradients automatically — computation graphs, backward
  pass, and gradient control.
tags: [pytorch, autograd, gradients, backpropagation, chain-rule, campusx]
---

# PyTorch Autograd

**📺 CampusX — Practical Deep Learning using PyTorch | Video 3**

> **What you'll learn:** How PyTorch automatically computes gradients using dynamic computation graphs and the chain rule — the core engine behind all neural network training.

---

## 1. Why Do We Need Gradients?

Before writing any autograd code, it's important to understand the problem it solves. Training a neural network is fundamentally an **optimization problem**: we want to find a set of weights `w` that minimizes the loss function `L(w)`.

The update rule for gradient descent is:

```
w_new = w_old - learning_rate × ∂L/∂w
```

The term `∂L/∂w` is the **gradient** — a vector that tells us:
- **Sign**: which direction to move the weight (positive gradient → decrease the weight)
- **Magnitude**: how large a step to take (steeper slope → larger gradient → larger update)

For a network with millions of parameters arranged in dozens of layers, computing all these gradients by hand is impossible — and recomputing them every time you change the architecture would be impractical. **Autograd** computes all gradients automatically, for any computation you write.

### Neural networks are nested functions

This is the key insight: even a tiny neural network is a **composition of functions**:

```text
input x
  → z = W·x + b          (linear transformation)
  → ŷ = activation(z)    (non-linearity)
  → L = loss(ŷ, y)       (compare to true label)
```

A deeper network is just a much longer chain of such compositions. The gradient of the loss with respect to any weight requires applying the chain rule through every layer between that weight and the loss. Autograd automates exactly this chain rule computation.

---

## 2. The Chain Rule — From Math to Code

The chain rule is the mathematical foundation of backpropagation. For a composition of functions `L = f(g(h(x)))`:

```
∂L/∂x = (∂L/∂f) × (∂f/∂g) × (∂g/∂h) × (∂h/∂x)
```

Each term is the local derivative of one function with respect to its input. Autograd computes each local derivative at each node in the network and multiplies them together following this rule.

## Visual Reference

![PyTorch autograd computation graph — forward and backward pass](https://datahacker.rs/wp-content/uploads/2021/01/54-1-1024x490.jpg)

The image above shows the key idea: autograd stores the **forward computation graph**, then walks through it in reverse to compute derivatives for every dependency. The forward pass builds the graph; `backward()` traverses it.

### Step-by-step example by hand

Let's trace the chain rule on a simple computation before letting PyTorch do it automatically:

```
x = 3
y = x²         → dy/dx = 2x = 6
z = y + 5      → dz/dy = 1
L = z

By chain rule:
dL/dx = (dL/dz) × (dz/dy) × (dy/dx)
      = 1 × 1 × 6 = 6
```

Now let PyTorch compute the exact same gradient:

```python
import torch

# requires_grad=True tells PyTorch: "x is a learnable quantity.
# Record every operation involving x so we can differentiate through them."
x = torch.tensor(3.0, requires_grad=True)

y = x ** 2         # y = x²    — PyTorch records this operation
z = y + 5          # z = x² + 5 — PyTorch records this too

# backward() traverses the recorded graph in reverse,
# applying the chain rule at each step, and deposits the final gradient into x.grad
z.backward()

print(x.grad)      # tensor(6.)  ← dz/dx = 2x = 2×3 = 6  ✅
```

**What each line does:**
- `requires_grad=True` opts this tensor into the computation graph. Every operation on `x` will be recorded.
- `y = x ** 2` adds a "PowBackward" node to the graph. PyTorch knows the local derivative of `x**2` is `2x`.
- `z = y + 5` adds an "AddBackward" node. The local derivative of `y+5` with respect to `y` is `1`.
- `z.backward()` triggers: starting from `z`, multiply local derivatives backward through the graph, accumulating into `x.grad`.
- `x.grad` now holds `6.0` — the result of the chain rule applied automatically.

---

## 3. The Computation Graph

Every time you perform an operation on a tensor that has `requires_grad=True`, PyTorch silently adds a **node** to a directed acyclic graph (DAG). This graph records what was computed so it can be traversed in reverse during `backward()`:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2         # PowBackward node is added
z = y * 3          # MulBackward node is added
L = z.sum()        # SumBackward node is added

# Each tensor stores a reference to the function that created it
print(L.grad_fn)                      # SumBackward0
print(L.grad_fn.next_functions)        # ((MulBackward0, 0),)
print(L.grad_fn.next_functions[0][0].next_functions)  # ((PowBackward0, 0),)
```

Visualized as a graph:

```
x (leaf) → [Pow ×²] → y → [Mul ×3] → z → [Sum] → L
                                              ↑
                                        .backward() starts here
                                        and traverses back toward x
```

**Key properties of the graph:**
- Built **dynamically** — each Python line adds a node as it executes
- **Destroyed after backward** by default (to save memory — the graph is single-use)
- Can contain Python `if`, `for`, `while` — the graph reflects the actual execution path, not a static template

This dynamic nature is what makes PyTorch's autograd powerful: the graph is just a record of what actually ran, so any Python computation can be differentiated through.

---

## 4. `requires_grad` — Who Participates in the Graph?

Not every tensor participates in gradient tracking. Only tensors explicitly marked with `requires_grad=True` (or derived from such tensors) are tracked:

```python
# ── Explicitly tracked tensor ──────────────────────────────────
a = torch.tensor(2.0, requires_grad=True)   # Tracked leaf tensor

# ── Untracked tensor ───────────────────────────────────────────
b = torch.tensor(3.0)                        # NOT tracked (default: False)

# ── Derived tensors inherit tracking from their inputs ─────────
c = a * b
print(c.requires_grad)    # True — because a is tracked, c inherits grad-tracking

d = b * b
print(d.requires_grad)    # False — neither parent is tracked

# ── Model parameters are always tracked ───────────────────────
import torch.nn as nn
linear = nn.Linear(3, 2)
print(linear.weight.requires_grad)   # True — always!
print(linear.bias.requires_grad)     # True — always!
```

**Leaf tensors are the ones that actually receive gradients:**

In the computation graph, **leaf tensors** are those created directly by the user (not by operations on other tensors). Model weights and biases are always leaves. During backward:
- Gradients flow from the loss backward through all non-leaf intermediate tensors
- Gradients are **stored** only in leaf tensors (i.e., `.grad` is populated only for leaves)
- The optimizer then reads these `.grad` values and updates the leaf tensors (the weights)

---

## 5. `.backward()` — The Backward Pass

Calling `.backward()` on a scalar output triggers the full gradient computation:

### What does `.backward()` actually mean in plain language?

It means:

- start from the final result, usually the loss
- move backward through the recorded computation graph
- compute how much each input contributed to that result

In training, we usually care about:

- how much each model weight affected the loss

Those gradient values are what the optimizer uses next.

So `.backward()` is not a mysterious training ritual. It is the step that answers:

> "How should each parameter change to reduce the loss?"

```python
# ── Case 1: Scalar output (the usual training case) ───────────
x = torch.tensor([2.0, 3.0], requires_grad=True)
loss = (x ** 2).sum()   # loss = x[0]² + x[1]²  — a scalar

loss.backward()          # Computes dloss/dx for each element

print(x.grad)   # tensor([4., 6.])
# dloss/dx[0] = 2*x[0] = 2*2 = 4
# dloss/dx[1] = 2*x[1] = 2*3 = 6

# ── Case 2: Non-scalar output (less common) ────────────────────
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2   # y is a vector, not a scalar

# y.backward()   # ❌ RuntimeError: grad can only be implicitly created for scalar outputs

# Option A: Reduce to scalar first (the standard approach in training)
# This is what CrossEntropyLoss, MSELoss, etc. do internally
y.sum().backward()
print(x.grad)   # tensor([2., 4., 6.])  — dloss/dx = 2x for each element

# Option B: Pass an external gradient (vector-Jacobian product)
x2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y2 = x2 ** 2
# Passing torch.ones(3) is equivalent to summing and doing backward
y2.backward(torch.ones(3))
print(x2.grad)   # tensor([2., 4., 6.])
```

### When inputs are vectors, autograd returns partial derivatives

With a vector input, the gradient is really a set of **partial derivatives** — one for each input component:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).mean()   # y = (x₁² + x₂² + x₃²) / 3

y.backward()
print(x.grad)   # tensor([0.6667, 1.3333, 2.0000])

# Manual verification:
# ∂y/∂x₁ = 2x₁/3 = 2(1)/3 = 0.6667  ✅
# ∂y/∂x₂ = 2x₂/3 = 2(2)/3 = 1.3333  ✅
# ∂y/∂x₃ = 2x₃/3 = 2(3)/3 = 2.0000  ✅
```

---

## 6. Gradient Accumulation — A Critical Gotcha

Gradients **accumulate by default** in PyTorch. This is an intentional design choice (it enables gradient accumulation training tricks), but it is one of the most common bugs for beginners:

```python
x = torch.tensor(2.0, requires_grad=True)

for i in range(3):
    y = x ** 2
    y.backward()   # Each call ADDS to x.grad, doesn't replace it
    print(f"Pass {i+1}: x.grad = {x.grad}")

# Pass 1: x.grad = tensor(4.)
# Pass 2: x.grad = tensor(8.)   ← 4 + 4, accumulated!
# Pass 3: x.grad = tensor(12.)  ← 4 + 4 + 4, accumulated!

# ✅ FIX: Zero out the gradient before each backward call
x = torch.tensor(2.0, requires_grad=True)
for i in range(3):
    if x.grad is not None:  # First iteration: grad is None, not zero
        x.grad.zero_()      # In-place reset to zero
    y = x ** 2
    y.backward()
    print(f"Pass {i+1}: x.grad = {x.grad}")
# All show: tensor(4.)  ✅
```

This is exactly why every training loop calls `optimizer.zero_grad()` **before** `loss.backward()`. The optimizer's `zero_grad()` method calls `.zero_()` on every parameter's `.grad` attribute, resetting them so the new backward pass computes fresh gradients.

---

## 7. Controlling Gradient Flow

Sometimes you want to compute values without building a computation graph, or break the gradient flow through part of the graph. PyTorch provides three tools for this.

### Why would we ever stop gradients on purpose?

Because not every tensor computation is part of learning.

Common examples:

- validation: you want predictions, not gradients
- logging: you want the loss value, not the whole graph kept in memory
- transfer learning: you want some layers frozen
- inference: you want maximum speed and minimum memory use

So the tools below are all answers to slightly different versions of:

> "I want this computation, but I do not want normal gradient tracking here."

### 7.1 `torch.no_grad()` — disable tracking for a block of code

```python
x = torch.tensor(3.0, requires_grad=True)

# Default: operations build the computation graph
y = x ** 2
print(y.requires_grad)   # True — y is part of the graph

# Inside no_grad: no graph is built (faster, less memory)
with torch.no_grad():
    y = x ** 2
    print(y.requires_grad)   # False — not tracked, can't call .backward()

# Decorator form — useful for inference functions
@torch.no_grad()
def predict(model, x):
    return model(x)   # No graph built inside this function
```

Use `no_grad()` during:
- Model evaluation / validation loop (you only need predictions, not gradients)
- Computing metrics (accuracy, F1 score)
- Inference in production
- Manual parameter updates (see the training pipeline video)

### 7.1.1 `torch.inference_mode()` — even stricter inference

```python
with torch.inference_mode():
    preds = model(x)
```

`inference_mode()` is a newer, slightly more aggressive version of `no_grad()`. It disables additional autograd bookkeeping that `no_grad()` still does, making it slightly faster for pure inference. Use it whenever you are sure no gradient computation will ever be needed inside the block.

### 7.2 `.detach()` — cut one tensor's connection to the graph

While `no_grad()` affects a whole code block, `.detach()` surgically removes one specific tensor from the graph:

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2              # y is connected to x in the graph

# detach creates a new tensor that shares x's data but has no graph connection
z = y.detach()
print(z.requires_grad)              # False — not connected to any graph
print(z.data_ptr() == y.data_ptr()) # True — shares the SAME memory, no copy

# Most common use: extract a loss scalar for logging without keeping the graph
loss = criterion(output, target)
loss_value = loss.item()    # ✅ Python float — no graph reference
# OR
loss_tensor = loss.detach() # ✅ Tensor value — no graph reference

# Anti-pattern: storing the tensor itself in a list holds the entire graph in memory!
# losses = []
# losses.append(loss)        # ❌ Keeps graph alive — memory leak over many epochs
# losses.append(loss.item()) # ✅ Safe
```

### 7.3 `tensor.requires_grad_(bool)` — toggle tracking in-place

```python
x = torch.rand(3, 3)
x.requires_grad_(True)    # Enable gradient tracking
x.requires_grad_(False)   # Disable gradient tracking

# Most important use case: freezing layers in transfer learning
# When a parameter has requires_grad=False, backward() skips it completely
# The optimizer also has nothing to update for it
for param in model.backbone.parameters():
    param.requires_grad_(False)  # Freeze backbone — only the head will train
```

### Which tool to use when

| Situation | Tool |
|---|---|
| Freeze a layer permanently (transfer learning) | `param.requires_grad_(False)` |
| Stop gradient through one specific tensor | `tensor.detach()` |
| No gradients needed for a whole code block | `with torch.no_grad():` |
| Pure inference (fastest) | `with torch.inference_mode():` |

---

## 8. Retain Graph for Multiple Backward Passes

By default, the computation graph is **freed after the first `backward()` call** to save memory. If you need to call `backward()` multiple times on the same graph (e.g., for two different losses), use `retain_graph=True`:

### Why is the graph freed?

Because storing the graph and its intermediate values costs memory.

In the usual training pattern, once gradients are computed, PyTorch no longer needs that graph, so it deletes it to save memory.

That is why a second backward pass on the same graph fails unless you explicitly ask PyTorch to keep it.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

y.backward()               # ✅ First backward — graph is freed after this
# y.backward()             # ❌ RuntimeError: graph was freed

# Use retain_graph=True to keep the graph alive
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)   # ✅ Graph is kept
y.backward()                     # ✅ Can backward again
print(x.grad)   # tensor(8.)  ← 4 + 4 (accumulated — don't forget to zero_grad!)
```

`retain_graph=True` is useful in:
- GANs (generator and discriminator losses share parts of the graph)
- Multi-task learning (multiple losses on one forward pass)
- Computing higher-order derivatives

---

## 9. Higher-Order Derivatives

Sometimes we need gradients of gradients — for example, in meta-learning, physics-informed networks, or gradient penalty regularization:

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3

# create_graph=True: build a graph through the gradient computation itself
# This allows us to differentiate the gradient
first_grad = torch.autograd.grad(y, x, create_graph=True)[0]
print(first_grad)      # 3x² = 3(4) = 12  ✅

# Now differentiate the first gradient to get the second derivative
second_grad = torch.autograd.grad(first_grad, x)[0]
print(second_grad)     # 6x = 6(2) = 12  ✅
```

This is not needed for standard supervised training, but it appears in:
- MAML (meta-learning through gradients)
- Physics-informed neural networks (loss includes PDEs)
- Gradient penalty in WGAN-GP
- Some custom regularization methods

---

## 10. Full Backpropagation Example

Let's trace through a real 2-layer network manually and verify that PyTorch computes the same gradients. This proves that autograd is not magic — it's just automated chain rule:

```python
import torch

torch.manual_seed(0)

# ── Simple 1→1→1 network (single neuron per layer) ────────────
x     = torch.tensor([[2.0]])   # Input: one feature
y     = torch.tensor([[5.0]])   # Target: the true output we want
W1    = torch.tensor([[0.5]], requires_grad=True)  # Weight in layer 1
b1    = torch.tensor([[0.1]], requires_grad=True)  # Bias in layer 1
W2    = torch.tensor([[1.0]], requires_grad=True)  # Weight in layer 2
b2    = torch.tensor([[0.0]], requires_grad=True)  # Bias in layer 2

# ── Forward pass ──────────────────────────────────────────────
h     = x @ W1.T + b1           # h = W1*x + b1 = 0.5*2 + 0.1 = 1.1
h_act = torch.relu(h)           # relu(1.1) = 1.1  (positive, so unchanged)
out   = h_act @ W2.T + b2      # out = 1.0 * 1.1 + 0 = 1.1
loss  = ((out - y) ** 2)        # MSE loss: (1.1 - 5)² = 15.21

# ── Backward pass (autograd) ──────────────────────────────────
loss.backward()

# ── Print the gradients autograd computed ─────────────────────
print(f"dL/dW2 = {W2.grad.item():.4f}")   # Gradient w.r.t. layer 2 weight
print(f"dL/dW1 = {W1.grad.item():.4f}")   # Gradient w.r.t. layer 1 weight
print(f"dL/db1 = {b1.grad.item():.4f}")
print(f"dL/db2 = {b2.grad.item():.4f}")

# ── Manual verification of dL/dW2 ─────────────────────────────
# dL/d(out)  = 2 * (out - y) = 2 * (1.1 - 5) = -7.8
# dL/dW2     = dL/d(out) × d(out)/dW2 = -7.8 × h_act = -7.8 × 1.1 = -8.58
print(f"\nManual dL/dW2 = {2*(1.1-5)*1.1:.4f}")   # -8.58 ✅
```

**Why this full example matters:**
- It mirrors exactly what happens inside a real network: linear layer → activation → output layer → loss
- `loss.backward()` computes every partial derivative in one call, even through multiple operations
- The manual check proves that PyTorch is applying the chain rule correctly — not doing anything mysterious

---

## 11. Leaf vs Non-Leaf Tensors

```python
x = torch.tensor(2.0, requires_grad=True)    # Leaf: user created directly
y = x ** 2                                    # Non-leaf: created by an operation

print(x.is_leaf)   # True
print(y.is_leaf)   # False

# Only leaf tensors get .grad populated by default (to save memory)
y.backward()
print(x.grad)   # tensor(4.)  ← Leaf gets its gradient stored ✅
print(y.grad)   # None        ← Non-leaf: gradient computed but not stored

# Force gradient storage for a non-leaf tensor (useful for debugging)
x2 = torch.tensor(2.0, requires_grad=True)
y2 = x2 ** 2
y2.retain_grad()    # Ask PyTorch to store y2's gradient even though it's non-leaf
(y2 * 3).backward()
print(y2.grad)      # tensor(3.)  — dy2_times3/dy2 = 3
```

**Why PyTorch doesn't store every intermediate gradient by default:**
- A large model has millions of intermediate activations
- Storing gradients for every one of them would multiply memory usage dramatically
- The optimizer only needs leaf tensor gradients (the weights)
- Use `retain_grad()` only when debugging specific intermediate values

---

## 12. Common Autograd Bugs

```python
# ── Bug 1: Storing tensor objects in a list (graph memory leak) ──
history = []
history.append(loss.item())    # ✅ Python float — no graph reference held
# history.append(loss)         # ❌ Holds the entire computation graph in memory!

# ── Bug 2: Forgetting to zero gradients ───────────────────────
# ALWAYS call optimizer.zero_grad() before loss.backward()
optimizer.zero_grad()   # Reset gradients to zero
loss.backward()         # Compute fresh gradients
optimizer.step()        # Update weights using those gradients

# ── Bug 3: In-place modification on a tensor needed by backward ──
# Some in-place ops overwrite values that backward needs for its calculation
# Result: RuntimeError during backward about "in-place operation on a version"
# Fix: use out-of-place equivalents (x = x + 1 instead of x.add_(1))

# ── Bug 4: Calling backward twice without retain_graph=True ──────
# By default, the graph is freed after the first backward
# Calling backward again raises RuntimeError about freed buffers
```

When autograd breaks, the usual causes are:
1. The graph was already freed (called backward twice without retain_graph)
2. An in-place operation modified a value needed for backward
3. A tensor was detached earlier than expected
4. The output is non-scalar and no external gradient was supplied

---

## 13. Interview Questions

<details>
<summary><strong>Q1: What is Autograd in PyTorch? How does it work?</strong></summary>

Autograd is PyTorch's automatic differentiation engine. When you perform operations on tensors with `requires_grad=True`, PyTorch builds a dynamic computation graph — a DAG where each node represents an operation and stores the local derivative formula for that operation. When `.backward()` is called on a scalar output (typically the loss), PyTorch traverses this graph in reverse, applying the **chain rule** at each node to accumulate gradients into the `.grad` attribute of all leaf tensors. This eliminates the need to manually derive and implement gradient formulas.
</details>

<details>
<summary><strong>Q2: What is the computation graph?</strong></summary>

It's a directed acyclic graph (DAG) where nodes are tensors and edges represent operations connecting them. Each result tensor stores a reference (`grad_fn`) to the operation that created it and its inputs. When `backward()` is called, PyTorch traverses from the output (loss) back to the inputs, computing partial derivatives via the chain rule at each step. The graph is built **dynamically** during the forward pass — each Python line that operates on a tracked tensor adds a node. This allows Python control flow (if/for/while) to exist inside model forward methods.
</details>

<details>
<summary><strong>Q3: Why must we call optimizer.zero_grad() in every training step?</strong></summary>

PyTorch **accumulates** gradients in `.grad` by default — each backward call adds to the existing gradient rather than replacing it. Without zeroing, the gradient from step N is added to gradient from step N-1, giving step N a gradient that's twice what it should be. This design is intentional: it enables **gradient accumulation** where you sum gradients over multiple mini-batches before updating (useful when GPU memory is too small for large batch sizes). But in standard training with one backward per step, you zero before each backward to get clean, correct gradients.
</details>

<details>
<summary><strong>Q4: What is the difference between no_grad() and detach()?</strong></summary>

- `torch.no_grad()`: context manager that disables gradient tracking for **all operations** in the block. No graph is built at all — faster and uses less memory. Used for inference or evaluation loops where you never need gradients.
- `.detach()`: creates a **new tensor** from existing data, with no connection to the computation graph. The original graph still exists and remains intact. Used to stop gradients flowing through one specific path — e.g., logging loss values, stop-gradient in RL target networks.
</details>

<details>
<summary><strong>Q5: What are leaf tensors? Why do only leaf tensors get .grad?</strong></summary>

Leaf tensors are created directly by the user (not by operations on other tensors). Model parameters (`nn.Parameter`) are always leaves. Non-leaf tensors are intermediate results of operations. PyTorch only stores `.grad` for leaves by default to save memory — intermediate gradients are computed and used in the chain rule but then discarded. Use `.retain_grad()` to force storage of intermediate gradients (useful for debugging hidden layer activations or diagnosing gradient flow).
</details>

<details>
<summary><strong>Q6: What does retain_graph=True do in backward()?</strong></summary>

Normally, after `backward()` completes, PyTorch frees the computation graph to save memory (the graph nodes and stored intermediate values are no longer needed). `retain_graph=True` keeps the graph alive, allowing multiple backward calls on the same graph. Use cases: GANs (generator and discriminator losses share part of the graph and each needs a separate backward), multi-task learning with separate per-task losses, computing second-order derivatives.
</details>

---

## 🔗 References

- [PyTorch Autograd Tutorial](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [CampusX Video 3](https://www.youtube.com/watch?v=BECZ0UB5AR0)
