---
id: lstm-what-and-why
title: "What LSTM is and why it was created"
sidebar_label: "61 · What is LSTM"
sidebar_position: 61
slug: /theory/dnn/what-lstm-is-and-why-it-was-created
description: "The vanishing gradient problem in vanilla RNNs, why LSTM was invented, and the intuition behind the cell state as a protected memory lane."
tags: [lstm, rnn, vanishing-gradients, long-term-dependencies, sequence-modeling, deep-learning]
---

# What LSTM is and why it was created

Vanilla RNNs theoretically remember all past inputs through the hidden state $h_t$. In practice they cannot. Gradients backpropagated through many time steps either vanish (shrink to zero, preventing learning from distant past) or explode (grow without bound, destabilizing training). The Long Short-Term Memory (LSTM), introduced by Hochreiter and Schmidhuber in 1997, was designed specifically to fix the vanishing gradient problem and enable learning of long-range dependencies.

## One-line definition

An LSTM is a recurrent network unit with an explicit memory cell $c_t$ protected by learned gates (forget, input, output) that control what information is kept, added, or read — enabling gradient flow over hundreds of time steps.

![Standard RNN (top) vs LSTM (bottom) — the LSTM's repeating module contains four interacting layers instead of one](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)
*Source: [Colah's Blog — Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (CC BY 4.0)*

## Why vanilla RNNs fail on long sequences

The vanilla RNN hidden state update:

$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b)
$$

Backpropagation through time computes:

$$
\frac{\partial \mathcal{L}}{\partial h_{t-k}} = \left(\prod_{i=t-k}^{t-1} \frac{\partial h_{i+1}}{\partial h_i}\right) \frac{\partial \mathcal{L}}{\partial h_t}
$$

Each factor $\frac{\partial h_{i+1}}{\partial h_i} = \text{diag}(\tanh'(z_i)) W_h$ involves:
- The tanh derivative: $\tanh'(z) \in (0, 1]$ — always less than 1
- The weight matrix $W_h$

If the singular values of $W_h$ multiplied by the tanh derivatives are $< 1$, the product shrinks exponentially. For 100 time steps: $(0.9)^{100} \approx 2.7 \times 10^{-5}$ — essentially zero. The model cannot learn a dependency between time step 1 and time step 100.

## The LSTM solution: a protected memory cell

The key insight: separate the hidden state (short-term, updated every step) from the cell state (long-term, updated via addition rather than multiplication):

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

The critical property: the cell state update is **additive**. Adding information does not shrink the gradient the way multiplying by a matrix does. The gradient through the cell state highway:

$$
\frac{\partial c_t}{\partial c_{t-1}} = f_t
$$

The forget gate $f_t \in (0, 1)$ can be close to 1 when the model needs to remember something. When $f_t \approx 1$, the gradient flows unimpeded through the cell state for many time steps.

## The "Constant Error Carousel"

Hochreiter and Schmidhuber called this mechanism the **Constant Error Carousel (CEC)**: if the forget gate is 1 and the input gate is 0, the cell state is copied exactly: $c_t = c_{t-1}$. The gradient flows back unchanged:

$$
\frac{\partial \mathcal{L}}{\partial c_s} = \frac{\partial \mathcal{L}}{\partial c_t} \quad \text{for all } s \le t
$$

This is what enables learning dependencies across hundreds of time steps.

## Mental model: the cell state as a conveyor belt

Think of the cell state $c_t$ as a conveyor belt running through the entire sequence:

```
c_0 ─→ c_1 ─→ c_2 ─→ c_3 ─→ ... ─→ c_T
       ↑  ↑       ↑  ↑
       f  i       f  i
```

- The **forget gate** ($f_t$) decides what to erase from the belt
- The **input gate** ($i_t$) decides what new information to add
- The **output gate** ($o_t$) decides what to read from the belt into $h_t$

The belt can carry information for many steps undisturbed if the forget gate remains open and the input gate closed.

## Why "Long Short-Term Memory"

The name is explained by the architecture:
- **Long**: the cell state can carry information over long sequences
- **Short-Term Memory**: the hidden state $h_t$ is the traditional short-term memory (reset at every step by the output gate)

An LSTM is a combination: long-term storage (cell) + short-term context (hidden state).

## LSTM vs vanilla RNN: what each component buys

| Mechanism | Vanilla RNN | LSTM |
|---|---|---|
| Hidden state | $h_t = \tanh(W_h h_{t-1} + ...)$ | $h_t = o_t \odot \tanh(c_t)$ |
| Memory | Hidden state only | Cell state + hidden state |
| Gradient path | Multiplicative (vanishes) | Additive through cell (preserved) |
| Long-range deps | Poor (> 20 steps) | Good (> 200 steps) |
| Parameters | $d^2 + dE$ | $4(d^2 + dE)$ (×4 for 4 gates) |

## PyTorch: intuition demo

```python
import torch
import torch.nn as nn


# ============================================================
# Gradient vanishing in vanilla RNN vs LSTM
# ============================================================
def test_gradient_flow(model_type: str, sequence_len: int = 100):
    """
    Measure gradient magnitude at the first time step after backprop.
    A healthy model should have non-negligible gradient here.
    """
    input_size, hidden_size = 10, 32
    x = torch.randn(1, sequence_len, input_size)   # (batch, T, input)
    x.requires_grad_(True)

    if model_type == "rnn":
        model = nn.RNN(input_size, hidden_size, batch_first=True)
    else:
        model = nn.LSTM(input_size, hidden_size, batch_first=True)

    output, _ = model(x)
    loss = output[:, -1, :].sum()   # gradient from the last step
    loss.backward()

    # Gradient at the first input: how much signal reaches the beginning?
    first_step_grad = x.grad[:, 0, :].abs().mean().item()
    return first_step_grad


print("Gradient magnitude at step 0 (100 steps back):")
for T in [20, 50, 100]:
    rnn_grad  = test_gradient_flow("rnn",  T)
    lstm_grad = test_gradient_flow("lstm", T)
    print(f"  T={T:3d}: RNN={rnn_grad:.2e}   LSTM={lstm_grad:.2e}")


# ============================================================
# Demonstrate long-range memorization task
# ============================================================
class LongRangeTask(nn.Module):
    """
    Copy task: given a sequence [signal, zeros...], output the signal
    after T steps. Tests long-range memory.
    """
    def __init__(self, hidden_size: int, use_lstm: bool = True):
        super().__init__()
        if use_lstm:
            self.rnn = nn.LSTM(1, hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.out(output[:, -1, :])  # predict from final hidden state


# Create the copy task: remember a value seen T steps ago
def create_copy_task(batch: int = 64, T: int = 50):
    """Input: [value, 0, 0, ..., 0] of length T. Target: value."""
    signal = torch.randn(batch, 1)
    zeros = torch.zeros(batch, T - 1)
    x = torch.cat([signal, zeros], dim=1).unsqueeze(-1)  # (B, T, 1)
    y = signal   # target is the initial signal value
    return x, y


# Training loop (abbreviated)
for use_lstm in [True, False]:
    model_name = "LSTM" if use_lstm else "RNN"
    model = LongRangeTask(hidden_size=32, use_lstm=use_lstm)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(200):
        x, y = create_copy_task()
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            losses.append(loss.item())
    print(f"{model_name} copy task losses: {[f'{l:.4f}' for l in losses]}")
```

## When LSTM is overkill

LSTM was the state of the art for sequence modeling from ~1997 to ~2017. Today:

- **Short sequences (< 50 steps)**: vanilla RNN often works fine
- **Long sequences**: Transformers have largely replaced LSTMs for NLP because attention directly computes dependencies between any two positions (no gradient path length issue)
- **LSTM advantages remain**: streaming/online processing (processes one step at a time, constant memory), very long sequences where transformer quadratic attention is prohibitive, time series with irregular sampling

## Interview questions

<details>
<summary>What exactly causes vanishing gradients in vanilla RNNs, and how does LSTM solve it?</summary>

The gradient at time step $t - k$ is the product of $k$ Jacobians: $\prod \frac{\partial h_{i+1}}{\partial h_i} = \prod \text{diag}(\tanh') W_h$. The tanh derivative is at most 1, and if the spectral radius of $W_h$ is $< 1$, the product shrinks exponentially with $k$. After 100 steps, gradients effectively reach zero. LSTM solves this by introducing the cell state, which is updated additively: $c_t = f_t c_{t-1} + i_t \tilde{c}_t$. The gradient through the cell state is $\partial c_t / \partial c_{t-1} = f_t$. When $f_t \approx 1$, gradient flows unchanged — no vanishing. The model learns to keep $f_t$ high when it needs long-term memory.
</details>

<details>
<summary>Why is the cell state update additive while the hidden state update is multiplicative?</summary>

The hidden state $h_t = o_t \odot \tanh(c_t)$ is a nonlinear transformation of the cell state — it is the "output" that feeds forward. The cell state $c_t = f_t c_{t-1} + i_t \tilde{c}_t$ uses addition to update the memory. This additive structure is the key to the gradient highway: the derivative of addition is 1 (no shrinking), whereas multiplication through tanh and weight matrices gives derivatives $< 1$ (exponential decay). The cell state acts as an "accumulator" that the gates learn to manage selectively.
</details>

## Common mistakes

- Confusing $h_t$ and $c_t$ — $h_t$ is the output and short-term state; $c_t$ is the long-term cell memory
- Thinking LSTM completely eliminates vanishing gradients — it greatly reduces the problem but does not eliminate it; very long sequences (> 1000 steps) still challenge LSTMs
- Using LSTM for classification without taking the correct output — for many-to-one, use `h_n` (the final hidden state), not the entire output sequence

## Final takeaway

LSTMs were created to solve the vanishing gradient problem that makes vanilla RNNs unable to learn long-range dependencies. The cell state highway with additive updates allows gradients to flow unchanged over many time steps when the forget gate is near 1. The mental model: the cell state is a conveyor belt that carries information across time, controlled by three learned gates. LSTMs dominated sequence modeling from 1997 to 2017 and remain useful for streaming and online prediction tasks where transformers are too memory-heavy.

## References

- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
