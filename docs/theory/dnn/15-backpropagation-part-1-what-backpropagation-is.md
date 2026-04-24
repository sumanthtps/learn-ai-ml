---
id: backpropagation-part-1
title: "Backpropagation part 1: what backpropagation is"
sidebar_label: "15 · Backpropagation I"
sidebar_position: 15
slug: /theory/dnn/backpropagation-part-1-what-backpropagation-is
description: "A deep explanation of backpropagation: why it exists, how gradients flow, what the chain rule is doing, and how learning actually happens."
tags: [backpropagation, gradients, chain-rule, optimization, deep-learning]
---

# Backpropagation part 1: what backpropagation is

After forward propagation gives a prediction and the loss tells us how wrong that prediction is, we still need one more step: we must determine how each parameter contributed to that error. Backpropagation performs exactly this computation.

## One-line definition

Backpropagation is the algorithm that computes how much each parameter in a neural network contributed to the final error, so we can update those parameters to reduce the error.

![Artificial neural network — backpropagation computes the gradient of the loss with respect to each weight by applying the chain rule layer by layer from output back to input](https://commons.wikimedia.org/wiki/Special:Redirect/file/Artificial_neural_network.svg)
*Source: [Wikimedia Commons — Artificial neural network](https://commons.wikimedia.org/wiki/File:Artificial_neural_network.svg) (CC BY-SA 4.0)*

## Why this topic matters

Forward propagation explains how a network predicts. Backpropagation explains how it learns.

Without backpropagation, deep learning would be impractical because:

- networks have millions or billions of parameters
- manually differentiating each parameter is impossible
- efficient training requires reusing intermediate computations

Backpropagation is the engine behind gradient-based learning.

## The learning loop in one picture

```mermaid
flowchart LR
    A[Input x] --> B[Forward Pass]
    B --> C[Prediction y_hat]
    C --> D[Loss L(y_hat, y)]
    D --> E[Backward Pass]
    E --> F[Gradients dL/dtheta]
    F --> G[Optimizer Update]
    G --> H[New Parameters]
```

## The central idea

A neural network is a composition of functions:

$$
\hat{y} = f_L(f_{L-1}(\cdots f_2(f_1(x)) \cdots))
$$

The loss is a function of the prediction:

$$
\mathcal{L} = \mathcal{L}(\hat{y}, y)
$$

We want the gradient of the loss with respect to each parameter:

$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}}, \quad \frac{\partial \mathcal{L}}{\partial b^{(l)}}
$$

Backpropagation computes these efficiently by moving backward through the graph and repeatedly applying the chain rule.

The continuity from the previous notes is:

- perceptron and MLP notes defined the forward computation
- loss-function notes defined the objective
- this note explains how the objective produces gradients
- the next notes use those gradients for optimization

## The chain rule intuition

Suppose:

$$
z = wx + b,\quad a = \sigma(z),\quad \mathcal{L} = (a-y)^2
$$

Then:

$$
\frac{\partial \mathcal{L}}{\partial w}
=
\frac{\partial \mathcal{L}}{\partial a}
\cdot
\frac{\partial a}{\partial z}
\cdot
\frac{\partial z}{\partial w}
$$

This is the chain rule in action.

Interpretation:

- how sensitive is the loss to the output?
- how sensitive is the output to the pre-activation?
- how sensitive is the pre-activation to the weight?

Multiply those sensitivities and you get the effect of the weight on the final loss.

## What gradients represent

A gradient tells you:

- direction: should this parameter go up or down?
- magnitude: how strongly should it change?

If:

$$
\frac{\partial \mathcal{L}}{\partial w} > 0
$$

then increasing $w$ increases the loss, so gradient descent moves $w$ downward.

If:

$$
\frac{\partial \mathcal{L}}{\partial w} < 0
$$

then increasing $w$ decreases the loss, so gradient descent moves $w$ upward.

## Why backpropagation is efficient

Naively, you could compute each derivative from scratch. That would repeat the same work again and again.

Backpropagation avoids that by:

- storing intermediate activations from the forward pass
- computing local derivatives once
- reusing upstream gradients layer by layer

That is why the backward pass is fast enough to train deep networks.

## Local gradient view

For layer $l$:

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad a^{(l)} = \phi(z^{(l)})
$$

Define the error signal at layer $l$ as:

$$
\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial z^{(l)}}
$$

Then:

$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T
$$

and

$$
\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}
$$

For hidden layers:

$$
\delta^{(l)} = \left((W^{(l+1)})^T \delta^{(l+1)}\right) \odot \phi'(z^{(l)})
$$

This equation is the heart of backpropagation.

For the output layer in binary classification with sigmoid and binary cross-entropy, the error term becomes:

$$
\delta^{(L)} = a^{(L)} - y
$$

This is one of the most important simplifications in introductory deep learning.

## The key mental model

Backpropagation is not "magic differentiation." It is message passing:

- the output layer computes how wrong the network was
- that error signal is passed backward
- each layer transforms that signal according to its local derivative
- each parameter receives credit or blame

## Worked micro-example

Suppose a 1-neuron model:

$$
z = wx + b,\quad a = \sigma(z)
$$

Binary cross-entropy loss:

$$
\mathcal{L} = -\left[y \log(a) + (1-y)\log(1-a)\right]
$$

For sigmoid + BCE, the derivative simplifies beautifully:

$$
\frac{\partial \mathcal{L}}{\partial z} = a - y
$$

Then:

$$
\frac{\partial \mathcal{L}}{\partial w} = (a-y)x
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = a-y
$$

This is why sigmoid + BCE is such a standard binary-classification pair.

## PyTorch example

```python
import torch
import torch.nn.functional as F

x = torch.tensor([[1.0, 2.0]])
target = torch.tensor([[1.0]])

W1 = torch.randn(2, 3, requires_grad=True)
b1 = torch.zeros(3, requires_grad=True)
W2 = torch.randn(3, 1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)

z1 = x @ W1 + b1
a1 = torch.relu(z1)
z2 = a1 @ W2 + b2
loss = F.binary_cross_entropy_with_logits(z2, target)

loss.backward()

print("loss:", float(loss))
print("grad W1:", W1.grad)
print("grad b1:", b1.grad)
print("grad W2:", W2.grad)
print("grad b2:", b2.grad)
```

## What is happening in this code

- the forward pass builds a computational graph
- `loss.backward()` traverses that graph backward
- PyTorch applies the chain rule automatically
- the gradients are stored in `.grad`

That is autograd, which is built on backpropagation.

## Why activations matter for backpropagation

The derivative of the activation controls gradient flow.

Examples:

- sigmoid can saturate and produce tiny gradients
- tanh also saturates, though it is zero-centered
- ReLU avoids saturation on the positive side and helps deeper networks train

This is why activation choice affects optimization, not just expressiveness.

## Interview questions

<details>
<summary>Is backpropagation the same as gradient descent?</summary>

No. Backpropagation computes gradients. Gradient descent uses those gradients to update parameters.
</details>

<details>
<summary>Why do we need the forward pass first?</summary>

Because backward derivatives depend on intermediate values like activations and pre-activations from the forward pass.
</details>

<details>
<summary>What does "error signal" mean?</summary>

It means how much a unit or parameter contributed to the final loss, expressed as a derivative.
</details>

<details>
<summary>Why is the chain rule so important?</summary>

Because a deep network is a composition of functions, and the chain rule is exactly the calculus tool for differentiating compositions.
</details>

<details>
<summary>Why do vanishing gradients happen?</summary>

Because repeated multiplication by small derivatives can shrink the backward signal exponentially across layers or time steps.
</details>

<details>
<summary>Why is sigmoid plus BCE such a common pairing?</summary>

Because the output-layer derivative simplifies nicely to <code>a - y</code>, making optimization cleaner and more interpretable.
</details>

<details>
<summary>What does backpropagation output?</summary>

It outputs gradients of the loss with respect to parameters and, conceptually, with respect to intermediate nodes in the computation graph.
</details>

## Common mistakes

- thinking backpropagation and optimizer step are the same
- ignoring gradient shapes
- forgetting that gradients accumulate in PyTorch unless cleared
- using an unstable loss/activation combination
- memorizing formulas without understanding what the error signal means

## Advanced perspective

At scale, backpropagation is best understood as reverse-mode automatic differentiation. It is especially efficient when:

- the model has many parameters
- the loss is scalar
- the forward graph can be decomposed into local differentiable operations

This is exactly the setting of deep learning.

## High-yield interview checks

<details>
<summary>What problem does backpropagation solve?</summary>

It efficiently computes gradients of the loss with respect to all parameters in a deep network.
</details>

<details>
<summary>Why is backpropagation efficient?</summary>

Because it reuses intermediate computations and applies reverse-mode differentiation instead of recomputing each derivative from scratch.
</details>

<details>
<summary>What are the prerequisites for backpropagation?</summary>

Differentiable operations, a scalar loss, and the chain rule.
</details>

## Final takeaway

Backpropagation is the reason neural networks are trainable. It turns "the model was wrong" into a precise parameter-by-parameter instruction for how to become less wrong.

## References

- CampusX YouTube: Backpropagation in Deep Learning Part 1
