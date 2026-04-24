---
id: forward-propagation
title: "Forward propagation and how a neural network predicts"
sidebar_label: "10 · Forward Propagation"
sidebar_position: 10
slug: /theory/dnn/forward-propagation-and-how-a-neural-network-predicts
description: "How input moves through an MLP, how activations are computed layer by layer, and how logits or outputs are produced."
tags: [forward-propagation, mlp, activations, logits, prediction]
---

# Forward propagation and how a neural network predicts

Now that the structure of an MLP is clear, we can study the actual computation performed during prediction.

## What forward propagation means

Forward propagation is the process of moving input from the first layer to the last layer and computing all intermediate activations.

![Multi-layer neural network — forward propagation flows left to right: each layer computes a weighted sum then applies a nonlinear activation, passing the result to the next layer](https://commons.wikimedia.org/wiki/Special:Redirect/file/Multi-Layer_Neural_Network-Vector.svg)
*Source: [Wikimedia Commons — Multi-Layer Neural Network](https://commons.wikimedia.org/wiki/File:Multi-Layer_Neural_Network-Vector.svg) (CC BY-SA 4.0)*

For one hidden-layer MLP:

$$
z^{(1)} = W^{(1)}x + b^{(1)}
$$

$$
a^{(1)} = \phi(z^{(1)})
$$

$$
z^{(2)} = W^{(2)}a^{(1)} + b^{(2)}
$$

$$
\hat{y} = g(z^{(2)})
$$

This sequence is the forward pass.

## Step-by-step interpretation

### Step 1: linear combination

Each neuron computes a weighted sum plus bias:

$$
z = Wx + b
$$

### Step 2: activation

The activation function introduces nonlinearity:

$$
a = \phi(z)
$$

### Step 3: repeat across layers

The output of one layer becomes the input of the next.

### Step 4: output head

The final layer depends on the task:

- binary classification: often one logit or sigmoid output
- multiclass classification: logits, then softmax for interpretation
- regression: often a linear output

## Logits, probabilities, and predictions

This distinction is very important.

### Logit

A raw score before normalization.

### Probability

A value after sigmoid or softmax.

### Prediction

The final chosen label or numeric output.

For multiclass classification:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

For binary classification:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

## Why forward propagation comes before loss

The course continuity here is:

$$
\text{MLP structure} \rightarrow \text{forward pass} \rightarrow \text{loss} \rightarrow \text{backpropagation}
$$

We cannot define training until we know how the model produces a prediction in the first place.

## A PyTorch example

```python
import torch
import torch.nn as nn

X = torch.randn(5, 3)

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.relu(z1)
        z2 = self.fc2(a1)
        return z2

model = TinyMLP()
logits = model(X)
probs = torch.softmax(logits, dim=1)

print("logits shape:", logits.shape)
print("probabilities shape:", probs.shape)
```

## Reading the code correctly

- `fc1` computes the first affine transform
- `relu` applies the nonlinearity
- `fc2` computes final logits
- `softmax` converts logits into class probabilities for interpretation

## Interview questions

<details>
<summary>Is softmax part of the model or only interpretation?</summary>

It depends on implementation. Many training APIs use raw logits internally and apply the stable softmax logic inside the loss function.
</details>

<details>
<summary>What is the difference between forward propagation and inference?</summary>

Forward propagation is the computation itself. Inference usually means using that computation at prediction time after training.
</details>

<details>
<summary>Why are activations needed between layers?</summary>

Without them, the entire network collapses into one linear transformation.
</details>

<details>
<summary>What is the difference between logits and probabilities?</summary>

Logits are raw unnormalized scores. Probabilities are transformed outputs, usually after sigmoid or softmax.
</details>

<details>
<summary>Why does the output layer depend on the task?</summary>

Because regression, binary classification, and multiclass classification require different output interpretations and often different losses.
</details>

<details>
<summary>Why must forward propagation be understood before backpropagation?</summary>

Because backpropagation differentiates the exact computations performed during the forward pass.
</details>

## Final takeaway

Forward propagation is the exact recipe by which input becomes prediction. Once this is clear, the next step is to define how wrong that prediction is with a loss function.

## References

- CampusX YouTube: Forward Propagation | How a neural network predicts output?
