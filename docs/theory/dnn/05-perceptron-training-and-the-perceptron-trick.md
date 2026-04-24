---
id: perceptron-training
title: "Perceptron training and the perceptron trick"
sidebar_label: "05 · Perceptron Training"
sidebar_position: 5
slug: /theory/dnn/perceptron-training-and-the-perceptron-trick
description: "How a perceptron learns from mistakes, the classic perceptron update rule, and why this is the ancestor of gradient-based training."
tags: [perceptron, training, update-rule, learning-rate, ann]
---

# Perceptron training and the perceptron trick

The previous note explained what a perceptron computes. This note explains how its parameters move during learning.

## The core question

If a point is classified incorrectly, how should the weights change so that the decision boundary moves in the correct direction?

That is the perceptron training idea.

## Binary labels

The classical perceptron often uses labels:

$$
y \in \{-1, +1\}
$$

Prediction is:

$$
\hat{y} = \text{sign}(w^T x + b)
$$

## The perceptron update rule

For a misclassified sample:

$$
w \leftarrow w + \eta y x
$$

$$
b \leftarrow b + \eta y
$$

where:

- $\eta$ is the learning rate
- $y$ tells the update direction

![Perceptron training — the decision boundary moves in response to misclassified points; the update rule adjusts weights to reduce errors](https://commons.wikimedia.org/wiki/Special:Redirect/file/Artificial_neural_network.svg)
*Source: [Wikimedia Commons — Artificial Neural Network](https://commons.wikimedia.org/wiki/File:Artificial_neural_network.svg) (CC BY-SA 4.0)*
- $x$ tells how much each feature contributes

## Why the update makes sense

### If the true label is +1

and the model predicts incorrectly, we want:

$$
w^T x + b
$$

to increase. Adding $\eta x$ pushes the score upward.

### If the true label is -1

and the model predicts incorrectly, we want the score to decrease. Adding $\eta (-x)$ pushes it downward.

So the perceptron learns by shifting the decision boundary after mistakes.

## The perceptron trick

The phrase "perceptron trick" usually refers to the geometric intuition that every misclassified point nudges the separating line in the direction that would classify that point correctly next time.

This is easier to think about geometrically than algebraically:

- positive misclassified point -> boundary moves toward classifying it as positive
- negative misclassified point -> boundary moves toward classifying it as negative

## Continuity with later optimization

The perceptron update is not yet the full modern gradient-descent story, but it prepares the exact mindset:

- define a prediction rule
- measure whether it is wrong
- update parameters in a direction that should reduce future error

This is the ancestor of:

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
$$

which appears later in backpropagation and gradient descent.

## When perceptron training converges

If the data is linearly separable, the perceptron algorithm can converge to a separating hyperplane.

If the data is not linearly separable, it may keep making mistakes forever.

That fact sets up the next two chapters:

- loss functions for smoother training
- the failure of a single perceptron on nonlinear problems

## A simple manual implementation

```python
import torch

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

# AND gate in {-1, +1} form
y = torch.tensor([-1.0, -1.0, -1.0, 1.0])

w = torch.zeros(2)
b = torch.tensor(0.0)
lr = 0.1

for _ in range(20):
    for xi, yi in zip(X, y):
        score = torch.dot(w, xi) + b
        pred = 1.0 if score >= 0 else -1.0
        if pred != yi:
            w = w + lr * yi * xi
            b = b + lr * yi

print("weights:", w)
print("bias:", b)
```

## What this code is showing

- no backpropagation is used yet
- no smooth loss is used yet
- the update happens only on mistakes
- this is a mistake-driven learning rule

## Interview questions

<details>
<summary>Why does the learning rate matter in perceptron training?</summary>

It controls the step size. Too large can overshoot; too small can make learning slow.
</details>

<details>
<summary>Why are labels often written as -1 and +1?</summary>

Because the update rule becomes clean and symmetric in that representation.
</details>

<details>
<summary>Is perceptron training the same as logistic regression training?</summary>

No. Logistic regression uses a smooth probabilistic model and optimizes a differentiable loss. The perceptron algorithm uses a mistake-driven update rule.
</details>

<details>
<summary>Why does the perceptron update happen only on mistakes?</summary>

Because the classic algorithm is driven by misclassification. If a point is already classified correctly, the rule does not adjust weights for that sample.
</details>

<details>
<summary>When does the perceptron algorithm converge?</summary>

It can converge when the data is linearly separable. If the data is not linearly separable, it may keep updating indefinitely.
</details>

<details>
<summary>How is perceptron training related to gradient descent?</summary>

It is an earlier learning rule that shares the core idea of updating parameters to reduce mistakes, but it does not yet use smooth differentiable losses like modern gradient descent training.
</details>

## Final takeaway

Perceptron training is the first example in the course of learning by parameter updates. It is simple, local, and geometric, and it prepares the ground for loss functions and gradient descent.

## References

- CampusX YouTube: Perceptron Trick | How to train a Perceptron
