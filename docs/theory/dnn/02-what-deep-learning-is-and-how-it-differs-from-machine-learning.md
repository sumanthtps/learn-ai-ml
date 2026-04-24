---
id: what-is-deep-learning
title: "What deep learning is and how it differs from machine learning"
sidebar_label: "02 · DL vs ML"
sidebar_position: 2
slug: /theory/dnn/what-deep-learning-is-and-how-it-differs-from-machine-learning
description: "What deep learning means, how it differs from classical machine learning, and why representation learning changed the field."
tags: [deep-learning, machine-learning, representation-learning, ann, foundations]
---

# What deep learning is and how it differs from machine learning

Deep learning is best understood as a special part of machine learning where the model learns useful internal representations automatically instead of depending heavily on manually engineered features.

![AI ⊃ ML ⊃ Deep Learning — deep learning is the subset that learns hierarchical representations automatically from raw data](https://commons.wikimedia.org/wiki/Special:Redirect/file/AI-ML-DL.svg)
*Source: [Wikimedia Commons — AI-ML-DL hierarchy](https://commons.wikimedia.org/wiki/File:AI-ML-DL.svg) (CC BY-SA 4.0)*

## AI, ML, and deep learning

The hierarchy is:

$$
\text{Artificial Intelligence} \supset \text{Machine Learning} \supset \text{Deep Learning}
$$

- Artificial Intelligence is the broad goal of building systems that behave intelligently.
- Machine Learning is the subset where systems learn patterns from data.
- Deep Learning is the subset of machine learning that uses multi-layer neural networks.

## The central difference

In classical machine learning, a large part of success often comes from manually designing features:

$$
x \xrightarrow{\text{feature engineering}} \phi(x) \xrightarrow{\text{model}} \hat{y}
$$

In deep learning, the network tries to learn the intermediate representation itself:

$$
x \xrightarrow{\text{layer 1}} h^{(1)} \xrightarrow{\text{layer 2}} h^{(2)} \xrightarrow{\cdots} \hat{y}
$$

This is why deep learning is strongly associated with representation learning.

## Why this matters

For structured tabular data, classical ML often works very well. But for images, speech, text, and video, it is hard to hand-design the right features. Deep learning becomes powerful in exactly those settings because raw data has hierarchical patterns.

Examples:

- image: edges -> textures -> parts -> objects
- text: characters -> words -> phrases -> meaning
- speech: frequencies -> phonemes -> words -> sentences

## Why neural networks are suited for this

A neural network repeatedly applies:

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
$$

$$
a^{(l)} = \phi(z^{(l)})
$$

Each layer transforms the representation. Early layers learn simple patterns; later layers learn more abstract ones.

## Classical ML vs deep learning

| Aspect | Classical ML | Deep Learning |
| --- | --- | --- |
| Feature design | Often manual | Mostly learned |
| Data need | Works on smaller datasets | Usually benefits from more data |
| Compute need | Lower | Higher |
| Interpretability | Often easier | Often harder |
| Best domains | tabular, low-data settings | images, text, speech, large-scale tasks |

## Why deep learning became practical only later

Deep learning ideas are old, but they became successful at scale because of three big shifts:

- more data
- GPUs and specialized hardware
- better training methods such as ReLU, improved initialization, and stronger optimizers

Without these, training deep networks was unstable and slow.

## Continuity with the next note

This note gives the field-level difference. The next note narrows that broad picture into model families:

- ANN for dense/tabular settings
- CNN for spatial data
- RNN/LSTM for sequential data
- transformer for attention-based sequence modeling

So the flow is:

$$
\text{Why deep learning?} \rightarrow \text{What kinds of networks exist?}
$$

## A minimal example

```python
import torch
import torch.nn as nn

x = torch.tensor([[0.2, 0.5, -0.1]], dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
)

logits = model(x)
print(logits)
```

This tiny network already shows the core deep-learning idea: instead of predicting directly from the raw input with one formula, the model builds an internal hidden representation first.

## Interview questions

<details>
<summary>Is deep learning always better than machine learning?</summary>

No. On many small, clean tabular datasets, tree-based models or linear models may outperform deep networks.
</details>

<details>
<summary>Why is it called deep learning?</summary>

Because the model contains multiple stacked hidden layers, so the learned representation becomes progressively deeper.
</details>

<details>
<summary>What is representation learning?</summary>

It means the model learns useful features internally rather than relying only on human-designed features.
</details>

<details>
<summary>Why does deep learning need more compute?</summary>

Because it trains many parameters and repeatedly performs large matrix operations during forward and backward passes.
</details>

<details>
<summary>When does classical machine learning still make more sense than deep learning?</summary>

When the dataset is small, the features are already informative, interpretability matters more, or the problem is structured tabular prediction.
</details>

<details>
<summary>What is the biggest conceptual difference between feature engineering and representation learning?</summary>

Feature engineering asks the human to design the transformation. Representation learning asks the model to discover useful internal transformations from data.
</details>

## Common mistakes

- thinking deep learning is a completely different field from machine learning
- assuming more layers automatically mean better performance
- ignoring data size and problem type when choosing between ML and DL
- treating hidden layers as magic instead of learned transformations

## Final takeaway

Classical machine learning often learns from features. Deep learning tries to learn the features and the prediction function together.

## References

- CampusX YouTube: What is Deep Learning? Deep Learning vs Machine Learning
