---
id: what-is-deep-learning
title: "What deep learning is and how it differs from machine learning"
sidebar_label: "02 · DL vs ML"
sidebar_position: 2
slug: /theory/dnn/what-deep-learning-is-and-how-it-differs-from-machine-learning
description: "What deep learning means, how it differs from classical machine learning, why representation learning changed the field, and when to use each."
tags: [deep-learning, machine-learning, representation-learning, ann, foundations, feature-engineering]
---

# What deep learning is and how it differs from machine learning

Deep learning is best understood as a special part of machine learning where the model learns useful internal representations automatically instead of depending heavily on manually engineered features. The key distinction is **end-to-end learning**: instead of the human designing features and then the model learning to classify those features, the network learns both features and classifier jointly from raw data.

![AI ⊃ ML ⊃ Deep Learning — deep learning is the subset that learns hierarchical representations automatically from raw data](https://commons.wikimedia.org/wiki/Special:Redirect/file/AI-ML-DL.svg)
*Source: [Wikimedia Commons — AI-ML-DL hierarchy](https://commons.wikimedia.org/wiki/File:AI-ML-DL.svg) (CC BY-SA 4.0)*

## One-line definition

Deep learning is machine learning using multi-layer neural networks that automatically learn hierarchical representations from raw data, removing the need for manual feature engineering.

## AI, ML, and deep learning: the hierarchy

The relationship is a strict subset hierarchy:

$$
\text{Artificial Intelligence} \supset \text{Machine Learning} \supset \text{Deep Learning}
$$

- **Artificial Intelligence (AI)**: The broad field of building systems that perceive, reason, and act intelligently — including robotics, game playing, planning, and more.
- **Machine Learning (ML)**: The subset of AI where systems **learn patterns from data** rather than being explicitly programmed with rules. Examples: decision trees, SVMs, random forests, neural networks, gradient boosting.
- **Deep Learning (DL)**: The subset of ML using **neural networks with multiple hidden layers**. Examples: convolutional neural networks (CNNs), recurrent neural networks (RNNs), Transformers.

Not all machine learning is deep learning. Conversely, not all AI uses machine learning (some uses symbolic reasoning, planning algorithms, etc.).

## The central difference: feature engineering vs representation learning

### Classical ML: manually designed features

In classical machine learning, the pipeline is explicit:

$$
\text{Raw data } x \xrightarrow{\text{human-designed features}} \phi(x) \xrightarrow{\text{classifier (SVM, tree, etc.)}} \hat{y}
$$

**Example: Image classification the classical ML way**

A human manually extracts features from an image:
- Edge orientations (using Sobel filters)
- Color histograms
- Texture statistics (using gabor filters)
- Shape descriptors

Then trains an SVM or random forest on those features.

**The bottleneck**: The quality of predictions is heavily dependent on the quality of hand-crafted features. Good features require domain expertise and manual trial-and-error.

### Deep learning: learned representations

In deep learning, the pipeline is implicit:

$$
\text{Raw data } x \xrightarrow{\text{layer 1}} h^{(1)} \xrightarrow{\text{layer 2}} h^{(2)} \xrightarrow{\cdots \text{layer } L} \hat{y}
$$

Each hidden layer $h^{(l)}$ is an **intermediate representation** that the network learns automatically through backpropagation. No human designs these representations — they emerge from the optimization process.

**Example: Image classification the deep learning way**

- **Layer 1** learns edge detectors (lines, corners, curves)
- **Layer 2** learns texture detectors (combines edges into textures)
- **Layer 3** learns part detectors (eyes, noses, wheels)
- **Layer 4** learns object detectors (faces, cars, dogs)
- **Output layer** classifies based on learned object detectors

The network **discovers this hierarchy automatically** from data, without any human guidance about what edges or parts should look like.

## Why this matters: the representation learning breakthrough

For **structured tabular data** (spreadsheets, tabular data with numeric/categorical features), classical ML works very well because the features are often already useful — each column is already a meaningful attribute.

But for **unstructured raw data** (images, audio, text), it is nearly impossible to hand-design good features:

- **Images**: What is the right set of filters? Sobel? Gabor? How many scales? There are infinite choices.
- **Text**: Should you count word frequency? N-grams? TF-IDF? Again, endless choices.
- **Audio**: Spectrograms? Mel-frequency cepstral coefficients (MFCCs)? MFCCs work but are domain-specific heuristics.

**Deep learning removes this bottleneck** by letting the network discover useful features. The network finds that edges are useful (learns edge detectors in early layers), then discovers that edges combine into textures (learns texture detectors), and so on.

## Hierarchical feature learning: the core insight

Raw data contains hierarchical structure. A CNN learns to exploit this:

| Domain | Hierarchy |
|--------|-----------|
| **Vision** | pixels → edges → textures → parts → objects |
| **Language** | characters → subwords → words → phrases → sentences → meaning |
| **Speech** | waveform → frequencies → phonemes → words → sentences |

Each level of the hierarchy can be learned by composing operations from the previous level. A deep network is optimized for discovering this structure.

## Why neural networks are suited for representation learning

A neural network repeatedly applies:

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)} \quad \text{(linear transformation)}
$$

$$
a^{(l)} = \phi(z^{(l)}) \quad \text{(nonlinear activation)}
$$

This composition of linear transformations and nonlinear activations is flexible enough to learn any hierarchical structure. Early layers learn simple patterns; deeper layers combine those patterns into more abstract concepts.

## Classical ML vs deep learning: a detailed comparison

| Aspect | Classical ML | Deep Learning |
|--------|---|---|
| **Feature design** | Manual, requires domain expertise | Automatically learned from data |
| **Data requirement** | Works well with 100s–1000s of samples | Typically needs 1000s–millions of samples |
| **Compute requirement** | Can run on a CPU | Needs GPUs/TPUs for practical training |
| **Training time** | Minutes to hours | Hours to weeks |
| **Interpretability** | Often easier (e.g., feature importance in trees) | Very difficult (black-box representations) |
| **Best use cases** | Tabular data, small datasets, when interpretability is critical | Images, text, speech, large-scale tasks |
| **When it works with less data** | ✓ Better generalization with less data | ✗ Overfits badly on small datasets |
| **Transfer learning** | Limited; features don't transfer well | Excellent; learned representations transfer across tasks |

### When classical ML is still better

- **Small datasets** (&lt;1000 samples): Deep networks overfit; tree-based models generalize better
- **Tabular data with strong domain features**: If each column is already a meaningful feature (e.g., age, income, credit score), classical ML is simpler and faster
- **Interpretability required**: If you need to explain why a prediction was made, tree-based models are more interpretable
- **Limited compute**: Training a deep network on a laptop is painful; classical ML is fast
- **Extremely sparse data**: When most features are zero (e.g., user-item matrices), classical methods often outperform neural networks

## Why deep learning became practical only in the 2010s

Deep learning ideas (backpropagation, neural networks) were invented in the 1980s, but they didn't become practical at scale until ~2010 because three things had to align:

### 1. Data explosion
The internet created billions of labeled images (ImageNet, 2009), text, and videos. Deep networks need massive amounts of data to avoid overfitting.

### 2. Hardware (GPUs)
Matrix multiplications — the core operation of neural networks — are perfectly suited to GPUs. GPU chips (originally designed for graphics) enabled 100× speedup for neural network training.

### 3. Better algorithms
- **ReLU activation** (Krizhevsky et al., 2012): Fixes vanishing gradient problem
- **Better initialization** (He initialization, Xavier initialization): Networks train faster and more stably
- **Batch normalization** (Ioffe & Szegedy, 2015): Stabilizes training, allows higher learning rates
- **Better optimizers** (Adam, RMSprop): Faster convergence than vanilla SGD

Without all three, deep networks were unstable and slow. With all three, they became the dominant paradigm.

## The deep learning workflow: end-to-end learning

The beauty of deep learning is simplicity in principle:

1. Define a neural network architecture
2. Initialize weights randomly
3. Show it labeled data
4. Compute loss (difference between prediction and truth)
5. Backpropagate to update weights
6. Repeat until convergence

The network automatically discovers what features to learn. No human designs the features at any stage.

## Continuity with the next topics

This note gives the field-level difference. The progression ahead is:

1. **Note 3**: What kinds of architectures exist (ANN vs CNN vs RNN)?
2. **Note 4–7**: Building blocks (perceptron, why they fail)
3. **Note 8–10**: Network basics (notation, forward pass, how to represent data)
4. **Note 11–13**: First projects (train networks on real problems)
5. **Note 14–17**: Learning mechanism (loss, backpropagation)
6. **Note 18+**: Advanced techniques (normalization, optimizers, regularization)

The flow is:

$$
\text{Why DL?} \rightarrow \text{What architectures exist?} \rightarrow \text{How to build them?} \rightarrow \text{How to train them?}
$$

## Worked example: classical ML vs deep learning on MNIST

**Handwritten digit classification (MNIST: 28×28 pixel images)**

### Classical ML approach
```
Raw image (784 pixels)
  → Extract 100 hand-designed features (HOG, SIFT, etc.)
  → Train SVM classifier
  Accuracy: ~95%
```

### Deep learning approach
```
Raw image (784 pixels)
  → Layer 1: Learn edge detectors (64 filters)
  → Layer 2: Learn texture detectors (128 filters)
  → Layer 3: Dense layer (256 neurons, learns digit parts)
  → Output: 10 neurons (one per digit)
  Accuracy: ~99.5%
```

The deep network automatically discovers that edges → textures → digit parts → digit classification is the right hierarchy, without any human guidance.

## PyTorch comparison: classical ML vs deep learning code

```python
# ============================================================
# Classical ML: manually extract features, then classify
# ============================================================
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)  # 64 features already extracted
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Human chooses these features; this is the bottleneck
svm = SVC(kernel='rbf')
svm.fit(X, y)
accuracy = svm.score(X, y)  # ~97%

# ============================================================
# Deep Learning: learn features end-to-end
# ============================================================
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Raw pixel data: no manual feature engineering
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST('./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Define network: let it learn features
model = nn.Sequential(
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Train: network automatically discovers useful features
for epoch in range(5):
    for images, labels in train_loader:
        images = images.view(-1, 28*28)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Accuracy: ~99%
```

The key difference: the classical ML user manually chose SVM on handcrafted features. The deep learning user wrote a network and let it discover what features matter.

## Interview questions

<details>
<summary>Is deep learning always better than machine learning?</summary>

No. Deep learning excels on unstructured data (images, text, audio) and large datasets. Classical ML is often better on small, clean tabular datasets where interpretability matters and features are already meaningful.
</details>

<details>
<summary>Why is it called "deep" learning?</summary>

Because the model contains multiple stacked hidden layers. The learned representations become progressively deeper — each layer refines the representation from the previous layer. Early layers learn simple patterns; later layers (deeper in the network) learn abstract concepts.
</details>

<details>
<summary>What is representation learning?</summary>

It is the process where a model learns to automatically extract useful internal features (representations) from raw data, rather than relying on human-engineered features. Each hidden layer computes a new representation.
</details>

<details>
<summary>Why does deep learning need more compute?</summary>

Because it trains millions of parameters through many forward and backward passes. Each forward pass computes all hidden layer outputs; each backward pass computes gradients for all parameters. Classical ML methods (like decision trees) are much simpler computationally.
</details>

<details>
<summary>When should you use classical ML instead of deep learning?</summary>

When: (1) The dataset is small (&lt;1000 samples), (2) your features are already informative (structured tabular data), (3) interpretability is critical, (4) compute is limited, or (5) you don't have time for hyperparameter tuning.
</details>

<details>
<summary>Can deep learning features transfer between tasks?</summary>

Yes — this is a major advantage. A CNN trained on ImageNet (1.2 million images) learns features (edge detectors, texture detectors, etc.) that transfer to other vision tasks. You can fine-tune a pre-trained network on your own small dataset. Classical ML features rarely transfer this way.
</details>

<details>
<summary>Why didn't deep learning dominate in the 1990s if the ideas were known?</summary>

Because three necessary ingredients were missing: (1) Large labeled datasets didn't exist (ImageNet was created in 2009), (2) GPUs weren't widely available, (3) Training tricks (ReLU, good initialization, batch norm) hadn't been invented. Without all three, training deep networks was unreliable and slow.
</details>

## Common mistakes

- Thinking deep learning is a completely separate field from machine learning (it's a subset)
- Assuming more layers automatically mean better performance (depth helps, but training gets harder)
- Choosing deep learning for small tabular datasets (classical ML is better here)
- Treating hidden layers as magic instead of learned transformations
- Ignoring the importance of data quality — garbage in, garbage out applies to both ML and DL
- Forgetting that deep networks need regularization on small datasets to avoid overfitting

## Advanced perspective

From an information theory perspective, representation learning is about finding a compressed representation of the data that is sufficient for the downstream task. Early layers perform dimensionality reduction by identifying the most important patterns. Deeper layers compose these patterns into increasingly abstract concepts. This hierarchical compression is why deep networks can learn with fewer parameters than shallow, wide networks on many real-world tasks.

## Final takeaway

Classical machine learning relies on humans to design features; deep learning learns features automatically. This distinction doesn't make deep learning universally better — it makes it better for unstructured data and large-scale tasks where feature engineering is impractical. Understanding when each paradigm is appropriate is more important than assuming one is always superior.

## References

- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. NIPS 2012.
