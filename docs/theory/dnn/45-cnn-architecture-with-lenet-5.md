---
id: cnn-architecture-lenet5
title: "CNN architecture with LeNet-5"
sidebar_label: "45 · LeNet-5"
sidebar_position: 45
slug: /theory/dnn/cnn-architecture-with-lenet-5
description: "LeNet-5 layer by layer: convolutions, subsampling, fully connected layers, and the design decisions that made it the template for all modern CNNs."
tags: [cnn, lenet, architecture, convolution, pooling, deep-learning]
---

# CNN architecture with LeNet-5

LeNet-5 (LeCun et al., 1998) was the first CNN trained end-to-end with backpropagation on a real problem — handwritten digit classification. It achieved near-human accuracy on MNIST decades before deep learning became widespread. Every modern CNN (AlexNet, VGG, ResNet) inherits LeNet's core design: alternating convolution and pooling blocks that extract increasingly abstract features, followed by fully connected layers for classification.

## One-line definition

LeNet-5 is a 7-layer CNN with two conv-pool blocks followed by three fully connected layers, designed to classify $32 \times 32$ grayscale images of handwritten digits.

## Architecture overview

```
Input: 1 × 32 × 32 (grayscale digit)

C1: Conv(1→6, K=5, S=1, P=0)    → 6 × 28 × 28   [6 feature maps, 5×5 kernels]
S2: AvgPool(K=2, S=2)            → 6 × 14 × 14   [subsampling]
C3: Conv(6→16, K=5, S=1, P=0)   → 16 × 10 × 10  [16 feature maps]
S4: AvgPool(K=2, S=2)            → 16 × 5 × 5    [subsampling]
C5: Conv(16→120, K=5, S=1, P=0) → 120 × 1 × 1   [full connection by convolution]
F6: Linear(120 → 84)             → 84
Output: Linear(84 → 10)          → 10 logits
```

![LeNet-5 full architecture — two conv-pool blocks followed by fully connected layers for digit classification](https://commons.wikimedia.org/wiki/Special:Redirect/file/LeNet-5_architecture.svg)
*Source: [Wikimedia Commons — LeNet-5 architecture](https://commons.wikimedia.org/wiki/File:LeNet-5_architecture.svg) (CC BY-SA 4.0)*

## Layer-by-layer breakdown

### C1 — First convolution

- Input: $1 \times 32 \times 32$
- 6 filters of size $5 \times 5$, stride 1, no padding
- Output: $6 \times 28 \times 28$
- Parameters: $6 \times (5 \times 5 \times 1 + 1) = 156$

The 6 filters learn low-level features: edges, curves, blobs.

### S2 — First subsampling

- $2 \times 2$ average pooling, stride 2
- Output: $6 \times 14 \times 14$
- No learnable parameters (LeCun used a learned multiplier per channel, but modern implementations use standard AvgPool)

### C3 — Second convolution

- 16 filters of size $5 \times 5$
- Output: $16 \times 10 \times 10$
- Parameters: $16 \times (5 \times 5 \times 6 + 1) = 2{,}416$

In the original paper, C3 does not connect to all 6 S2 channels — only specific combinations — to reduce parameters. Modern implementations use full connectivity.

### S4 — Second subsampling

- $2 \times 2$ average pooling, stride 2
- Output: $16 \times 5 \times 5$

### C5 — Third convolution (effectively fully connected)

- 120 filters of size $5 \times 5$
- Input is $16 \times 5 \times 5$, so the kernel covers the entire spatial extent
- Output: $120 \times 1 \times 1$ — equivalent to a dense layer
- Parameters: $120 \times (5 \times 5 \times 16 + 1) = 48{,}120$

### F6 — Fully connected

- Linear: $120 \to 84$
- Activation: tanh in the original (ReLU in modern practice)
- Parameters: $120 \times 84 + 84 = 10{,}164$

### Output

- Linear: $84 \to 10$
- Original: Euclidean RBF units. Modern: softmax cross-entropy
- Parameters: $84 \times 10 + 10 = 850$

## Parameter count summary

| Layer | Output shape | Parameters |
|---|---|---|
| C1 Conv(1→6, 5×5) | 6×28×28 | 156 |
| S2 AvgPool(2) | 6×14×14 | 0 |
| C3 Conv(6→16, 5×5) | 16×10×10 | 2,416 |
| S4 AvgPool(2) | 16×5×5 | 0 |
| C5 Conv(16→120, 5×5) | 120×1×1 | 48,120 |
| F6 Linear(120→84) | 84 | 10,164 |
| Output Linear(84→10) | 10 | 850 |
| **Total** | | **~61,706** |

Compare to an MLP on the same task: a dense network with one hidden layer of 84 neurons would need $32 \times 32 \times 84 = 86{,}016$ parameters just for the first layer, and loses all spatial structure.

## Key design principles introduced by LeNet-5

1. **Local connectivity**: each neuron connects to a small local patch, not all inputs
2. **Weight sharing**: all positions in a feature map share the same filter weights — translation equivariance
3. **Hierarchical feature extraction**: early layers detect edges, later layers detect digits
4. **Subsampling for invariance**: pooling makes the representation invariant to small spatial shifts
5. **Alternating conv-pool**: the conv-pool-conv-pool pattern became the template for all subsequent CNNs

## PyTorch implementation

```python
import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    LeNet-5 for MNIST / grayscale 32×32 input.
    Updated with ReLU (original used tanh/sigmoid).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Feature extractor: two conv-pool blocks
        self.features = nn.Sequential(
            # C1: 1×32×32 → 6×28×28
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            # S2: 6×28×28 → 6×14×14
            nn.AvgPool2d(kernel_size=2, stride=2),

            # C3: 6×14×14 → 16×10×10
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            # S4: 16×10×10 → 16×5×5
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        # C5: 16×5×5 → 120 (via 5×5 conv that covers entire spatial extent)
        self.c5 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5),   # → 120×1×1
            nn.ReLU(),
        )
        # Classifier: F6 + output
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # → (B, 16, 5, 5)
        x = self.c5(x)                # → (B, 120, 1, 1)
        x = x.flatten(1)              # → (B, 120)
        return self.classifier(x)     # → (B, 10)


# Verify shapes at each stage
model = LeNet5()
x = torch.randn(8, 1, 32, 32)    # batch of 8 MNIST-style images

# Trace through manually
with torch.no_grad():
    f = model.features(x)
    print(f"After features: {f.shape}")   # (8, 16, 5, 5)
    c = model.c5(f)
    print(f"After C5:       {c.shape}")   # (8, 120, 1, 1)
    out = model.classifier(c.flatten(1))
    print(f"Output:         {out.shape}") # (8, 10)

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total:,}")   # ~61,706


# ============================================================
# Training on MNIST
# ============================================================
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# Uncomment to train:
# train_dataset = datasets.MNIST('./data', train=True, transform=transform, download=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# criterion = nn.CrossEntropyLoss()
#
# for epoch in range(10):
#     for x, y in train_loader:
#         optimizer.zero_grad()
#         loss = criterion(model(x), y)
#         loss.backward()
#         optimizer.step()
```

## LeNet-5 vs modern CNNs

| Feature | LeNet-5 (1998) | Modern CNN (ResNet, 2015) |
|---|---|---|
| Activation | tanh / sigmoid | ReLU |
| Pooling | average pooling | max pooling or strided conv |
| Normalization | none | batch normalization |
| Depth | 7 layers | 50–152 layers |
| Skip connections | none | residual connections |
| Parameters | ~61K | 25M (ResNet-50) |
| Input | 32×32 grayscale | 224×224 RGB |
| Training | SGD | SGD/Adam + lr schedule |

The conceptual pattern — alternating convolution and downsampling, followed by dense classification — is identical. ResNet is LeNet with depth, skip connections, and batch norm.

## Interview questions

<details>
<summary>Why does LeNet-5 use average pooling while modern CNNs prefer max pooling or strided convolution?</summary>

LeCun used average pooling (subsampling in his terminology) combined with a learnable multiplier and bias per channel. Average pooling produces smoother gradients than max pooling. Max pooling became standard in AlexNet because it preserves the strongest activation (better for detecting whether a feature is present) and empirically performs better for classification. Today, strided convolution is often preferred because it is learnable — the network decides how to downsample rather than using a fixed rule.
</details>

<details>
<summary>What does weight sharing accomplish in a CNN, and how does LeNet-5 use it?</summary>

Weight sharing means all spatial positions in a feature map use the same filter weights. A single $5 \times 5$ filter is convolved across the entire $28 \times 28$ image, detecting the same pattern wherever it appears. Without sharing, each position would need its own $5 \times 5$ weights — 784 times more parameters. Weight sharing gives two benefits: drastically fewer parameters and translation equivariance — the feature map responds identically to the same pattern regardless of its position in the image.
</details>

<details>
<summary>Why does C5 in LeNet-5 use a convolution rather than a linear layer?</summary>

After S4, the feature map is $16 \times 5 \times 5$. Using a $5 \times 5$ conv filter produces a $1 \times 1$ output — mathematically identical to a dense linear layer over the flattened $16 \times 5 \times 5 = 400$ inputs. LeCun used the conv formulation to maintain the interpretation of local receptive fields and to make the network work with variable input sizes more naturally. Modern implementations typically just flatten and use a linear layer, which is equivalent.
</details>

## Common mistakes

- Using a $28 \times 28$ input (standard MNIST size) without padding — the network expects $32 \times 32$. Use `transforms.Resize(32)` or add 2 pixels of padding
- Forgetting that C5 requires the spatial size to be exactly $5 \times 5$ — if input is not $32 \times 32$, you will get a size mismatch at C5
- Not normalizing MNIST input — training is much slower without normalization

## Final takeaway

LeNet-5 introduced the CNN design pattern: local convolutions with weight sharing extract spatial features, alternating pooling provides spatial invariance, and fully connected layers classify the learned representations. With only ~62K parameters, it achieved near-human accuracy on MNIST in 1998. Every modern CNN (AlexNet → VGG → ResNet → EfficientNet) inherits this pattern while adding depth, skip connections, batch normalization, and better activations.

## References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE.
