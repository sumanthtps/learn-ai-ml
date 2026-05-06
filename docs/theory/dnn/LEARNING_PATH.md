---
id: dnn-learning-path
title: "DNN Learning Path: Zero to Hero"
sidebar_label: "Learning Path Guide"
sidebar_position: 0
slug: /theory/dnn/learning-path-zero-to-hero
description: "Recommended reading order and learning sequences for mastering deep neural networks from fundamentals to transformers."
---

# Deep Neural Networks: Zero to Hero Learning Path

This guide shows you the optimal order to read the 95 DNN notes, grouped by learning stage. Each stage builds on the previous one.

## Stage 1: Foundations (3 hours)

**Goal**: Understand what deep learning is and why it works.

| File | Title | Time | Key Concept |
|------|-------|------|---|
| **01** | Course Roadmap | 20m | Understand the overall structure |
| **02** | DL vs ML | 30m | Learn why representation learning matters |
| **03** | NN Types | 25m | See the landscape: ANN, CNN, RNN, Transformer |
| **04** | Perceptron Basics | 35m | First learnable classifier |
| **05** | Perceptron Training | 30m | How weights update (the perceptron trick) |
| **06** | Perceptron Losses | 25m | Why sigmoid + BCE works together |
| **07** | Why Perceptron Fails | 20m | XOR problem motivation |
| **08** | MLP Notation | 20m | Standardize mathematical notation |
| **09** | MLP Intuition | 40m | **Why hidden layers solve XOR** |

**Checkpoint**: You should understand:
- Why deep learning learns features automatically
- Why a single perceptron cannot solve XOR
- How hidden layers transform data into linearly separable representations
- Mathematical notation for weights, biases, activations

**Next**: Forward references in these files point to Stage 2

---

## Stage 2: Forward Propagation & Loss (2.5 hours)

**Goal**: Understand how a network makes predictions and measures error.

| File | Title | Time | Key Concept |
|------|-------|------|---|
| **10** | Forward Propagation | 30m | Input → hidden layers → output |
| **11-13** | Projects (Churn, MNIST, Admissions) | 45m | Apply forward pass to real data |
| **14** | Loss Functions | 40m | **MSE, MAE, Huber, BCE, CrossEntropy** |
| **19** | MLP Memoization | 25m | Cache activations for backprop |

**Checkpoint**: You should understand:
- How input flows through each layer
- How to choose loss based on task (regression vs classification)
- Why CrossEntropy + softmax is standard
- The relationship between loss gradient and weight updates

**Why this matters**: You now understand the forward problem. Next: how to reverse it (backprop).

---

## Stage 3: Backpropagation & Gradient Descent (3 hours)

**Goal**: Understand how weights are learned.

| File | Title | Time | Key Concept |
|------|-------|------|---|
| **15** | Backprop Part 1 | 35m | What backpropagation is (intuition) |
| **16** | Backprop Part 2 | 40m | **Chain rule mechanics** |
| **17** | Backprop Part 3 | 35m | Why backprop works (gradient flow) |
| **18** | Vanishing/Exploding Gradients | 40m | Failure modes of backprop |
| **20** | Gradient Descent Variants | 35m | Batch SGD, mini-batch, momentum |

**Checkpoint**: You should understand:
- How to compute $\frac{\partial \ell}{\partial W}$ for each layer
- The chain rule in neural networks
- Why gradients vanish in deep networks
- The difference between batch, SGD, and mini-batch training

**Critical**: Files 15-17 form a trilogy. Read in order. File 18 shows why this matters.

---

## Stage 4: Improving Training (2.5 hours)

**Goal**: Learn techniques that make neural networks train faster and more stably.

| File | Title | Time | Key Concept |
|------|-------|------|---|
| **21** | Improve Performance | 35m | Regularization overview |
| **22** | Early Stopping | 30m | Prevent overfitting by monitoring validation |
| **23** | Data Scaling | 30m | Normalize features (critical for convergence) |
| **24-25** | Dropout | 35m | **Stochastic regularization** |
| **26** | Regularization (L1/L2) | 30m | Weight decay prevents large weights |
| **27-28** | Activation Functions | 35m | ReLU, sigmoid, tanh, SELU, variants |

**Checkpoint**: You should understand:
- Why regularization prevents overfitting
- How dropout and weight decay work mechanically
- Why ReLU is better than sigmoid for deep networks
- How data scaling affects convergence

**Theme**: All about training stability and generalization.

---

## Stage 5: Weight Initialization & Normalization (2 hours)

**Goal**: Understand why the starting point matters.

| File | Title | Time | Key Concept |
|------|-------|------|---|
| **29** | Weight Init Basics | 35m | Why zero/random initialization matters |
| **30** | Xavier & He Init | 40m | **Principled initialization for different activations** |
| **31** | Batch Normalization | 35m | Normalize activations during training |
| **79** | LayerNorm vs BatchNorm | 40m | When to use which normalization |

**Checkpoint**: You should understand:
- Why $\text{Var}(W) = 2/(n_{in} + n_{out})$ (Xavier)
- Why $\text{Var}(W) = 2/n_{in}$ for ReLU (He)
- How BatchNorm stabilizes training
- When LayerNorm is better (transformers)

**Flow**: 29 → 30 (initialization), then 31 → 79 (normalization).

---

## Stage 6: Optimizers (1.5 hours)

**Goal**: Understand algorithms for finding good minima.

| File | Title | Time | Key Concept |
|------|-------|------|---|
| **32** | Optimizers Overview | 40m | Why plain SGD fails (ravines, saddle points) |
| **33** | Exponential Moving Average | 25m | Building block for adaptive methods |
| **34** | SGD with Momentum | 25m | **Accelerate through ravines** |
| **35** | Nesterov | 20m | Lookahead gradient |
| **36** | Adagrad | 20m | Per-parameter adaptive learning rates |
| **37** | RMSprop | 20m | Decay old gradient information |
| **38** | Adam | 25m | **Momentum + adaptive (industry standard)** |
| **39** | Hyperparameter Tuning | 30m | Learning rate, batch size, schedules |

**Checkpoint**: You should understand:
- Why momentum helps in ravines
- Why adaptive methods (Adam, RMSprop) help with sparse gradients
- When SGD+momentum is better than Adam for generalization
- How to tune learning rate and batch size

**Pattern**: Each optimizer solves a specific problem. Understand the problem, then the solution.

---

## Stage 7: Convolutional Neural Networks (3.5 hours)

**Goal**: Understand networks optimized for spatial data (images).

| File | Title | Time | Key Concept |
|------|-------|------|---|
| **40** | What is CNN | 40m | **Why dense networks fail on images** |
| **41** | Visual Cortex & Cat Experiment | 30m | Biological inspiration |
| **42** | Convolution Operation | 35m | Filters, feature maps, local connectivity |
| **43** | Padding & Stride | 30m | Control spatial dimensions |
| **44** | Pooling (Max Pooling) | 30m | **Downsampling for invariance** |
| **45** | LeNet-5 | 35m | First successful CNN architecture |
| **46** | CNN vs ANN | 25m | Comparison: parameters, receptive fields |
| **47-48** | Backprop in CNNs | 50m | How gradients flow through conv layers |
| **49** | Cat vs Dog Project | 40m | Implement and train on real images |
| **50** | Data Augmentation | 30m | Generate synthetic training data |
| **51** | Pretrained Models | 35m | Use ImageNet-trained CNNs |
| **52** | What CNN Sees | 30m | Visualize filters and feature maps |
| **53** | Transfer Learning | 40m | **Fine-tune vs feature extraction** |
| **54** | PyTorch Functional Style | 35m | nn.Module vs F.* functions |

**Checkpoint**: You should understand:
- How convolution differs from dense layers (weight sharing, local connectivity)
- Why pooling provides spatial invariance
- The LeNet architecture pattern (still used in ResNets)
- When to use transfer learning vs training from scratch

**Key insight**: CNNs are just ANNs with a different connectivity pattern.

---

## Stage 8: Recurrent Neural Networks (3 hours)

**Goal**: Understand networks for sequential data.

| File | Title | Time | Key Concept |
|------|-------|------|---|
| **55** | Why RNNs | 40m | **ANN fails on sequences** |
| **56** | RNN Architecture | 35m | Recurrence relation, hidden state |
| **57** | Sentiment Analysis (RNN) | 35m | Apply RNN to text |
| **58** | RNN Mappings | 30m | One-to-one, many-to-one, many-to-many |
| **59** | Backprop Through Time | 40m | **Unroll and apply backprop** |
| **60** | RNN Problems | 25m | Vanishing/exploding gradients worse in RNNs |
| **61-62** | LSTM | 50m | **Gates solve gradient problems** |
| **63** | LSTM Project (Next Word Pred) | 35m | Build language model |
| **64** | GRU | 35m | Simplified LSTM |
| **65** | Deep RNNs | 30m | Stack multiple RNN layers |
| **66** | Bidirectional RNNs | 30m | Process sequences in both directions |

**Checkpoint**: You should understand:
- Why vanilla RNNs fail (vanishing gradients through time)
- How LSTM gates (input, forget, output) control information flow
- The difference between LSTM and GRU
- When bidirectional processing helps

**Pattern**: Vanilla RNN → LSTM (fix gradient problem) → Transformers (parallelize).

---

## Stage 9: Attention & Transformers (4 hours)

**Goal**: Understand the modern architecture powering LLMs.

| File | Title | Time | Key Concept |
|------|-------|------|---|
| **67** | History of LLMs | 35m | LSTM → attention → Transformers |
| **68** | Seq2Seq Encoder-Decoder | 40m | Two RNNs for translation |
| **69** | Attention Mechanism | 45m | **Query, key, value for seq2seq** |
| **70** | Bahdanau vs Luong | 30m | Different attention variants |
| **71** | Intro to Transformers | 35m | Replace RNN with self-attention |
| **72** | What is Self-Attention | 40m | **Same sequence attends to itself** |
| **73** | Self-Attention Code | 30m | Implement from scratch |
| **74** | Scaled Dot-Product Attention | 30m | Stable attention computation |
| **75** | Geometric Intuition | 35m | Why attention is similarity-based |
| **76** | Why "Self" Attention | 25m | Q, K, V all from same sequence |
| **77** | Multi-Head Attention | 35m | **Parallel attention heads** |
| **78** | Positional Encoding | 40m | Inject position information |
| **80** | Transformer Encoder | 35m | Stack of self-attention + FFN blocks |
| **81** | Masked Self-Attention | 30m | Prevent attending to future tokens |
| **82** | Cross-Attention | 30m | Query from one sequence, K/V from another |
| **83** | Transformer Decoder | 40m | Auto-regressive generation |
| **84** | Transformer Inference | 40m | **Generate token by token** |
| **85** | Training Objectives | 35m | Teacher forcing, causal masking |

**Checkpoint**: You should understand:
- Why self-attention is a replacement for recurrence
- How multi-head attention parallelize computation
- The encoder-decoder pattern
- Causal masking for left-to-right generation

**Theme**: From sequential (LSTM) to parallel (Transformer) computation.

---

## Stage 10: Large Language Models (2.5 hours)

**Goal**: Understand pretraining and fine-tuning of LLMs.

| File | Title | Time | Key Concept |
|------|-------|------|---|
| **86** | Tokenization | 40m | **BPE, WordPiece, SentencePiece** |
| **87** | BERT (Encoder Pretraining) | 40m | Masked language model |
| **88** | GPT (Decoder-Only) | 40m | **Causal language model** |
| **89** | T5 (Encoder-Decoder) | 35m | Text-to-text framework |
| **90** | Fine-tuning | 40m | **Adapt pretrained models** |
| **91** | LoRA (Parameter-Efficient) | 35m | Low-rank adaptation |
| **92** | FlashAttention | 30m | Optimize attention computation |
| **93** | Scaling Laws | 35m | How performance scales with size |
| **94** | In-Context Learning | 35m | Prompting and few-shot learning |
| **95** | RLHF & Instruction Tuning | 35m | Make LLMs follow instructions |

**Checkpoint**: You should understand:
- Differences between BERT (bidirectional), GPT (causal), T5 (seq2seq)
- Why pretrain on raw text, then fine-tune on task
- When LoRA is better than full fine-tuning
- How scaling laws predict performance

**Summit**: LLMs are transformers + pretraining + scale. You now understand all three.

---

## Recommended Reading Schedules

### **Intensive (2 weeks, 40 hours)**
Complete all 10 stages. Best for full-time learners.

### **Part-Time (8 weeks, 40 hours)**
Stages 1-3 (week 1-2), Stages 4-5 (week 3-4), Stages 6 (week 5), Stages 7 (week 6), Stages 8-9 (week 7), Stage 10 (week 8).

### **Just Transformers (3 days)**
Skip to Stage 1 (foundations) + Stage 9 (transformers). ~8 hours.

### **Just CNNs (2 days)**
Stages 1-3 (foundations) + Stage 7 (CNNs). ~6 hours.

### **Practitioner's Path (2 weeks)**
Stages 1-3 (understand basics), Stage 4 (training tricks), Stage 6 (optimizers), Stage 7 (CNNs), Stage 9 (transformers), Stage 10 (LLMs).

---

## Learning Strategies

### **Read Actively**
- Don't just read the math — recreate it
- Run the PyTorch code
- Ask "why?" at each step

### **Use Checkpoints**
After each stage, code up the concepts without looking:
- Stage 1 checkpoint: Implement XOR-solving MLP
- Stage 3 checkpoint: Implement backprop from scratch
- Stage 7 checkpoint: Build a 3-layer CNN
- Stage 9 checkpoint: Implement self-attention

### **Connect to Projects**
- Files 11-13 (ANN projects)
- File 49 (CNN project)
- Files 57, 63 (RNN/LSTM projects)
- Suggest: Build a sentiment classifier (RNN) and image classifier (CNN)

### **Read Interview Questions**
Each file has 5-8 interview questions at the bottom. These reveal what's important.

---

## Continuity Links

Forward references (what's coming next):
- File 02 → File 03 (what architectures exist?)
- File 09 → File 10 (how does input flow?)
- File 14 → File 15 (now learn backprop)
- File 30 → File 31 (initialization → normalization)
- File 40 → File 42 (what is CNN? → how convolution works)
- File 55 → File 56 (why RNN? → how RNN architecture)
- File 71 → File 72 (what is transformer? → what is self-attention?)

Backward references (where did this come from?):
- File 10 references File 09 (MLP intuition)
- File 15 references Files 09-10 (forward pass must be understood first)
- File 40 references Files 09-10 (CNNs extend MLPs)
- File 55 references Files 09-10 (RNNs extend MLPs)

---

## Knowledge Map: How Topics Connect

```
Foundations (01-09)
    ↓
Forward Pass & Loss (10-14)
    ↓
Backpropagation (15-20)
    ↓
    ├→ Training Tricks (21-28)
    │   ↓
    │   Initialization & Normalization (29-31, 79)
    │   ↓
    │   Optimizers (32-39)
    │   
    ├→ CNNs (40-54)
    │   └→ Transfer Learning (53)
    │
    └→ RNNs (55-66)
        ↓
        LSTMs (61-63)
        ↓
        Attention (69-70)
        ↓
        Transformers (71-85)
        ↓
        LLMs (86-95)
```

--- 

## File Difficulty Levels

**Beginner** (no prerequisites beyond Stage 1):
Files 02-10, 40-42, 55-56, 71-72

**Intermediate** (need stages 1-3):
Files 14-20, 24-28, 43-54, 59-66, 73-78

**Advanced** (need full understanding):
Files 30-31, 79, 81-95

---

## Final Notes

- **Don't skip foundation stages** — they form the conceptual basis for everything
- **Coding is essential** — reading is 20%, coding is 80% of learning
- **Return to earlier files** — as you learn, earlier files reveal deeper meaning
- **Interview questions are hints** — if you can't answer them, reread the file

The journey is: **perception → intuition → mathematics → implementation → mastery**.

Good luck! 🚀
