---
id: scaling-laws
title: "Transformer scaling laws"
sidebar_label: "93 · Scaling Laws"
sidebar_position: 93
slug: /theory/dnn/transformer-scaling-laws
description: "How model performance scales with parameters, data, and compute — the Kaplan (GPT-3) scaling laws, the Chinchilla correction, and practical implications for training LLMs."
tags: [scaling-laws, chinchilla, llm, pre-training, transformers, deep-learning]
---

# Transformer scaling laws

> **TL;DR.** Scaling laws are **power-law relationships** between training loss and three knobs: parameters N, tokens D, and compute C. Kaplan et al. (OpenAI, 2020) said "make models bigger". Chinchilla (DeepMind, 2022) corrected it: for a fixed compute budget, **N and D should grow together** — train smaller models on more tokens. The "Chinchilla-optimal" rule of thumb: D ≈ 20 × N (20 tokens per parameter). This single insight is why LLaMA 7B trained on 1T+ tokens beats GPT-3 175B at many tasks despite being 25× smaller.

Scaling laws are empirical equations that predict how a language model's loss changes as a function of model size, training data size, and compute budget. They are the reason the LLM field moved from 100M to 100B+ parameter models in five years — because the laws predicted it would work before anyone built the models.

## Try it interactively

- **[Epoch AI — Compute trends](https://epochai.org/data/notable-ai-models)** — explore frontier model training compute over time, plotted against scaling-law predictions
- **[Chinchilla paper Colab](https://github.com/google-deepmind/chinchilla)** — re-derive the optimal N/D split for any compute budget
- **[Hoffmann et al. interactive viewer](https://arxiv.org/abs/2203.15556)** — the Chinchilla paper itself, with all loss curves
- **[Compute-optimal calculator](https://www.lesswrong.com/posts/midXmMb2Xg37F2Kgn/new-scaling-laws-for-large-language-models)** — given budget, compute optimal N and D
- **[Notable LLMs leaderboard](https://lmsys.org/blog/2024-08-13-empirical-scaling-laws/)** — frontier scaling research, updated regularly

## One-line definition

Scaling laws are power-law relationships between a model's training loss and the number of parameters $N$, training tokens $D$, and compute $C$ — showing that performance improves predictably as any of these quantities increases.

![BERT BASE (110M params, 12 layers) vs BERT LARGE (340M params, 24 layers) — scaling laws predict the precise performance gain from this 3× parameter increase](https://jalammar.github.io/images/bert-base-bert-large-encoders.png)
*Source: [Jay Alammar — The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)*

## Why this topic matters

Scaling laws answer the most important question in LLM training: given a fixed compute budget, how do I allocate it between model size and data? They are used by every major AI lab to plan training runs. They explain why LLaMA 3 (8B parameters, 15T tokens) outperforms GPT-3 (175B, 300B tokens): more data per parameter is often more efficient than more parameters.

## The original Kaplan scaling laws (2020)

Kaplan et al. (OpenAI, 2020) trained hundreds of models to identify power-law scaling:

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}
$$

where $L$ is the language modeling loss (cross-entropy), $N$ is the number of non-embedding parameters, $D$ is the number of training tokens, and $\alpha_N \approx \alpha_D \approx 0.076$.

**Key findings**:
1. Loss follows a power law in model size $N$ (double $N$ → loss decreases by $\sim 5\%$)
2. Loss follows a power law in data size $D$ (double data → similar $5\%$ improvement)
3. Model size should scale faster than data: for a given compute budget, use a larger model but train it for fewer tokens

**Compute-optimal under Kaplan**: given compute $C \propto N \times D$, loss is minimized by allocating most compute to model size:

$$
N_{\text{opt}} \propto C^{0.73}, \quad D_{\text{opt}} \propto C^{0.27}
$$

GPT-3 followed this guidance: 175B parameters, 300B tokens (only ~1.7 tokens per parameter).

## The Chinchilla correction (2022)

Hoffmann et al. (DeepMind, 2022) found that Kaplan's recommendation was wrong for the practical regime. They trained over 400 models with up to 67B parameters and up to 1.4T tokens and found:

$$
N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}
$$

**The Chinchilla rule**: model size and training tokens should scale equally. For compute-optimal training:

$$
D_{\text{opt}} \approx 20 \times N
$$

**20 tokens per parameter** is the compute-optimal ratio.

| Model | Parameters | Training tokens | Tokens/param | Compute-optimal? |
|---|---|---|---|---|
| GPT-3 (2020) | 175B | 300B | 1.7× | Under-trained by 10× |
| Chinchilla (2022) | 70B | 1.4T | 20× | Yes (Chinchilla-optimal) |
| LLaMA 2 7B (2023) | 7B | 2T | 285× | Over-trained (better for inference) |
| LLaMA 3 8B (2024) | 8B | 15T | 1875× | Heavily over-trained |

**The twist**: Chinchilla-optimal means minimizing loss for a given training compute budget. But for inference-heavy deployments, it may be better to **over-train a smaller model** — a smaller but better-trained model is cheaper to serve than a larger, less-trained model with the same performance.

This is the insight behind LLaMA: train a 7B model for 2T tokens (far more than Chinchilla-optimal) to get a small, fast model that outperforms much larger, under-trained models.

## The scaling law formula

The combined compute-optimal loss (Chinchilla formulation):

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

where:
- $E \approx 1.69$: irreducible entropy (the minimum possible loss on web text)
- $A, \alpha$: parameters governing model size scaling
- $B, \beta$: parameters governing data scaling
- Fitted values: $\alpha \approx 0.34$, $\beta \approx 0.28$, $A = 406.4$, $B = 410.7$

The $E$ term is irreducible loss — even an infinite model trained on infinite data cannot achieve loss below this threshold, because natural language has inherent uncertainty.

## Emergent capabilities

Scaling also produces qualitative phase transitions — capabilities that are absent at small scale and appear suddenly at larger scale:

| Capability | Approximate emergence | Example |
|---|---|---|
| In-context learning (few-shot) | ~few billion parameters | GPT-3 |
| Chain-of-thought reasoning | ~50–100B parameters | GPT-4 |
| Instruction following | Fine-tuning dependent | ChatGPT |
| Code generation | ~10B+ parameters | Codex |
| Multi-step arithmetic | ~100B+ parameters | GPT-4 |

These are called "emergent" because they are not predictable from scaling laws alone — they appear as discontinuous jumps rather than smooth power-law improvements.

## Python code: visualizing scaling behavior

```python
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Chinchilla scaling law: L(N, D) = E + A/N^alpha + B/D^beta
# ============================================================
E = 1.69      # irreducible entropy
A = 406.4
B = 410.7
alpha = 0.34
beta = 0.28


def loss(N, D):
    """Predicted cross-entropy loss given N parameters and D training tokens."""
    return E + A / (N ** alpha) + B / (D ** beta)


# Explore the loss surface
N_values = np.logspace(8, 12, 50)   # 100M to 1T parameters
D_values = np.logspace(8, 13, 50)   # 100M to 10T tokens


# ============================================================
# Chinchilla-optimal allocation for a fixed compute budget
# ============================================================
def optimal_allocation(compute_flops: float,
                        flops_per_token_per_param: float = 6.0):
    """
    Given a compute budget C ≈ 6 × N × D FLOPs,
    find the Chinchilla-optimal (N, D) pair.
    flops ≈ 6 × N × D (forward + backward for transformer)
    """
    # Chinchilla: N_opt = C^0.5 / sqrt(6), D_opt = C^0.5 / sqrt(6) * sqrt(6)
    # Simplified: N_opt ≈ D_opt ≈ sqrt(C / 6)
    N_opt = np.sqrt(compute_flops / flops_per_token_per_param)
    D_opt = compute_flops / (flops_per_token_per_param * N_opt)
    predicted_loss = loss(N_opt, D_opt)
    return N_opt, D_opt, predicted_loss


print("Chinchilla-optimal allocations:")
for compute_budget_petaflops in [1, 10, 100, 1000]:
    C = compute_budget_petaflops * 1e15   # convert to FLOPs
    N_opt, D_opt, L_opt = optimal_allocation(C)
    print(f"  {compute_budget_petaflops:5d} PF-days: "
          f"N={N_opt/1e9:.1f}B params, D={D_opt/1e9:.0f}B tokens, "
          f"loss={L_opt:.3f}, perplexity={np.exp(L_opt):.1f}")


# ============================================================
# Compare real models to Chinchilla predictions
# ============================================================
real_models = {
    "GPT-3":        {"N": 175e9, "D": 300e9,  "reported_ppl": 20.5},
    "Chinchilla":   {"N": 70e9,  "D": 1.4e12, "reported_ppl": 7.3},
    "LLaMA 2 7B":  {"N": 7e9,   "D": 2e12,   "reported_ppl": None},
    "LLaMA 3 8B":  {"N": 8e9,   "D": 15e12,  "reported_ppl": None},
}

print("\nPredicted loss for real models (Chinchilla formula):")
for name, info in real_models.items():
    L = loss(info["N"], info["D"])
    tokens_per_param = info["D"] / info["N"]
    compute = 6 * info["N"] * info["D"]
    print(f"  {name:20s}: predicted loss={L:.3f}, "
          f"tokens/param={tokens_per_param:.0f}, "
          f"compute≈{compute/1e21:.1f} ZFLOPs")


# ============================================================
# Visualization: loss vs. model size at fixed data
# ============================================================
D_fixed = 300e9   # 300B tokens (GPT-3 scale)
Ns = np.logspace(7, 12, 100)
losses_fixed_data = [loss(n, D_fixed) for n in Ns]

plt.figure(figsize=(8, 5))
plt.loglog(Ns / 1e9, losses_fixed_data, "b-", linewidth=2)
plt.xlabel("Model size (billions of parameters)")
plt.ylabel("Predicted cross-entropy loss")
plt.title(f"Loss vs. Model Size (D = 300B tokens)")
plt.grid(True, which="both", alpha=0.3)

# Mark specific models
for name, info in real_models.items():
    l = loss(info["N"], D_fixed)
    plt.scatter(info["N"] / 1e9, l, s=100, zorder=5)
    plt.annotate(name, (info["N"] / 1e9, l), textcoords="offset points",
                 xytext=(5, 5), fontsize=8)

plt.tight_layout()
plt.savefig("scaling_law.png", dpi=150)
# plt.show()  # uncomment to display
```

## Practical implications

**Given a compute budget, what should you do?**

1. **Small compute (< 1 PF-day)**: use an existing pre-trained model. Training from scratch is wasteful.

2. **Medium compute (1–100 PF-days)**: train a model around the Chinchilla-optimal size. Balance parameters and data ~equally.

3. **Large compute (> 100 PF-days)**: train a smaller-than-Chinchilla-optimal model for more tokens. The inference cost savings outweigh the training inefficiency for widely deployed models.

4. **Domain-specific applications**: always fine-tune or continue pre-training on domain data rather than training from scratch — transfer from a general model is far more compute-efficient.

## Scaling beyond language

Scaling laws have been validated in:
- **Vision**: ViT performance scales predictably with model size and image data
- **Code**: Codex and Code Llama scale well with code-specific data
- **Multimodal**: Gemini, GPT-4V — scaling applies across modalities
- **Reasoning**: some reasoning benchmarks show slower scaling (emergent rather than smooth)

## Interview questions

<details>
<summary>What is the Chinchilla finding and how did it change LLM training practice?</summary>

Kaplan et al. (2020) recommended allocating most compute to model size (large models, few tokens). Hoffman et al. (2022) showed this was wrong: for compute-optimal training, model size and training tokens should scale equally, with the rule of thumb being 20 tokens per parameter. The practical implication: GPT-3 (175B params, 300B tokens = 1.7 tokens/param) was dramatically under-trained. A 70B model trained on 1.4T tokens (Chinchilla) achieved better performance with less inference cost. This led to a shift toward smaller but better-trained models (LLaMA, Mistral), which now power most production deployments.
</details>

<details>
<summary>Why do LLaMA models train for far more tokens than Chinchilla-optimal?</summary>

Chinchilla-optimal minimizes loss for a given training compute budget. But it ignores inference cost. A 7B model with 2T tokens (285 tokens/param) is much smaller and faster to serve than a 70B model with 300B tokens (4 tokens/param), even if both achieve similar perplexity. When a model is deployed to millions of users, inference compute dominates. It is more economical to spend more compute during training to get a smaller, faster-to-serve model. LLaMA's "over-training" trades training compute for inference efficiency.
</details>

<details>
<summary>What are emergent capabilities and why don't scaling laws predict them?</summary>

Scaling laws describe smooth power-law decreases in loss. Emergent capabilities are qualitative skills — few-shot learning, chain-of-thought reasoning, arithmetic — that appear suddenly at certain scales rather than improving smoothly. They are not visible in the aggregate loss metric (perplexity) because they correspond to specific subcapabilities that require crossing a threshold. Scaling laws predict when the model will achieve a certain loss, but cannot predict when specific qualitative capabilities will emerge.
</details>

## Common mistakes

- Confusing Kaplan (2020) and Chinchilla (2022) — they give opposite guidance. Chinchilla is the current standard.
- Ignoring the irreducible entropy term $E$ — loss cannot go below ~1.69 nats no matter how much you scale
- Assuming emergent capabilities are unpredictable from first principles — they are predictable in aggregate (at scale), just not from loss alone
- Planning a training run without checking compute-optimal allocation — building a 100B model and training on 100B tokens is 10× under-trained by Chinchilla

## Final takeaway

Scaling laws give LLM training a principled framework. Loss decreases as a power law in both parameters and training tokens. The Chinchilla correction (2022) found the optimal is ~20 tokens per parameter — not the large-model-few-tokens approach GPT-3 used. Modern production LLMs (LLaMA 3, Mistral) intentionally over-train smaller models to get better inference efficiency at production scale. Understanding scaling laws is how you allocate a training compute budget without building and wasting costly experiments.

## References

- Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. OpenAI.
- Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models (Chinchilla). DeepMind / NeurIPS.
- Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models.
- Wei, J., et al. (2022). Emergent Abilities of Large Language Models. TMLR.
