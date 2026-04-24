---
id: peft-lora
title: "Parameter-efficient fine-tuning: LoRA, adapters, and prefix tuning"
sidebar_label: "91 · LoRA & PEFT"
sidebar_position: 91
slug: /theory/dnn/parameter-efficient-fine-tuning-lora
description: "How to fine-tune large language models with a fraction of the parameters: LoRA's low-rank decomposition, adapter layers, prefix tuning, and why PEFT methods are essential for LLMs."
tags: [lora, peft, adapters, prefix-tuning, fine-tuning, llm, transformers, deep-learning]
---

# Parameter-efficient fine-tuning: LoRA, adapters, and prefix tuning

Full fine-tuning a 7B-parameter LLM requires updating 7 billion parameters and storing 7B gradients — consuming ~80 GB of GPU memory for a single fine-tuning run. Parameter-efficient fine-tuning (PEFT) methods achieve comparable task performance while updating only 0.1–1% of parameters. LoRA is the most widely used PEFT method today: it is the standard approach for instruction tuning, domain adaptation, and task-specific customization of large language models.

## One-line definition

PEFT methods adapt a pre-trained model to a new task by adding a small number of trainable parameters while keeping the original weights frozen — LoRA adds low-rank matrices to weight projections, adapters insert small bottleneck layers, and prefix tuning prepends learnable tokens to the context.

![BERT BASE vs BERT LARGE — fine-tuning a model this size from scratch would require updating hundreds of millions of parameters; LoRA reduces this by 100–1000×](https://jalammar.github.io/images/bert-base-bert-large.png)
*Source: [Jay Alammar — The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)*

## Why this topic matters

LoRA is how most LLM fine-tuning in industry and research is done. It reduces GPU memory requirements by 3–10x compared to full fine-tuning, enables fine-tuning on consumer hardware (single 24 GB GPU), and produces results within 1–2% of full fine-tuning. Understanding LoRA is essential for any practical work with LLMs.

## The core problem: full fine-tuning at scale

For a 7B parameter LLM (e.g., LLaMA 2 7B):

| Stage | GPU memory |
|---|---|
| Model weights (bfloat16) | ~14 GB |
| Gradients | ~14 GB |
| Optimizer states (AdamW: 2 moments) | ~56 GB |
| Activations | Variable |
| **Total (full fine-tuning)** | **~100+ GB** |

This requires multiple high-end GPUs. LoRA reduces the trainable parameters (and thus gradient + optimizer memory) by keeping the base model frozen and adding tiny trainable matrices.

## LoRA: Low-Rank Adaptation

**Key insight**: the weight update $\Delta W$ that occurs during fine-tuning is intrinsically low-rank. Rather than storing the full $\Delta W \in \mathbb{R}^{d \times k}$ (large matrix), we decompose it as:

$$
\Delta W = BA
$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with $r \ll \min(d, k)$.

The modified forward pass:

$$
h = W_0 x + \frac{\alpha}{r} \Delta W x = W_0 x + \frac{\alpha}{r} B A x
$$

- $W_0$: frozen pre-trained weights
- $B, A$: trainable LoRA matrices
- $r$: rank (typically 4, 8, 16, 32)
- $\alpha$: scaling factor (often set to $r$ so $\alpha/r = 1$)

**Initialization**: $A$ is initialized from $\mathcal{N}(0, \sigma^2)$; $B$ is initialized to 0. So $\Delta W = BA = 0$ at the start — the model begins as the original pre-trained model and learns the adaptation.

### Parameter savings

For a linear layer $W \in \mathbb{R}^{d \times k}$ with $d = k = 4096$ and $r = 8$:

| | Full fine-tuning | LoRA |
|---|---|---|
| Trainable params | $4096 \times 4096 = 16.7M$ | $2 \times 4096 \times 8 = 65.5K$ |
| Reduction | — | **256× fewer** |

For LLaMA 2 7B with LoRA applied to all Q/K/V/O projections, $r=16$:
- Full fine-tuning: 7B trainable parameters
- LoRA: ~21M trainable parameters (~0.3%)

### Where to apply LoRA

LoRA is typically applied to the attention projection matrices and sometimes the FFN:

| Matrix | Apply LoRA? | Notes |
|---|---|---|
| $W^Q$ (query) | Yes | Standard |
| $W^K$ (key) | Yes | Standard |
| $W^V$ (value) | Yes | Standard |
| $W^O$ (output) | Yes | Standard |
| $W_1$ (FFN up) | Optional | More capacity |
| $W_2$ (FFN down) | Optional | More capacity |
| Embedding | No | Not typically |

## Adapter layers

Adapters (Houlsby et al., 2019) insert small bottleneck layers inside each transformer block:

```
Input → Pre-trained layer → Adapter(Down-project → Activation → Up-project) → Add → LayerNorm → Next layer
```

The adapter down-projects from $d_{\text{model}}$ to a small bottleneck dimension $m$ (typically 64 or 128), applies a nonlinearity, then up-projects back:

$$
\text{Adapter}(h) = h + W_{\text{up}} \cdot f(W_{\text{down}} h)
$$

- $W_{\text{down}} \in \mathbb{R}^{m \times d}$, $W_{\text{up}} \in \mathbb{R}^{d \times m}$
- Residual connection: if adapter contribution is small at init, the block is approximately identity

**Comparison with LoRA**: Adapters add inference overhead (two extra linear layers per block). LoRA has zero inference overhead because $\Delta W = BA$ can be merged into $W_0 + \Delta W$ after training.

## Prefix tuning

Prefix tuning (Li & Liang, 2021) prepends learnable "prefix" tokens to the key and value of every attention layer. These are continuous vectors (not real tokens from the vocabulary) that can encode task-specific information:

$$
K = [K_{\text{prefix}}; K_{\text{input}}], \quad V = [V_{\text{prefix}}; V_{\text{input}}]
$$

The model's self-attention now attends to both the original input and the learnable prefix. Only the prefix parameters are trained.

**Problem**: directly optimizing prefix vectors is unstable. In practice, a small MLP reparameterizes the prefix: $\text{Prefix} = \text{MLP}(P)$ where $P$ is the actual trainable parameter.

## Comparison of PEFT methods

| Method | Added params | Inference overhead | Merge into weights? | Best for |
|---|---|---|---|---|
| Full fine-tuning | 100% | None | — | Best performance, large GPU |
| LoRA | 0.1–1% | None (mergeable) | Yes | LLM fine-tuning standard |
| QLoRA | 0.1–1% | None | Yes | 4-bit quantized LLMs, low memory |
| Adapters | 0.5–5% | Yes (extra layers) | No | Multi-task serving |
| Prefix tuning | 0.1–1% | Yes (longer context) | No | Few training examples |
| Prompt tuning | < 0.01% | Yes (extra tokens) | No | Very small models or large models |

## QLoRA: LoRA on quantized models

QLoRA (Dettmers et al., 2023) enables fine-tuning 65B parameter models on a single 48 GB GPU by:
1. Loading the base model in 4-bit NormalFloat (NF4) quantization
2. Applying LoRA adapters in 16-bit (bfloat16)
3. Using paged optimizers to handle memory spikes

Memory for fine-tuning LLaMA 2 7B:

| Method | GPU memory | GPU count |
|---|---|---|
| Full fine-tuning (bfloat16) | ~100 GB | 4× A100 |
| LoRA (bfloat16) | ~24 GB | 1× A100 or 1× RTX 4090 |
| QLoRA (4-bit) | ~10 GB | 1× RTX 3080 |

## Python code: LoRA with HuggingFace PEFT

```python
# pip install transformers peft bitsandbytes accelerate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


# ============================================================
# Standard LoRA (bfloat16)
# ============================================================

model_name = "gpt2"   # Small model for demo; use "meta-llama/Llama-2-7b-hf" in practice
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                          # rank — controls capacity vs. efficiency
    lora_alpha=16,                # scaling: alpha/r applied to BA
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],   # GPT-2's attention projections
    bias="none",
)

# Apply LoRA to the model
lora_model = get_peft_model(base_model, lora_config)
lora_model.print_trainable_parameters()
# Example output: "trainable params: 294,912 || all params: 124,734,720 || trainable%: 0.2364"


# ============================================================
# Training loop (simplified)
# ============================================================
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

texts = [
    "Transformers are the backbone of modern NLP.",
    "LoRA reduces fine-tuning costs dramatically.",
    "Self-attention allows tokens to interact directly.",
]
encoded = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")

optimizer = AdamW(lora_model.parameters(), lr=3e-4)   # LoRA uses higher LR than full fine-tuning
total_steps = 5
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1, num_training_steps=total_steps)

lora_model.train()
for step in range(total_steps):
    input_ids = encoded["input_ids"]
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    outputs = lora_model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    print(f"Step {step+1}/{total_steps}: loss={loss.item():.4f}")


# ============================================================
# Merge LoRA back into base model (zero inference overhead)
# ============================================================
# After training, merge BA into W0 for deployment
merged_model = lora_model.merge_and_unload()
# merged_model is now a standard model with W0 + BA merged into each weight
print(f"\nMerged model type: {type(merged_model)}")


# ============================================================
# Manual LoRA implementation (to understand the math)
# ============================================================
class LoRALinear(nn.Module):
    """
    Linear layer with LoRA: y = (W0 + BA) x
    W0 is frozen; B and A are trainable.
    """

    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 16.0):
        super().__init__()
        import torch.nn as nn
        self.r = r
        self.scale = alpha / r

        # Frozen original weights
        self.W0 = nn.Linear(in_features, out_features, bias=False)
        for param in self.W0.parameters():
            param.requires_grad = False

        # Trainable LoRA matrices
        self.A = nn.Linear(in_features, r, bias=False)    # down-project
        self.B = nn.Linear(r, out_features, bias=False)   # up-project

        # Initialize: A ~ N(0, σ²), B = 0
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W0(x) + self.scale * self.B(self.A(x))


import torch.nn as nn
# Demo
lora_layer = LoRALinear(in_features=512, out_features=512, r=8, alpha=16)

trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in lora_layer.parameters() if not p.requires_grad)
print(f"\nLoRALinear: trainable={trainable:,}, frozen={frozen:,}")
# trainable = 8*512 + 512*8 = 8192  (tiny!)
# frozen    = 512*512 = 262144

x = torch.randn(4, 10, 512)   # (batch, seq, d)
out = lora_layer(x)
print(f"LoRALinear output: {out.shape}")   # (4, 10, 512)
```

## Rank selection guide

| Rank $r$ | Use case | Trainable params | Task performance |
|---|---|---|---|
| 2–4 | Memory-constrained, simple tasks | Very few | Lower bound |
| 8 | Default, most tasks | Standard | Good |
| 16 | Complex tasks, domain shift | Moderate | Better |
| 32–64 | Maximum capacity, near full fine-tuning | Significant | Near full FT |

**Rule of thumb**: start with $r=8$. If performance is insufficient, try $r=16$ or $r=32$. Going above $r=64$ rarely helps and approaches the cost of full fine-tuning.

## Interview questions

<details>
<summary>Why does LoRA work? Why is the weight update intrinsically low-rank?</summary>

Empirical evidence from Aghajanyan et al. (2020) shows that when fine-tuning a pre-trained model on a downstream task, the weight updates $\Delta W$ have low "intrinsic rank" — the task-specific adaptation can be captured in a low-dimensional subspace. Intuitively: the pre-trained model has already learned rich general representations. Fine-tuning on a specific task only needs to shift these representations slightly in a low-dimensional direction, not rewrite the entire weight matrix. The low-rank decomposition $\Delta W = BA$ exploits this structure, using only $2 \times d \times r$ parameters instead of $d^2$.
</details>

<details>
<summary>What is the difference between LoRA and adapters?</summary>

Both add small trainable modules while freezing the base model. LoRA decomposes the weight update as a low-rank product and can be merged into the original weights after training — zero inference overhead. Adapters insert extra feedforward layers in the transformer, which add computation at every forward pass. LoRA has become the dominant method because it can be merged, making it transparent to the inference pipeline. Adapters are preferred in multi-task settings where you want to swap task-specific modules at inference.
</details>

<details>
<summary>Why is the B matrix initialized to zero in LoRA?</summary>

$\Delta W = BA$. If $B$ is initialized to zero, then at the start of training $\Delta W = 0$, so the model outputs exactly the same as the frozen base model. This is an ideal starting point: the adaptation starts at zero and learns incrementally. If both $A$ and $B$ were initialized randomly, the initial adapter would perturb the pre-trained model's behavior before any training has happened, potentially degrading starting performance and making optimization harder.
</details>

## Common mistakes

- Using a learning rate too low for LoRA (2e-5) — LoRA benefits from higher LRs (3e-4) since it has far fewer parameters and can afford more aggressive updates
- Forgetting to merge the LoRA weights before deployment — running with separate B and A matrices adds overhead
- Applying LoRA only to Q and V but not K and O — including all attention projections usually improves results
- Not printing `model.print_trainable_parameters()` — easy way to verify the PEFT configuration is correct

## Final takeaway

LoRA is the industry standard for fine-tuning LLMs. It freezes all pre-trained weights and adds tiny low-rank matrices $B$ and $A$ to each target layer. The product $BA$ approximates the weight update with 10–1000× fewer parameters. After training, $BA$ can be merged into the original weights for zero inference overhead. QLoRA combines LoRA with 4-bit quantization, enabling fine-tuning 7B+ models on consumer GPUs. The combination of pre-training + LoRA fine-tuning is the standard workflow for adapting modern LLMs to custom applications.

## References

- Hu, E., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
- Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS.
- Houlsby, N., et al. (2019). Parameter-Efficient Transfer Learning for NLP (Adapters). ICML.
- Li, X., & Liang, P. (2021). Prefix-Tuning: Optimizing Continuous Prompts for Generation. ACL.
