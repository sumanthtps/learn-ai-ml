---
id: gpt-decoder-only
title: "GPT: decoder-only causal language modeling"
sidebar_label: "88 · GPT"
sidebar_position: 88
slug: /theory/dnn/gpt-decoder-only-causal-lm
description: "GPT's decoder-only architecture, causal language modeling objective, how prompting works without fine-tuning, and the architectural choices shared by LLaMA, Mistral, and modern LLMs."
tags: [gpt, decoder-only, causal-lm, llm, transformers, deep-learning]
---

# GPT: decoder-only causal language modeling

> **TL;DR.** GPT is the simplest possible transformer: a stack of decoder blocks (causal self-attention + FFN, no cross-attention) trained on one objective — *predict the next token*. The causal mask means every position is both an input and a training target, so a sequence of T tokens gives T-1 gradient signals. Scale this up enough (GPT-3, 175B params) and the model starts solving tasks just by reading examples in the prompt — no fine-tuning needed. Every modern LLM (LLaMA, Mistral, Claude, Gemini) is a polished version of this pattern.

GPT (Generative Pre-trained Transformer) demonstrated that a single pre-trained language model — trained purely to predict the next token — could be fine-tuned for almost any NLP task. GPT-3 (2020) then showed the model could perform tasks without any fine-tuning, just by reading a few examples in the prompt. This was the foundation for the modern LLM era.

## Try it interactively

- **[OpenAI Playground](https://platform.openai.com/playground)** — interact with GPT-3.5/4 directly: tweak temperature, top-p, max tokens
- **[Hugging Face GPT-2 demo](https://huggingface.co/openai-community/gpt2)** — original 2019 GPT-2 in your browser, free
- **[nanoGPT (Karpathy)](https://github.com/karpathy/nanoGPT)** — train a tiny GPT on Shakespeare in ~30 minutes on a laptop
- **[Karpathy — Let's build GPT (YouTube)](https://www.youtube.com/watch?v=kCc8FmEb1nY)** — 2-hour live coding of a working GPT from scratch
- **[bbycroft LLM Visualization](https://bbycroft.net/llm)** — animated 3D walkthrough of every part of a working transformer
- **[Replicate base-model LLaMA](https://replicate.com/explore)** — try a non-RLHF'd model to feel the raw "next-token predictor" behavior

## A real-world analogy

GPT is like an **autocomplete engine that has read the entire internet**. At every cursor position, it asks: "given everything to the left, what's the most likely next token?" That's it — that's the whole game. Train it on enough text and "the most likely next token" turns out to encode grammar, world knowledge, reasoning patterns, and even instruction-following (because the training data contained instructions and their answers). Few-shot prompting is just providing the autocomplete engine with a few examples so it knows what *kind* of completion you want.

## One-line definition

GPT is a stack of transformer decoder blocks trained with the causal language modeling objective (predict the next token) on a large text corpus, producing a model that generates text, solves tasks via prompting, and fine-tunes to diverse tasks with a simple linear head.

![Autoregressive decoding — the model generates one token at a time, each time attending only to previously generated tokens via masked self-attention](https://jalammar.github.io/images/t/transformer_decoding_2.gif)
*Source: [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)*

## Why this topic matters

Decoder-only transformers are the dominant architecture in modern AI: GPT-4, LLaMA, Mistral, Claude, Gemini, and Gemma are all decoder-only. Understanding GPT's architecture, training objective, and prompting behavior is the foundation for working with any modern large language model.

## Architecture

GPT uses transformer decoder blocks with **only causal self-attention** — no cross-attention (no encoder to attend to). The "decoder" label is slightly misleading since GPT is not a decoder in the seq2seq sense; it is more accurately called a **causal language model**.

Each GPT block:
$$
H' = H + \text{MHA}(\text{LayerNorm}(H), \text{causal mask})
$$
$$
H_{\text{out}} = H' + \text{FFN}(\text{LayerNorm}(H'))
$$

Two sublayers: causal self-attention + FFN (no cross-attention — one fewer sublayer than the encoder-decoder decoder).

| Model | $d_{\text{model}}$ | Layers | Heads | Parameters | Vocab |
|---|---|---|---|---|---|
| GPT (2018) | 768 | 12 | 12 | 117M | 40k |
| GPT-2 (2019) | 1600 | 48 | 25 | 1.5B | 50,257 |
| GPT-3 (2020) | 12,288 | 96 | 96 | 175B | 50,257 |
| LLaMA 2 7B | 4,096 | 32 | 32 | 7B | 32,000 |
| LLaMA 3 70B | 8,192 | 80 | 64 | 70B | 128,256 |

## Training objective: causal language modeling

On a sequence of tokens $x_1, x_2, \ldots, x_T$:

$$
\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$

The model predicts token $t$ from only $x_1, \ldots, x_{t-1}$ (enforced by the causal mask). Every position in the sequence produces a prediction and contributes to the loss — very data-efficient.

Training on the entire internet worth of text, the model learns:
- Grammar and syntax
- World knowledge (factual associations)
- Reasoning patterns
- Code logic
- Instruction following (implicitly, from instructions in the pre-training corpus)

## From pre-training to task solving: prompting

GPT showed that a language model trained only to predict the next token can solve NLP tasks without task-specific fine-tuning. The trick: **prompt the model** to continue a few-shot template.

**Zero-shot**: describe the task
```
Classify sentiment (positive/negative):
"The movie was amazing."
Sentiment:
```

**Few-shot**: show examples before the query (in-context learning)
```
Text: "I hated the food."    → Sentiment: negative
Text: "The service was great." → Sentiment: positive
Text: "The room was noisy."   → Sentiment:
```

GPT-3 with 175B parameters could perform competitively on many NLP benchmarks with zero-shot or few-shot prompting — no gradient updates needed.

## Key architectural improvements in modern LLMs

Modern LLMs (LLaMA, Mistral, Gemma) differ from GPT-2/GPT-3 in several architectural choices:

| Component | Original GPT | Modern LLMs (LLaMA-style) |
|---|---|---|
| Positional encoding | Learned absolute | RoPE (rotary position embedding) |
| Normalization | Post-norm | Pre-norm with RMSNorm |
| Activation | GELU | SwiGLU (Swish-Gated Linear Unit) |
| Attention | Standard MHA | Grouped-Query Attention (GQA) |
| FFN structure | 2-layer MLP | 3-matrix SwiGLU FFN |
| Context length | 512–2048 tokens | 4096–128k+ tokens |

### RMSNorm

$$
\text{RMSNorm}(x)_j = \frac{x_j}{\sqrt{\frac{1}{d}\sum_k x_k^2 + \epsilon}} \cdot \gamma_j
$$

Removes mean subtraction — simpler, faster, same performance.

### SwiGLU activation (FFN)

$$
\text{SwiGLU}(x) = \text{Swish}(W_1 x) \odot (W_3 x), \quad \text{then} \times W_2
$$

Uses three weight matrices instead of two. More expressive than GELU at the same compute.

### Grouped-Query Attention (GQA)

Instead of $h$ query heads each with their own key-value heads, GQA uses $h$ query heads but only $g < h$ key-value heads (each shared by multiple query heads). For LLaMA-3 70B: 64 query heads, 8 KV heads. Reduces memory during inference (smaller KV cache).

## Python code: GPT-style decoder-only model

```python
import torch
import torch.nn as nn
import math


class GPTConfig:
    vocab_size: int = 50257
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    max_len: int = 1024
    dropout: float = 0.1
    dim_feedforward: int = 3072   # 4 × d_model


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention with KV cache support."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)  # fused Q,K,V projection
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # Fused QKV projection
        qkv = self.qkv(x)   # (B, T, 3*d_model)
        Q, K, V = qkv.split(self.d_model, dim=-1)

        # Reshape to (B, heads, T, d_k)
        def to_heads(t):
            return t.reshape(B, T, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = to_heads(Q), to_heads(K), to_heads(V)

        # Causal scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)   # (B, h, T, T)

        # Create causal mask (upper triangle = -inf)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = scores.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ V   # (B, h, T, d_k)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.resid_dropout(self.out(out))


class GPTBlock(nn.Module):
    """One GPT decoder block: causal self-attention + FFN (pre-norm)."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int,
                 dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """Minimal GPT: token embedding + positional embedding + N blocks + LM head."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            GPTBlock(cfg.d_model, cfg.num_heads, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying: share embedding and unembedding weights (Devlin et al., 2018)
        self.lm_head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (batch, seq_len) token IDs
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_p: float = 0.9) -> torch.Tensor:
        """Autoregressive generation with top-p sampling."""
        import torch.nn.functional as F
        self.eval()
        x = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Trim to max context
            x_cond = x[:, -self.pos_emb.num_embeddings:]
            logits = self(x_cond)[:, -1, :]   # last position logits

            # Temperature
            logits = logits / temperature

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                remove = cum_probs - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits[sorted_idx] = sorted_logits

            probs = logits.softmax(dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_id], dim=1)

        return x


# ============================================================
# Demo
# ============================================================
cfg = GPTConfig()
cfg.d_model = 256
cfg.num_heads = 8
cfg.num_layers = 4
cfg.dim_feedforward = 1024
cfg.max_len = 128
cfg.vocab_size = 1000

model = GPT(cfg)
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")

# Forward pass
token_ids = torch.randint(0, cfg.vocab_size, (2, 20))
logits = model(token_ids)
print(f"Input:  {token_ids.shape}")   # (2, 20)
print(f"Output: {logits.shape}")      # (2, 20, 1000)

# Training loss
targets = token_ids[:, 1:]      # shift: predict token t+1
logits_shifted = logits[:, :-1, :]   # predictions for positions 0..T-2
import torch.nn.functional as F
loss = F.cross_entropy(logits_shifted.reshape(-1, cfg.vocab_size), targets.reshape(-1))
print(f"CLM loss: {loss.item():.4f}")
print(f"Perplexity: {loss.exp().item():.1f}")


# ============================================================
# Using HuggingFace GPT-2
# ============================================================
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2.eval()

prompt = "The transformer architecture"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

with torch.no_grad():
    # Generate 30 new tokens
    output_ids = gpt2.generate(
        input_ids,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"\nPrompt:    {prompt}")
print(f"Generated: {generated}")
```

### Try it yourself: experiments

| Question | Try this |
|----------|----------|
| Temperature 0 vs 1.5 | Same prompt, both temperatures, observe coherence vs creativity tradeoff |
| Few-shot prompting | Give 3 input/output examples, then a 4th input — see if the model continues the pattern |
| Effect of weight tying | Set `lm_head.weight = nn.Parameter(...)` (separate weights) — model uses ~30% more params |
| Profile generation speed | Time `model.generate(..., max_new_tokens=200)` with `use_cache=True` vs `False` — KV cache wins |
| Look inside the attention | Hook into a block's attention to extract the attention matrix; visualize the lower-triangular structure |
| Inject a system prompt | Prepend `"You are a helpful assistant. "` to your input — observe behavior shift (only works on RLHF'd models) |

## CLM training: the shifting trick

The most important implementation detail in CLM training: **shift the targets by 1**.

```
Input tokens:   [The, cat, sat, on, the, mat]
                  ↓    ↓    ↓   ↓    ↓    ↓
Target tokens:  [cat, sat, on, the, mat, EOS]
```

In code: `input = token_ids[:, :-1]`, `target = token_ids[:, 1:]`. The model sees the current token, predicts the next.

## GPT vs. BERT: which to use?

| Task | BERT | GPT |
|---|---|---|
| Text classification | ✓ Best | ✓ Works (with prompting) |
| Named entity recognition | ✓ Best | ✗ Not natural |
| Open-ended generation | ✗ Not causal | ✓ Native |
| Few-shot prompting | ✗ Poor | ✓ Native |
| Semantic search | ✓ Good embeddings | ✓ With embedding extraction |
| Code generation | ✗ | ✓ |
| Reasoning chains | ✗ | ✓ Chain-of-thought prompting |

## Cross-references

- **Prerequisite:** [81 — Masked Self-Attention](./81-masked-self-attention-in-the-transformer-decoder.md) — the causal mask GPT depends on
- **Prerequisite:** [83 — Decoder Architecture](./83-transformer-decoder-architecture.md) — the 2-sublayer decoder block (no cross-attention)
- **Prerequisite:** [84 — Inference](./84-transformer-inference-step-by-step.md) — autoregressive generation + KV caching
- **Prerequisite:** [85 — Training Objectives](./85-transformer-training-objectives.md) — CLM in detail
- **Related:** [87 — BERT (Encoder-Only)](./87-bert-encoder-pretraining.md) — the contrasting "understand" model
- **Follow-up:** [90 — Fine-Tuning](./90-fine-tuning-transformers.md) — adapting pretrained GPTs to specific tasks (LoRA, instruction-tuning)

## Interview questions

<details>
<summary>How does GPT solve NLP tasks without task-specific fine-tuning?</summary>

GPT uses in-context learning: the prompt contains a task description and (for few-shot) examples of input-output pairs. The model conditions its generation on the prompt and produces an appropriate continuation. Because the pre-training corpus contained instructions, examples, and demonstrations of many tasks, the model has seen this pattern and can generalize. It does not update any weights — it reads the task specification and "solves" it by predicting the most likely continuation.
</details>

<details>
<summary>Why does GPT use weight tying between the embedding and the output (LM head) projection?</summary>

The token embedding $W_{\text{emb}} \in \mathbb{R}^{|\text{vocab}| \times d}$ maps token IDs to vectors. The LM head $W_{\text{out}} \in \mathbb{R}^{d \times |\text{vocab}|}$ maps the final hidden state back to logits over the vocabulary. Tying them ($W_{\text{out}} = W_{\text{emb}}^T$) reduces parameters by $|\text{vocab}| \times d$ (23M for GPT-2) and improves training — the embedding and output projection are used in complementary ways, so sharing weights regularizes both.
</details>

<details>
<summary>What is the key architectural difference between GPT and BERT?</summary>

Two differences: (1) mask — GPT uses a causal (lower-triangular) mask so each token can only see past tokens; BERT uses no mask (bidirectional). (2) pre-training objective — GPT is trained to predict the next token (CLM), giving it generation capability; BERT is trained to predict masked tokens (MLM), giving it bidirectional contextual representations. GPT has no cross-attention; BERT has no causal masking. Both are stacks of transformer blocks, but optimized for different capabilities.
</details>

## Common mistakes

- Applying the causal mask incorrectly — `diagonal=1` means token 0 can attend to itself; `diagonal=0` would mask the current position
- Forgetting to shift targets by 1 — common bug that causes loss to drop to 0 immediately (the model is predicting the current token it already sees)
- Not using weight tying between embedding and LM head — wastes parameters and reduces performance
- Sampling with temperature=0 (which collapses to greedy) for diverse generation tasks

## Final takeaway

GPT is a causal language model: a stack of decoder blocks trained to predict the next token. It has no encoder, no cross-attention — just causal self-attention and FFN, repeated $N$ times. The CLM objective makes every token position a training signal, enabling efficient scaling to 175B+ parameters. GPT-3 demonstrated that scale alone — without task-specific training — can produce models that solve NLP tasks via in-context learning. Modern LLMs (LLaMA, Mistral, Claude) are all refined versions of this same decoder-only architecture.

## References

- Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training (GPT).
- Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners (GPT-2).
- Brown, T., et al. (2020). Language Models are Few-Shot Learners (GPT-3). NeurIPS.
- Touvron, H., et al. (2023). LLaMA 2: Open Foundation and Fine-Tuned Chat Models.
