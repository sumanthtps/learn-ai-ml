---
id: flashattention
title: "FlashAttention and efficient transformers"
sidebar_label: "92 · FlashAttention"
sidebar_position: 92
slug: /theory/dnn/flashattention-efficient-transformers
description: "Why standard attention is memory-bound, how FlashAttention reorders computation to avoid materializing the N×N matrix, and the landscape of efficient attention variants for long contexts."
tags: [flashattention, efficient-attention, long-context, transformers, deep-learning]
---

# FlashAttention and efficient transformers

> **TL;DR.** Standard attention is **memory-bound**, not compute-bound — the N×N attention matrix shuttles back and forth between fast SRAM and slow HBM, and that I/O cost dominates. **FlashAttention** computes the *same exact output*, but tiled: it loads small blocks of Q/K/V into SRAM, runs softmax + matmul without ever writing the N×N matrix to HBM, and uses a clever "online softmax" trick to merge results. Memory drops from O(N²) to O(N); wall-clock speed jumps 2–4×; long-context training becomes feasible. It's now the default in every production transformer (PyTorch's `F.scaled_dot_product_attention`, vLLM, all modern LLM training stacks).

Standard scaled dot-product attention materializes an $n \times n$ attention matrix in GPU memory, where $n$ is the sequence length. For $n=4096$: ~64 MB per head per batch item. For $n=32768$: ~4 GB. This memory cost makes long-context transformers prohibitively expensive. FlashAttention solves this by reordering the attention computation to never materialize the full attention matrix, reducing memory from $O(n^2)$ to $O(n)$ without changing the mathematical output.

## Try it interactively

- **[FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)** — the official Triton/CUDA implementation; one-liner replacement for nn.functional.scaled_dot_product_attention
- **[PyTorch SDPA docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)** — PyTorch 2+ automatically uses FlashAttention 2 when conditions allow
- **[Tri Dao — FlashAttention talk (YouTube)](https://www.youtube.com/results?search_query=tri+dao+flashattention)** — author's own explanation with diagrams
- **[Horace He — Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html)** — the canonical explanation of memory-bound vs compute-bound that motivates FlashAttention
- **[vLLM](https://github.com/vllm-project/vllm)** — production LLM serving framework built on FlashAttention + PagedAttention

## One-line definition

FlashAttention is an exact attention algorithm that computes the same result as standard attention but avoids materializing the full $N \times N$ attention matrix by using tiled computation with online softmax, reducing memory from $O(N^2)$ to $O(N)$ and significantly improving GPU throughput.

![Full self-attention matrix — every query attends to every key, producing an N×N matrix; FlashAttention computes the same result without ever storing this matrix in GPU HBM](https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)
*Source: [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)*

## Why this topic matters

FlashAttention is what enables modern LLMs to process long contexts (32k, 128k, 1M tokens). It is now the default attention implementation in PyTorch 2.0+ (`F.scaled_dot_product_attention`), HuggingFace Transformers, and every production LLM framework. Understanding FlashAttention explains why context length has exploded from 2048 tokens (GPT-3) to 128k+ (GPT-4, Claude 3) in just a few years.

## The bottleneck: GPU memory hierarchy

Modern GPUs have two types of memory:
- **HBM (High Bandwidth Memory)**: large (~40–80 GB), slow (~2 TB/s bandwidth)
- **SRAM (on-chip shared memory)**: tiny (~20 MB), very fast (~19 TB/s bandwidth)

Standard attention reads/writes matrices from HBM. The $n \times n$ attention matrix is too large for SRAM at any practical sequence length. Standard attention is therefore **memory-bound** — most time is spent waiting for data transfers to/from HBM, not computing.

```
Standard attention (n=4096, 1 head, float16):
1. Read Q, K from HBM → compute QK^T (n×n) → write to HBM    [expensive]
2. Read QK^T from HBM → apply mask → write to HBM             [expensive]
3. Read QK^T from HBM → compute softmax → write A to HBM      [expensive]
4. Read A, V from HBM → compute AV → write output to HBM      [expensive]
Memory: O(n²) reads + writes = 4096² = 16.7M floats per head
```

## FlashAttention: tiled computation with online softmax

FlashAttention splits Q, K, V into tiles and computes attention block by block entirely within SRAM. The key insight is that softmax can be computed incrementally using the **online softmax** trick.

### The online softmax trick

For a vector $x = [x_1, \ldots, x_n]$, standard softmax requires two passes: one to find $\max(x)$ and one to compute $\sum e^{x_i - \max(x)}$. But softmax can be computed in a single pass by maintaining running statistics:

$$
m_j = \max(m_{j-1}, x_j), \quad s_j = s_{j-1} e^{m_{j-1} - m_j} + e^{x_j - m_j}
$$

After processing all elements: $\text{softmax}(x_j) = e^{x_j - m_n} / s_n$.

FlashAttention uses this to process attention **block by block**: for each query block, iterate over all key-value blocks, update the running max and sum, and accumulate the weighted values — all in SRAM, never writing the full $n \times n$ score matrix to HBM.

```
FlashAttention (n=4096, 1 head):
For each Q_tile in Q:
    running_max = -inf, running_sum = 0, acc = 0
    For each KV_tile in (K, V):
        Load KV_tile from HBM to SRAM           [small block only]
        Compute scores = Q_tile @ KV_tile^T     [in SRAM]
        Update running max, sum (online softmax)
        acc += softmax_weights @ V_tile
    Write acc to HBM                            [one write per Q_tile]
Memory: O(n) — only tiles are in SRAM at any time
```

### Performance comparison

For sequence length $n$ on an A100 GPU:

| $n$ | Standard attention memory | FlashAttention memory | Speedup |
|---|---|---|---|
| 512 | 1 MB | 1 MB | 1.2× |
| 2048 | 16 MB | 4 MB | 2× |
| 4096 | 64 MB | 8 MB | 3–4× |
| 16384 | 1 GB | 32 MB | 6–8× |
| 65536 | 16 GB | 128 MB | OOM → feasible |

## FlashAttention-2 improvements

FlashAttention-2 (Dao, 2023) adds:
- Better parallelism across sequence dimension (not just batch and head)
- Reduced non-matrix-multiply operations
- ~2× speedup over FlashAttention on A100/H100

FlashAttention-3 (2024) targets Hopper (H100) architecture with specialized WGMMA instructions.

## Using FlashAttention in PyTorch

```python
import torch
import torch.nn.functional as F
import math

# ============================================================
# PyTorch 2.0+: F.scaled_dot_product_attention
# Uses FlashAttention automatically if available on the hardware
# ============================================================

batch, heads, seq_len, d_k = 4, 8, 1024, 64

Q = torch.randn(batch, heads, seq_len, d_k, device="cuda", dtype=torch.float16)
K = torch.randn(batch, heads, seq_len, d_k, device="cuda", dtype=torch.float16)
V = torch.randn(batch, heads, seq_len, d_k, device="cuda", dtype=torch.float16)

# PyTorch's fused attention (automatically selects FlashAttention if available)
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,          # FlashAttention
    enable_math=False,          # Disable standard math path
    enable_mem_efficient=False, # Disable xFormers memory-efficient path
):
    output_flash = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,   # Set True for decoder/causal attention
    )
print(f"FlashAttention output: {output_flash.shape}")   # (4, 8, 1024, 64)


# Causal attention for decoder (is_causal=True)
output_causal = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
print(f"Causal attention output: {output_causal.shape}")  # (4, 8, 1024, 64)


# ============================================================
# Standard attention (for comparison on CPU/no Flash support)
# ============================================================
def standard_attention(Q, K, V, causal=False):
    """Reference implementation — materializes full attention matrix."""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (B, H, N, N)
    if causal:
        mask = torch.triu(torch.ones(Q.size(-2), K.size(-2),
                                     device=Q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
    attn = scores.softmax(dim=-1)
    return attn @ V


# Both should give identical results (up to floating point differences)
Q_cpu = Q.float().cpu()
K_cpu = K.float().cpu()
V_cpu = V.float().cpu()

out_standard = standard_attention(Q_cpu, K_cpu, V_cpu)
out_flash_cpu = F.scaled_dot_product_attention(Q_cpu, K_cpu, V_cpu)

max_diff = (out_standard - out_flash_cpu).abs().max()
print(f"\nMax difference between standard and flash: {max_diff:.6f}")
# Should be very small (floating point precision difference only)


# ============================================================
# Memory comparison: standard vs. flash
# ============================================================
import time

def measure_memory(fn, *args, **kwargs):
    """Measure peak GPU memory during a function call."""
    if not torch.cuda.is_available():
        return None, fn(*args, **kwargs)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result = fn(*args, **kwargs)
    peak = torch.cuda.max_memory_allocated() / 1024**2   # MB
    return peak, result


# Use the PyTorch built-in which auto-selects the algorithm
for n in [512, 1024, 2048, 4096]:
    Q = torch.randn(1, 4, n, 64, device="cpu", dtype=torch.float32)
    K = torch.randn(1, 4, n, 64, device="cpu", dtype=torch.float32)
    V = torch.randn(1, 4, n, 64, device="cpu", dtype=torch.float32)
    # Show theoretical memory: n*n*4*4heads = attention matrix size
    attn_matrix_mb = (n * n * 4 * 4) / 1024**2   # float32, 4 heads
    print(f"n={n:5d}: attention matrix = {attn_matrix_mb:.1f} MB")
```

## Other efficient attention variants

### Sparse attention (Longformer, BigBird)

Instead of all $n^2$ pairs, attend only to a sparse pattern:
- **Local window**: each token attends to $w$ neighboring tokens: $O(n \cdot w)$
- **Global tokens**: a few special tokens attend to the entire sequence
- **Dilated strided**: every $k$-th token in a window

Used in Longformer (4096 tokens), BigBird (4096 tokens).

### Linear attention

Replace the $\text{softmax}(QK^T)$ computation with a kernel function that can be computed in $O(n)$:

$$
\text{Attn}(Q, K, V) \approx \phi(Q) \left(\phi(K)^T V\right)
$$

where $\phi$ is a feature map (e.g., $\phi(x) = \text{elu}(x) + 1$). The key trick: compute $K^T V$ first ($d \times d$ matrix), then multiply by $Q$ — $O(n d^2)$ instead of $O(n^2 d)$. Used in Performer, Linear Transformer.

### Sliding window attention (Mistral)

Each token attends only to the most recent $w$ tokens (e.g., $w=4096$). Combined with a large sliding window and group-query attention, Mistral 7B achieves competitive performance while processing longer sequences efficiently.

### Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

Reduce the number of key-value heads to decrease KV cache memory:
- **MQA**: all query heads share 1 K/V head
- **GQA**: $h$ query heads share $g$ K/V heads ($g < h$, e.g., 8 KV heads for 32 Q heads)

Used in LLaMA 3, Mistral, Gemma. Reduces KV cache by 4–8× without significant quality loss.

## The KV cache memory problem at scale

For a 70B model serving 1000 concurrent users with 32k context:

$$
\text{KV cache} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times \text{seq\_len} \times d_{\text{head}} \times \text{bytes}
$$

LLaMA 3 70B: $2 \times 80 \times 8 \times 32768 \times 128 \times 2\text{ bytes} \approx 8.5\text{ GB per user}$

For 1000 users: 8.5 TB — clearly infeasible. Techniques like paged attention (vLLM), prefix caching, and speculative decoding make large-scale LLM serving practical.

## Interview questions

<details>
<summary>Why is standard attention memory-bound rather than compute-bound?</summary>

Standard attention performs the following HBM reads/writes: read Q, K (compute QK^T) → write QK^T → read QK^T (softmax) → write A → read A, V (compute AV) → write output. The $n \times n$ attention matrix requires $O(n^2)$ HBM reads and writes. For $n=4096$: ~100 MB of HBM traffic per head per forward pass. The actual matrix multiplications are fast (SRAM operations), but the dominant cost is the slow HBM transfers. FlashAttention eliminates the intermediate $n \times n$ writes by keeping all intermediates in fast SRAM.
</details>

<details>
<summary>Does FlashAttention produce the exact same output as standard attention?</summary>

Yes — FlashAttention is mathematically exact (not an approximation). It computes the same result as $\text{softmax}(QK^T/\sqrt{d_k})V$ but via a tiled algorithm that never materializes the full $n \times n$ matrix. The online softmax maintains numerically exact running statistics, and the accumulated output equals the standard attention output up to floating-point precision. This is different from approximate attention methods (sparse attention, linear attention) which trade accuracy for efficiency.
</details>

<details>
<summary>What is grouped-query attention and why does it matter?</summary>

Multi-head attention uses $h$ query heads and $h$ key-value heads. The KV cache stores all $h$ K and V matrices per layer per step — $O(h \times n)$ memory. Grouped-query attention (GQA) reduces to $g < h$ KV heads (each shared by $h/g$ query heads). For LLaMA 3 70B: 64 query heads, 8 KV heads — 8× KV cache reduction. At inference, the KV cache is the main memory bottleneck (not model weights), so this reduction allows 8× more concurrent users or 8× longer context at the same memory cost.
</details>

## Common mistakes

- Not using `is_causal=True` when using `F.scaled_dot_product_attention` for a decoder — produces incorrect outputs without the causal mask
- Assuming FlashAttention is always faster for short sequences — for $n < 256$, standard attention may be faster (FlashAttention has tiling overhead)
- Forgetting that FlashAttention does not support custom attention biases easily — ALiBi-style biases require special handling in the tiled computation

## Final takeaway

FlashAttention eliminates the $O(n^2)$ memory bottleneck of standard attention by tiling the computation and using online softmax, keeping all intermediates in fast SRAM. The result is identical to standard attention but 3–8× faster and linear in memory. FlashAttention-2 is the default in all modern LLM frameworks and is what makes 128k+ context windows practical. Combined with GQA (fewer KV heads) and paged KV cache management, it enables efficient deployment of large transformers at scale.

## References

- Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS.
- Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.
- Ainslie, J., et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. EMNLP.
- Child, R., et al. (2019). Generating Long Sequences with Sparse Transformers (Sparse Attention). OpenAI.
