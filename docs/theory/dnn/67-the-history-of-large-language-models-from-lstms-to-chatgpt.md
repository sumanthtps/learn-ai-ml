---
id: llm-history
title: "The history of large language models from LSTMs to ChatGPT"
sidebar_label: "67 · LLM History"
sidebar_position: 67
slug: /theory/dnn/history-of-large-language-models-from-lstms-to-chatgpt
description: "The progression from n-gram models to LSTM language models to transformer-based LLMs — key papers, architectural shifts, and the timeline that led to ChatGPT."
tags: [llm, history, lstm, transformer, gpt, bert, chatgpt, deep-learning]
---

# The history of large language models from LSTMs to ChatGPT

Language modeling — predicting the next word given previous words — has been the central task driving NLP for decades. The models that emerged from this task became the foundation for every modern LLM. Understanding the progression from n-gram statistics to LSTM neural LMs to GPT-4 is not just historical context — it explains why each architectural choice was made and what problems it was solving.

## Timeline

```mermaid
timeline
    title Language Modeling History
    1990s : N-gram language models (statistical, no neural)
    2003 : Neural language model (Bengio et al.) — feedforward, fixed context
    2010 : Recurrent neural network LM (Mikolov et al.) — unlimited context
    2013 : Word2Vec — learned word embeddings as byproduct of LM training
    2014 : Seq2Seq (Sutskever et al.) — LSTM encoder-decoder for translation
    2015 : Attention for Seq2Seq (Bahdanau et al.) — the precursor to self-attention
    2017 : "Attention is All You Need" (Vaswani et al.) — Transformer architecture
    2018 : BERT (Google) — bidirectional pre-training; GPT-1 (OpenAI) — CLM pre-training
    2019 : GPT-2 (1.5B params) — emergent few-shot capabilities, initially withheld
    2020 : GPT-3 (175B) — in-context learning as a paradigm shift
    2022 : ChatGPT — GPT-3.5 + RLHF instruction tuning
    2023 : GPT-4, LLaMA, Mistral, Claude 2
    2024 : Claude 3, LLaMA 3, GPT-4o, Gemini 1.5
```

![The two-step process of modern LLM development: unsupervised pre-training on large corpora, then fine-tuning for specific tasks](https://jalammar.github.io/images/bert-transfer-learning.png)
*Source: [Jay Alammar — The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)*

## Era 1: Statistical language models (pre-2003)

N-gram models estimate $P(w_t \mid w_{t-1}, \ldots, w_{t-n+1})$ by counting co-occurrences in a corpus:

$$
P(w_t \mid w_{t-2}, w_{t-1}) \approx \frac{\text{count}(w_{t-2}, w_{t-1}, w_t)}{\text{count}(w_{t-2}, w_{t-1})}
$$

**Problems**:
- Exponential growth in vocabulary combinations (sparsity)
- Fixed, small context window ($n = 2, 3, 4$)
- No generalization: "The cat sat" and "The dog sat" are entirely different patterns

Smoothing techniques (Kneser-Ney, Good-Turing) mitigated sparsity but did not solve the fundamental limitation.

## Era 2: Neural language models and word embeddings (2003–2012)

### Bengio et al. (2003): A Neural Probabilistic Language Model

The first neural LM: embed words into a dense vector space, then use a feedforward network to predict the next word:

$$
P(w_t \mid w_{t-n+1}, \ldots, w_{t-1}) = f(e(w_{t-n+1}), \ldots, e(w_{t-1}))
$$

Key advance: words with similar meanings get similar embeddings through distributed representation. "The cat sat" and "The dog sat" generalize because "cat" and "dog" have similar embeddings.

**Limitation**: still fixed context window — just learned better, not longer.

### Mikolov et al. (2010): Recurrent Neural Network LM

Replace the feedforward with an RNN: unlimited context via the hidden state. The RNN LM became the standard for language modeling and was the first model to significantly reduce speech recognition perplexity.

### Word2Vec (Mikolov et al., 2013)

Not a language model per se — a training trick. Word2Vec trains a shallow network to predict context words (skip-gram) or the current word from context (CBOW). The learned embeddings became the standard input representation for all NLP models.

## Era 3: Sequence-to-sequence and attention (2013–2016)

### Seq2Seq (Sutskever et al., 2014)

LSTM encoder reads the entire source sentence into a single context vector; LSTM decoder generates the target. Achieved near state-of-the-art translation.

**Problem**: the encoder must compress arbitrarily long sentences into a single fixed-size vector. Long sentences lose information at the bottleneck.

### Attention (Bahdanau et al., 2015)

Instead of compressing the source into one vector, attention lets the decoder look at all encoder hidden states and compute a weighted average:

$$
c_t = \sum_s \alpha_{ts} h^{\text{enc}}_s, \quad \alpha_{ts} \propto \exp(e(h^{\text{dec}}_{t-1}, h^{\text{enc}}_s))
$$

This was the critical insight: give the model access to all past computations, not just the most recent hidden state. Attention improved machine translation dramatically and introduced the mechanism that would become the foundation of transformers.

## Era 4: The transformer (2017)

### "Attention is All You Need" (Vaswani et al., 2017)

Replaced the LSTM encoder-decoder with a fully attention-based architecture:
- **Self-attention**: each position attends to all other positions in the same sequence
- **Multi-head attention**: run attention in parallel in multiple subspaces
- **Positional encoding**: add position information since attention is order-agnostic
- **No recurrence**: all positions are processed in parallel — training is dramatically faster

Results: new SOTA on WMT translation. More importantly, the architecture scaled with compute in a way LSTMs did not.

**Why transformers replaced LSTMs:**

| Property | LSTM | Transformer |
|---|---|---|
| Parallel training | No (sequential by step) | Yes (all positions simultaneously) |
| Long-range dependencies | Degraded by gradient | Direct via attention |
| GPU utilization | Poor (sequential) | Excellent (matrix multiply) |
| Scaling behavior | Sublinear with size | Near-linear with size |
| Context limit | Unlimited (theoretically) | Context window (but very long) |

## Era 5: Pre-training revolution (2018–2019)

### BERT (Devlin et al., 2018)

**Key idea**: pre-train a bidirectional transformer encoder on masked language modeling (predict masked tokens) + next sentence prediction. Then fine-tune on downstream tasks with a task-specific head.

BERT-large: 340M parameters. Achieved state-of-the-art on 11 NLP benchmarks simultaneously. Demonstrated that pre-training on unlabeled text and fine-tuning is a universal recipe.

### GPT-1 (Radford et al., 2018)

Concurrent with BERT but uses a decoder-only (causal) transformer pre-trained on CLM (predict next token). Fine-tuned on downstream tasks. Established that the CLM objective alone, at sufficient scale, produces powerful representations.

### GPT-2 (Radford et al., 2019)

1.5B parameters on WebText (8M Reddit documents). Two surprises:
1. Coherent paragraphs with almost no fine-tuning — the model learned to follow prompts
2. OpenAI initially withheld it from release, citing potential misuse — the first public AI safety controversy

GPT-2 demonstrated **in-context learning**: the model could "follow a task description" from the prompt without gradient updates.

## Era 6: The scaling era (2020–2022)

### GPT-3 (Brown et al., 2020)

175B parameters. The breakthrough: **few-shot in-context learning** as a paradigm:

```
Input: "Translate English to French:
        sea otter → loutre de mer
        cheese → fromage
        horse → "
Output: "cheval"
```

No gradient updates. The model reads examples from the prompt and generalizes to new inputs. This showed that scale alone produced capabilities that no one had explicitly trained for.

### InstructGPT / ChatGPT (2022)

GPT-3 was powerful but often unhelpful, biased, or harmful. OpenAI applied RLHF (see note 95):
1. Supervised fine-tuning on human-written instruction-response pairs
2. Train a reward model on human preference rankings
3. PPO to optimize toward the reward model

The result: ChatGPT. The same model architecture as GPT-3 but dramatically better at following instructions, refusing harmful requests, and maintaining coherent conversations. Released November 30, 2022. Reached 1 million users in 5 days.

## Era 7: Open models and efficiency (2023–present)

### LLaMA (Touvron et al., 2023)

Meta released LLaMA-1 (7B–65B) as research weights. Key insight: train smaller models for many more tokens (see Chinchilla). LLaMA-2 and LLaMA-3 followed with permissive licenses. LLaMA-3 70B competes with GPT-3.5.

### Mistral 7B (2023)

7B parameter model outperforming LLaMA 13B. Introduced sliding window attention and grouped-query attention for efficient inference. Showed that architecture improvements matter as much as scale.

### Claude 2/3 (Anthropic, 2023–2024)

Trained with Constitutional AI (RLAIF) — generates feedback from a model using a written constitution rather than requiring human annotations for every preference. Claude 3 Opus achieves GPT-4 level performance.

## Key inflection points

| Year | Event | Why it mattered |
|---|---|---|
| 2003 | Neural LM | Words as dense vectors; generalization across similar words |
| 2014 | Seq2Seq | End-to-end sequence transformation — no feature engineering |
| 2015 | Attention | Direct access to any part of the context — the core of transformers |
| 2017 | Transformer | Parallel training at scale; replaced all RNN-based models |
| 2018 | BERT/GPT | Pre-train on unlabeled text → fine-tune — universal recipe |
| 2020 | GPT-3 | Scale produces emergent capabilities; in-context learning |
| 2022 | ChatGPT | RLHF turns a language model into an assistant |

## Interview questions

<details>
<summary>What was the key limitation of LSTM seq2seq models that attention solved?</summary>

Seq2Seq with LSTM requires compressing the entire source sentence into a single fixed-size context vector — the encoder's final hidden state. This is a severe information bottleneck for long sentences: the model cannot remember all relevant details when the context vector is only 256–1024 dimensions. Bahdanau attention solved this by allowing the decoder to directly query all encoder hidden states at every decoding step. Instead of a single fixed context, the decoder computes a weighted sum of all encoder states, with weights learned based on how relevant each source position is to the current decoding step. This eliminated the bottleneck and enabled handling much longer sequences.
</details>

<details>
<summary>Why did transformers replace LSTMs rather than simply making LSTMs larger?</summary>

LSTMs have two fundamental limits at scale: (1) Sequential computation — each time step depends on the previous, so you cannot parallelize training across the sequence dimension. A 1000-token LSTM requires 1000 sequential operations; a transformer processes all 1000 tokens in parallel. This makes LSTMs far slower on modern GPUs which excel at parallel matrix multiplication. (2) Gradient path — long-range dependencies require gradients to flow through many time steps, which degrades even with LSTM's memory mechanisms. Transformers have a constant-length gradient path between any two positions (one attention step). These two advantages — parallelism and O(1) gradient path — made transformers scale better with data and compute.
</details>

## Final takeaway

Language modeling progressed from statistical n-gram counting → neural representations (embeddings) → LSTM recurrence for unlimited context → attention for direct cross-position access → transformers for parallel training at scale → pre-training + fine-tuning as a universal paradigm → RLHF for instruction alignment. Each step solved a concrete limitation of the previous approach. ChatGPT is the combination of a transformer at scale (GPT-3) with RLHF alignment — technically not novel but enormously impactful in demonstrating that alignment turns a raw LM into a usable product.
