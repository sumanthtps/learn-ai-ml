---
id: in-context-learning
title: "In-context learning and prompting"
sidebar_label: "94 · In-Context Learning"
sidebar_position: 94
slug: /theory/dnn/in-context-learning-and-prompting
description: "How LLMs solve tasks from examples in the prompt without gradient updates — zero-shot, few-shot, chain-of-thought, and why in-context learning works mechanistically."
tags: [in-context-learning, prompting, few-shot, chain-of-thought, llm, deep-learning]
---

# In-context learning and prompting

GPT-3 introduced a surprising capability: by putting a few examples of a task in the prompt, the model could perform that task on new inputs — without any gradient updates. This ability, called in-context learning (ICL), is the foundation of modern LLM APIs. Understanding ICL, its mechanisms, and how to engineer effective prompts is essential for applied LLM work.

## One-line definition

In-context learning is the ability of a large language model to perform a new task by conditioning on a prompt containing task instructions and/or examples — without updating any model weights.

![BERT tasks — in-context learning generalizes this "add a task head" idea: instead of a learned head, the task description lives in the prompt itself, requiring zero additional parameters](https://jalammar.github.io/images/bert-tasks.png)
*Source: [Jay Alammar — The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)*

## Why this topic matters

ICL is the primary way LLMs are used in practice: via API calls with carefully crafted prompts. Understanding the mechanics — what kinds of prompts work, when few-shot beats zero-shot, when chain-of-thought is necessary — determines whether your LLM application works well or not. It also explains emergent model capabilities.

## Types of prompting

### Zero-shot prompting

No examples given — the model must perform the task from the task description alone:

```
Classify the sentiment of this review as positive or negative:
Review: "The movie was beautifully shot but the plot was predictable."
Sentiment:
```

Works well for tasks the model has seen many times during pre-training (sentiment, simple QA). Fails on unusual output formats or novel tasks.

### Few-shot prompting

Provide $k$ demonstration examples (input-output pairs) before the query:

```
Review: "I loved the acting."          → Sentiment: positive
Review: "The food was terrible."       → Sentiment: negative
Review: "Average experience overall."  → Sentiment: neutral
Review: "Truly exceptional film."      → Sentiment:
```

Few-shot prompting (GPT-3 paper) dramatically improves performance over zero-shot for most tasks. The model "reads" the pattern and applies it to the new input. No fine-tuning, no gradient steps.

### Chain-of-thought (CoT) prompting

For multi-step reasoning tasks (arithmetic, logic, commonsense), including intermediate reasoning steps in the demonstration dramatically improves accuracy:

```
Standard few-shot:
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each. How many does he have?
A: 11

Chain-of-thought:
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each. How many does he have?
A: Roger starts with 5 balls. 2 cans × 3 balls = 6 new balls. 5 + 6 = 11 balls.

Q: A store sells 15 apples per hour. How many in 8 hours minus 12 apples eaten?
A: [model must reason step by step]
```

Wei et al. (2022) showed CoT prompting unlocks reasoning in models with >~50B parameters. Zero-shot CoT works by simply appending "Let's think step by step."

### Instruction prompting (zero-shot with role)

Modern instruction-tuned models (ChatGPT, Claude) respond to natural language instructions in the system or user turn:

```
System: You are a helpful assistant specialized in Python code review.
User: Review the following function for bugs and suggest improvements:
[code]
```

This works because the model was fine-tuned on instruction-following data (RLHF, SFT) — see note 95.

## The prompt structure

A well-designed prompt has four optional components:

```
[System role]       → Who the model should be
[Task instruction]  → What to do
[Demonstrations]    → k examples of input → output (few-shot)
[Query]             → The actual input to process
[Output cue]        → The beginning of the expected output
```

Example for information extraction:

```
You are an expert information extractor.
Extract all named entities from the text and classify them as PERSON, ORG, or LOCATION.

Text: Apple Inc. hired Tim Cook as CEO.
Entities: PERSON: Tim Cook | ORG: Apple Inc.

Text: Barack Obama was born in Honolulu, Hawaii.
Entities: PERSON: Barack Obama | LOCATION: Honolulu, Hawaii

Text: Elon Musk founded SpaceX in Hawthorne.
Entities:
```

## Why in-context learning works: mechanistic view

ICL is not fine-tuning — no weights change. Several hypotheses explain why it works:

**1. Task location hypothesis**: the LLM stores many implicit task programs during pre-training. The prompt helps the model locate the right program in its weight space (Xie et al., 2021).

**2. In-context gradient descent**: Akyürek et al. (2022) showed that transformer attention can implement gradient descent steps — the in-context examples act as a mini-training set that updates the model's effective behavior via attention, not actual gradient descent.

**3. Pattern matching from pre-training**: the model has seen instruction-following patterns in web text. The prompt resembles those patterns and triggers the learned behavior.

In practice: ICL works best when the model is large enough (>1B params), the task is similar to pre-training distribution, and demonstrations are representative and correctly formatted.

## Demonstration selection and format

**Number of shots**: more shots → better, but diminishing returns. 4–8 shots is usually sufficient; 16+ rarely helps.

**Format consistency**: the output format of demonstrations must match the expected output format for the query. Even small inconsistencies (spacing, capitalization) hurt performance.

**Balanced demonstrations**: for classification, include roughly equal examples of each class. A skewed set biases the model toward over-represented classes.

**Example order**: the order of few-shot examples matters (Zhao et al., 2021). Examples at the end of the prompt have more influence than early ones. For robust results, average over multiple orderings.

**Label accuracy**: surprisingly, using wrong labels in demonstrations hurts performance less than having inconsistent format. The model picks up the pattern more than the specific label values.

## Advanced prompting techniques

### Self-consistency

Generate $k$ independent CoT chains (with temperature > 0), take the majority vote on the final answer. Significantly improves accuracy on arithmetic and logical reasoning:

```
Generate the answer 5 times with temperature=0.7, then:
- Answer 1: 42
- Answer 2: 42
- Answer 3: 44
- Answer 4: 42
- Answer 5: 43
→ Final answer: 42 (majority)
```

### Tree-of-thought

Instead of one CoT path, explore multiple reasoning paths in parallel and select the most promising (Yao et al., 2023). Useful for search-like problems where multiple approaches should be explored.

### Retrieval-augmented generation (RAG)

Retrieve relevant documents and include them in the prompt as context:

```
Context: [retrieved documents about the question]
Question: [user question]
Answer: [model generates answer grounded in context]
```

This is the standard way to give LLMs access to external knowledge without fine-tuning.

## Python code: prompting patterns

```python
# pip install openai transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name="gpt2"):
    """Load a causal LM for local prompting demos."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 50,
             temperature: float = 0.7, top_p: float = 0.9) -> str:
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Return only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ============================================================
# Zero-shot prompting
# ============================================================
def zero_shot_sentiment(model, tokenizer, review: str) -> str:
    prompt = (
        "Classify the sentiment of this review as positive or negative.\n"
        f"Review: {review}\n"
        "Sentiment:"
    )
    return generate(model, tokenizer, prompt, max_new_tokens=5).strip()


# ============================================================
# Few-shot prompting
# ============================================================
FEW_SHOT_EXAMPLES = [
    ("I loved this movie!", "positive"),
    ("Terrible experience, never again.", "negative"),
    ("It was okay, nothing special.", "neutral"),
]

def few_shot_sentiment(model, tokenizer, review: str,
                       examples=FEW_SHOT_EXAMPLES) -> str:
    examples_text = "\n".join(
        f"Review: {text}\nSentiment: {label}"
        for text, label in examples
    )
    prompt = (
        f"{examples_text}\n"
        f"Review: {review}\n"
        "Sentiment:"
    )
    return generate(model, tokenizer, prompt, max_new_tokens=5).strip()


# ============================================================
# Chain-of-thought prompting
# ============================================================
COT_EXAMPLES = """
Q: A bakery has 24 croissants. They sell 10 in the morning and bake 15 more. How many do they have?
A: The bakery starts with 24 croissants. They sell 10, leaving 24 - 10 = 14. They bake 15 more: 14 + 15 = 29.
Answer: 29

Q: A library has 50 books. 15 are checked out. 5 are returned and 3 more checked out. How many are checked out?
A: Initially 15 are checked out. 5 returned means 15 - 5 = 10 checked out. 3 more checked out: 10 + 3 = 13.
Answer: 13
"""

def chain_of_thought(model, tokenizer, question: str) -> str:
    prompt = (
        f"{COT_EXAMPLES}\n"
        f"Q: {question}\n"
        "A:"
    )
    return generate(model, tokenizer, prompt, max_new_tokens=80)


# ============================================================
# Self-consistency (majority vote over multiple generations)
# ============================================================
def self_consistency(model, tokenizer, question: str,
                     k: int = 5, temperature: float = 0.8) -> str:
    """Generate k answers and return the majority vote."""
    from collections import Counter

    answers = []
    for _ in range(k):
        full_response = chain_of_thought(model, tokenizer, question)
        # Extract the final number (simplified)
        import re
        numbers = re.findall(r'\d+', full_response.split("Answer:")[-1] if "Answer:" in full_response else full_response)
        if numbers:
            answers.append(numbers[-1])

    if not answers:
        return "No answer"
    majority = Counter(answers).most_common(1)[0][0]
    return f"{majority} (from {k} samples: {answers})"


# ============================================================
# RAG: retrieval-augmented generation (simplified)
# ============================================================
def rag_qa(model, tokenizer, question: str, context: str) -> str:
    """Answer a question given retrieved context."""
    prompt = (
        f"Context: {context}\n\n"
        f"Based on the context above, answer the following question.\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return generate(model, tokenizer, prompt, max_new_tokens=60).strip()


# Run demos
model, tokenizer = load_model("gpt2")

print("=== Zero-shot sentiment ===")
print(zero_shot_sentiment(model, tokenizer, "This was an amazing experience!"))

print("\n=== Few-shot sentiment ===")
print(few_shot_sentiment(model, tokenizer, "Absolutely fantastic, highly recommend!"))

print("\n=== Chain-of-thought ===")
print(chain_of_thought(model, tokenizer,
    "A store has 100 items. 30 are sold. 20 more arrive. How many are there?"))

print("\n=== RAG ===")
context = "The Eiffel Tower was built between 1887 and 1889. It was designed by Gustave Eiffel."
print(rag_qa(model, tokenizer, "When was the Eiffel Tower built?", context))
```

## Prompt engineering best practices

| Practice | Why it helps |
|---|---|
| Be explicit about output format | Reduces hallucinated formatting |
| Provide role context ("You are a...") | Activates task-relevant knowledge |
| Put the instruction at the end | Recency bias: model attends more to recent tokens |
| Use delimiters (`###`, `"""`) to separate sections | Reduces ambiguity about what is context vs. query |
| Use positive instructions ("Write X") not negative ("Don't write Y") | Models are better at following positive constraints |
| Ask for reasoning before the answer (CoT) | Improves accuracy on complex tasks |
| Test on edge cases explicitly in demonstrations | The model generalizes the demonstrated behavior |

## Limitations of in-context learning

| Limitation | Description |
|---|---|
| Context length | Can only fit a finite number of examples (limited by context window) |
| Inference cost | Each API call includes all demonstrations, paid per token |
| Sensitivity to format | Small changes in prompt wording can dramatically change output |
| Not true learning | ICL does not update weights — the model "forgets" after the session |
| Hallucination | The model generates fluently even when unsure — no uncertainty calibration by default |

## Interview questions

<details>
<summary>What is the difference between few-shot prompting and fine-tuning?</summary>

Few-shot prompting provides examples in the input context — no weights are updated, no gradient steps, and the examples are re-sent with every inference call (paying per-token cost). The model uses the examples for the current call only. Fine-tuning updates model weights on a training set — gradient steps adjust millions of parameters, the task is "baked in," and inference requires no examples in the prompt (reducing token cost and latency). Fine-tuning generally outperforms few-shot prompting for specialized tasks with sufficient training data; few-shot prompting is better for rapid prototyping or when training data is unavailable.
</details>

<details>
<summary>Why does chain-of-thought prompting improve performance on reasoning tasks?</summary>

Reasoning tasks require intermediate computation steps. A direct prediction from input to final answer requires the model to compress complex multi-step logic into a single forward pass from the last token of the prompt to the answer token. CoT externalizes the intermediate steps — the model generates them token by token. Each intermediate step is added to the context, giving the model more computation resources (more tokens) to arrive at the correct answer. Additionally, the pre-training corpus contains many step-by-step solutions; CoT prompts activate this pattern.
</details>

<details>
<summary>What makes in-context learning emergent — why does it fail in small models?</summary>

Small models lack sufficient capacity to store diverse task programs implicitly. For ICL to work, the model must: (1) understand the format of the demonstration, (2) identify what task is being demonstrated, (3) generalize the demonstrated pattern to the new input. These require large enough weights to store the task template and enough attention capacity to "read" the demonstration effectively. Empirically, meaningful ICL appears around 1B parameters for simple tasks and ~50B parameters for complex reasoning. Below these thresholds, the model ignores the demonstrations or copies their format without generalizing.
</details>

## Common mistakes

- Using inconsistent formats in demonstrations — the model picks up format patterns as strongly as semantic content
- Including demonstrations that are too similar to each other — the model needs to see the range of inputs the task covers
- Not using CoT for multi-step problems — a direct-answer prompt will fail on arithmetic, logic, and multi-hop reasoning
- Expecting ICL to replace fine-tuning for tasks requiring precise domain knowledge — for highly specialized tasks (medical, legal, code), fine-tuning on domain data usually outperforms prompting

## Final takeaway

In-context learning allows LLMs to perform new tasks by reading examples in the prompt — no gradient updates required. Zero-shot, few-shot, and chain-of-thought prompting are the three fundamental patterns. CoT dramatically improves multi-step reasoning by externalizing intermediate computation. ICL is powerful but limited by context length, token cost, and inference-time sensitivity. For production systems requiring consistent task behavior, supervised fine-tuning (or LoRA) on task-specific data is more reliable than prompting alone.

## References

- Brown, T., et al. (2020). Language Models are Few-Shot Learners (GPT-3). NeurIPS.
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS.
- Wang, X., et al. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models.
- Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with LLMs. NeurIPS.
