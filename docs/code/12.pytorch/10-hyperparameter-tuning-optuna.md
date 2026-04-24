---
id: 10-hyperparameter-tuning-optuna
title: "Video 10: Hyperparameter Tuning with Optuna"
sidebar_label: "10 · Hyperparameter Tuning (Optuna)"
sidebar_position: 10
description: Automated hyperparameter optimization for PyTorch models using Optuna — TPE, pruning, and visualization.
tags: [pytorch, optuna, hyperparameter-tuning, optimization, bayesian, campusx]
---

# Hyperparameter Tuning the ANN using Optuna
**📺 CampusX — Practical Deep Learning using PyTorch | Video 10**

> **What you'll learn:** How to stop guessing hyperparameters and automate the search using Optuna — a powerful Bayesian optimization framework.

---

## 1. The Hyperparameter Problem

By now (videos 7–9), you've tuned hyperparameters manually: tried different learning rates, adjusted dropout, changed model size. This manual process works for 2–3 parameters, but becomes unmanageable when you have 7–10 hyperparameters simultaneously. The combinations grow exponentially.

A neural network has many hyperparameters — none of them are learned during training:

| Category | Hyperparameters |
|---|---|
| Architecture | num_layers, hidden_dim, activation function |
| Regularization | dropout rate, weight_decay |
| Optimizer | optimizer type, learning rate, momentum, betas |
| Training | batch_size, num_epochs |
| Data | augmentation strength, normalization |

**Grid Search** (try all combinations): with 5 values each for 6 hyperparameters = 5⁶ = 15,625 experiments. Completely impractical.

**Random Search**: randomly sample combinations. Better than grid search because it doesn't waste trials on redundant configurations. But it ignores what it learned from previous trials — every trial is still random.

**Bayesian Optimization (Optuna)**: learns from past trials. After each trial, it updates a model of "which hyperparameter regions tend to give good results" and samples the next trial from that promising region. It's like a smart researcher who reads their own experiment notes before designing the next experiment.

The key insight: Optuna doesn't need to try 15,000 combinations. It finds nearly the same quality configuration in 50–100 trials by focusing on the most promising regions of the search space.

## Visual Reference

![Optuna contour plot — learning rate vs optimizer interaction](https://miro.medium.com/1*S2qQHG29t2r_nlUshciFTA.png)

*An Optuna contour plot showing how two hyperparameters (e.g. learning rate and optimizer type) jointly affect validation accuracy. Darker regions are better. Optuna's TPE sampler focuses new trials on the high-performing region rather than sampling uniformly — this is why it needs far fewer trials than grid or random search to find a good configuration.*

---

## 2. Installing Optuna

```bash
pip install optuna optuna-integration[pytorch]
pip install plotly   # For visualizations
```

---

## 3. Optuna Core Concepts

Before the table, here is the beginner-friendly picture:

When we train a model manually, we usually pick hyperparameters by hand:

- learning rate
- batch size
- hidden size
- dropout

Then we run training, look at validation performance, and try again.

Optuna automates that trial-and-error loop.

So instead of saying:

> "I will manually try 20 combinations."

we say:

> "Optuna, keep proposing combinations and tell me which one works best."

The terms below are just names for the pieces of that workflow.

| Concept | Meaning |
|---|---|
| **Study** | The optimization experiment (maximize val_acc or minimize val_loss) |
| **Trial** | One complete training run with one set of hyperparameters |
| **Objective function** | The function that trains the model and returns the metric to optimize |
| **Sampler** | Algorithm that suggests next hyperparameters (default: TPE) |
| **Pruner** | Stops bad trials early (saves compute) |

---

## 4. Basic Optuna Example

The core idea of Optuna is: wrap your entire training function in a function called `objective(trial)`, replace hardcoded hyperparameter values with `trial.suggest_*()` calls, and return the metric you care about. Optuna calls this function over and over, each time with smarter hyperparameter choices.

### What is the `objective(trial)` function?

This is the one concept that tends to feel abstract at first.

The `objective` function is simply:

- one complete train-and-evaluate run
- for one particular set of hyperparameters

So if Optuna asks for 50 trials, it will call `objective(trial)` 50 times. Each call gets a different set of suggested hyperparameters.

Inside that function, your job is always the same:

1. ask Optuna for hyperparameter values
2. build the model and training setup using those values
3. train
4. return one score

That returned score is how Optuna decides whether this trial was good or bad.

```python
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np

# ── Prepare Data ONCE (outside the objective function!) ─────
# Data preparation happens once, before any trials start.
# All trials share the same train/val split — otherwise we'd be
# comparing results on different data, which is unfair.
data = load_breast_cancer()
X    = StandardScaler().fit_transform(data.data)   # Normalize features to zero mean
y    = data.target                                  # Binary labels (0 or 1)

X_t  = torch.tensor(X, dtype=torch.float32)
y_t  = torch.tensor(y, dtype=torch.long)

dataset  = TensorDataset(X_t, y_t)
n_train  = int(0.8 * len(dataset))
# manual_seed(42) ensures the same split across all runs — reproducibility
train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train],
                                generator=torch.Generator().manual_seed(42))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── The Objective Function ───────────────────────────────────
# Optuna calls this function once per trial. Each call:
# 1. Samples a new set of hyperparameters via trial.suggest_*()
# 2. Builds a model with those hyperparameters
# 3. Trains it
# 4. Returns the metric to optimize
def objective(trial):
    # ① Sample hyperparameters — these replace the values you'd normally hardcode
    # trial.suggest_int: samples an integer between low and high (inclusive)
    n_layers    = trial.suggest_int("n_layers", 1, 4)

    # trial.suggest_categorical: picks one value from the list
    # Use this for discrete, non-numeric choices
    hidden_dim  = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])

    # trial.suggest_float: samples a float in [low, high]
    dropout     = trial.suggest_float("dropout", 0.1, 0.5)

    # log=True: samples on log scale — better for LR where 1e-4 and 1e-3
    # are very different, but 0.05 and 0.06 are nearly the same
    lr           = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    batch_size     = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    # ② Build DataLoaders — batch_size is now a hyperparameter too
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

    # ③ Build model dynamically based on sampled architecture
    # First layer: input → hidden_dim
    layers = [nn.Linear(30, hidden_dim),  # 30 features in breast cancer dataset
              nn.BatchNorm1d(hidden_dim),  # Normalize activations
              nn.ReLU(),
              nn.Dropout(dropout)]
    # Additional hidden layers (n_layers - 1 more, since first is already added)
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim),
                   nn.BatchNorm1d(hidden_dim),
                   nn.ReLU(),
                   nn.Dropout(dropout)]
    # Output layer: hidden_dim → 2 classes (malignant or benign)
    layers.append(nn.Linear(hidden_dim, 2))
    model = nn.Sequential(*layers).to(DEVICE)

    # ④ Build optimizer with sampled hyperparameters
    if optimizer_name == "Adam":
        opt = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                        weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()

    # ⑤ Train for a fixed number of epochs — same for every trial so results are comparable
    n_epochs = 50
    for epoch in range(n_epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad()
            criterion(model(X_b), y_b).backward()
            opt.step()

        # ⑥ Validate at each epoch — needed for pruning
        model.eval()
        correct = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                correct += (model(X_b).argmax(1) == y_b).sum().item()
        val_acc = correct / len(val_ds)

        # Report to Optuna: the pruner uses these intermediate values to decide
        # whether this trial is worth continuing. If accuracy at epoch 10 is
        # much worse than other trials at epoch 10, it gets pruned (stopped early).
        trial.report(val_acc, epoch)   # (value, step)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()  # Saves compute on bad trials

    # Return the final metric — Optuna maximizes (or minimizes) this value
    return val_acc

# ── Create and Run Study ─────────────────────────────────────
study = optuna.create_study(
    direction="maximize",   # "maximize" val_acc; use "minimize" for loss
    sampler=optuna.samplers.TPESampler(seed=42),   # TPE = Bayesian optimization (default)
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,    # Don't prune until at least 5 trials are complete
        n_warmup_steps=10      # Don't prune before epoch 10 (need some training to judge)
    )
)

study.optimize(
    objective,
    n_trials=50,           # Run 50 complete-or-pruned trials
    timeout=600,           # Also stop after 10 minutes (whichever comes first)
    n_jobs=1,              # Parallel trials — use 1 with CUDA (GPU can't share between processes)
    show_progress_bar=True
)

# ── Results ──────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Best trial: #{study.best_trial.number}")
print(f"Best val accuracy: {study.best_value:.4f}")
print("Best hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
```

### Reading the code as a story

The objective function is the heart of every Optuna workflow. Think of it as: "What would you do if someone handed you a set of hyperparameters and said 'try these'?" — you'd build a model with those settings, train it, evaluate it, and report the result. That's exactly what `objective(trial)` does, except `trial` is the one handing you the hyperparameters.

Optuna calls this function 50 times. The first few calls are exploratory (random-ish). After ~10 trials, TPE starts building a model of "which hyperparameter values tended to produce high accuracy" and starts sampling from those regions more often. Each trial informs the next.

### The transcript's concrete tuning mindset

CampusX treats Optuna as an outer loop around a normal PyTorch project:

- keep the training code mostly the same
- move hyperparameters into `trial.suggest_*()` calls
- return validation accuracy from the objective
- let the study run multiple full training trials

The video gradually expands the search space beyond just architecture:

- number of hidden layers
- neurons per layer
- epochs
- learning rate
- dropout rate
- batch size
- optimizer family
- weight decay

---

## 5. Hyperparameter Sampling Methods

### What are the `trial.suggest_*()` calls doing?

Each `suggest_*()` call tells Optuna:

- the name of a hyperparameter
- the allowed range or choices
- how that value should be sampled

So this is not random API boilerplate. This is where you define the search space.

Beginner mental model:

- your model code says how to train
- the `suggest_*()` calls say what Optuna is allowed to vary

```python
# Integer (uniform)
n_layers = trial.suggest_int("n_layers", low=1, high=6)
n_layers = trial.suggest_int("n_layers", low=1, high=6, step=2)  # 1, 3, 5

# Float (uniform)
dropout  = trial.suggest_float("dropout", low=0.0, high=0.7)

# Float (log-uniform) — best for learning rates, weight decay
lr       = trial.suggest_float("lr", low=1e-6, high=1e-1, log=True)

# Categorical (discrete choices)
optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
hidden    = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
act_fn    = trial.suggest_categorical("activation", ["relu", "gelu", "elu"])

# Conditional hyperparameters (SGD only if SGD is chosen)
optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
if optimizer_name == "SGD":
    momentum = trial.suggest_float("momentum", 0.0, 0.99)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
else:
    opt = optim.Adam(model.parameters(), lr=lr)
```

---

## 6. Pruners — Stop Bad Trials Early

### What is a pruner?

A pruner is an early-stop rule for hyperparameter trials.

Instead of waiting for every bad trial to finish all epochs, Optuna can stop obviously weak trials early and save compute for better candidates.

So:

- sampler chooses what to try next
- pruner decides when to stop a trial early

That distinction is easy to miss if you only see the class names.

```python
# MedianPruner: prune if intermediate value < median of completed trials
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,   # Don't prune until 5 trials complete
    n_warmup_steps=10,    # Don't prune first 10 epochs per trial
    interval_steps=1      # Check every epoch
)

# HyperbandPruner: more aggressive, bracket-based
pruner = optuna.pruners.HyperbandPruner(
    min_resource=5, max_resource=50, reduction_factor=3
)

# SuccessiveHalvingPruner (SHA)
pruner = optuna.pruners.SuccessiveHalvingPruner()

# No pruning
pruner = optuna.pruners.NopPruner()
```

---

## 7. Visualization

These plots are not just decoration. They help answer beginner questions like:

- "Is tuning actually improving over time?"
- "Which hyperparameters seem important?"
- "Are there obvious good or bad regions in the search space?"

```python
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_contour,
    plot_slice,
    plot_parallel_coordinate
)

# How val accuracy improved over trials
plot_optimization_history(study).show()

# Which hyperparameters matter most
plot_param_importances(study).show()

# 2D contour of two hyperparameters
plot_contour(study, params=["lr", "dropout"]).show()

# Marginal effect of each hyperparameter
plot_slice(study, params=["lr", "hidden_dim", "dropout"]).show()

# All trials in parallel coordinate view
plot_parallel_coordinate(study).show()
```

---

## 8. Samplers — How Optuna Suggests Hyperparameters

### What is a sampler in Optuna?

Here `sampler` means:

- the strategy Optuna uses to choose the next hyperparameter values

This is a different use of the word from PyTorch `DataLoader` samplers. Same English word, different context.

In Optuna:

- `RandomSampler` tries random values
- `TPESampler` learns from earlier trials and biases future trials toward promising regions
- other samplers use other search strategies

So when you choose a sampler, you are choosing the search behavior of the tuning algorithm itself.

| Sampler | Algorithm | Best for |
|---|---|---|
| `TPESampler` | Tree-structured Parzen Estimator (Bayesian) | **Default, most efficient** |
| `CmaEsSampler` | Evolution strategy | Continuous, medium-dimensional spaces |
| `RandomSampler` | Pure random | Baseline comparison |
| `GridSampler` | Exhaustive grid search | Very few hyperparameters |
| `NSGAIISampler` | Multi-objective Pareto | Multiple conflicting objectives |

```python
# Multi-objective study: maximize accuracy AND minimize model size
def multi_objective(trial):
    n_layers   = trial.suggest_int("n_layers", 1, 5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    ...
    val_acc = train_and_evaluate(...)
    n_params = sum(p.numel() for p in model.parameters())
    return val_acc, -n_params   # maximize acc, minimize param count

study = optuna.create_study(
    directions=["maximize", "maximize"],   # Both maximized
    sampler=optuna.samplers.NSGAIISampler()
)
study.optimize(multi_objective, n_trials=100)

# Pareto front: best tradeoffs
pareto_trials = study.best_trials
```

---

## 9. Hyperparameter Tuning and Experiment Tracking

The transcript makes an important practical point: hyperparameter tuning is much
more useful when paired with **experiment tracking**. Otherwise, you may find a
good trial but still struggle to understand:

- which hyperparameters mattered most
- whether improvements were stable or lucky
- what tradeoffs you made between speed and accuracy

```python
# Convert trials to a pandas-friendly table
df = study.trials_dataframe()
print(df[[
    "number", "value", "params_lr", "params_dropout",
    "params_hidden_dim", "params_optimizer", "state"
]].head())

# Sort best trials
best_trials = df.sort_values("value", ascending=False).head(10)
print(best_trials)
```

### Why this matters in practice

- tuning without records becomes guesswork
- records help you spot stable patterns instead of one-off wins
- the same Optuna workflow can be reused later for CNNs, RNNs, and other models

### What experiment tracking adds beyond "best params"

Another transcript point is that "best params" are only part of the story. With
tracked runs and plots, you can also ask:

- which parameter values got sampled most often by the search
- which value ranges repeatedly gave good accuracy
- whether some optimizer families consistently outperform others

That is why experiment tracking tools like MLflow are useful companions to
Optuna: they help turn tuning from a one-off search into something you can
actually analyze.

---

## 10. Saving and Loading Studies

### Why save a study?

Because hyperparameter tuning can take a long time.

If you only keep results in memory:

- an interrupted run is lost
- you cannot resume later
- you cannot compare future runs against the same study history

Saving the study turns tuning into an iterative process instead of a one-shot experiment.

```python
import optuna

# Save to SQLite (persistent across crashes)
study = optuna.create_study(
    study_name="breast_cancer_tuning",
    storage="sqlite:///optuna_study.db",
    direction="maximize",
    load_if_exists=True   # Resume if study exists
)

# Load later
study = optuna.load_study(
    study_name="breast_cancer_tuning",
    storage="sqlite:///optuna_study.db"
)

print(f"Trials completed: {len(study.trials)}")
```

---

## 11. Using Best Hyperparameters

```python
# After study.optimize() completes:
best_params = study.best_params
print(best_params)
# {'n_layers': 2, 'hidden_dim': 256, 'dropout': 0.23, 'lr': 0.00134, ...}

# Build final model with best hyperparameters
final_model = build_model(
    n_layers=best_params['n_layers'],
    hidden_dim=best_params['hidden_dim'],
    dropout=best_params['dropout']
).to(DEVICE)

final_optimizer = optim.AdamW(
    final_model.parameters(),
    lr=best_params['lr'],
    weight_decay=best_params['weight_decay']
)

# Train for longer on full training set (no validation split)
# Then evaluate on held-out test set
```

---

## 12. Interview Questions

<details>
<summary><strong>Q1: What is Optuna and why is it better than grid search?</strong></summary>

Optuna is an automatic hyperparameter optimization framework using Bayesian optimization (TPE sampler). It builds a probabilistic model of which hyperparameter regions give good results and samples the next trial from promising regions. Grid search tries all combinations — exponential in the number of hyperparameters. Optuna typically finds comparable or better results in 10–20× fewer trials by learning from previous experiments.
</details>

<details>
<summary><strong>Q2: What is the TPE sampler in Optuna?</strong></summary>

TPE (Tree-structured Parzen Estimator) is a Bayesian optimization algorithm. It models two distributions: `l(x)` — distribution of hyperparameters from good trials, and `g(x)` — from bad trials. It samples the next hyperparameter configuration to maximize `l(x)/g(x)` — favoring regions that were good before. It's called "tree-structured" because it handles conditional hyperparameters (e.g., momentum only if SGD is selected) efficiently.
</details>

<details>
<summary><strong>Q3: What is a pruner in Optuna and why use it?</strong></summary>

A pruner monitors intermediate training results and stops a trial early if it's clearly worse than previous trials. For example, MedianPruner stops a trial at epoch 10 if its accuracy at epoch 10 is below the median of all trials at epoch 10. This saves significant compute — instead of training a bad trial for 100 epochs, it's stopped at epoch 10. This lets you run more total trials in the same time budget.
</details>

<details>
<summary><strong>Q4: What is the difference between a hyperparameter and a model parameter?</strong></summary>

**Model parameters** (weights, biases) are learned during training by gradient descent — they are updated by the optimizer every batch. **Hyperparameters** are configuration choices made before training — they are NOT learned by gradient descent. Examples: learning rate (how fast to update weights), number of layers (model size), dropout rate (regularization strength), batch size (training dynamics). Hyperparameter tuning uses outer-loop optimization (Optuna) to find the best configuration.
</details>

<details>
<summary><strong>Q5: How do you avoid overfitting to the validation set during hyperparameter search?</strong></summary>

With many Optuna trials, you're implicitly optimizing for the validation set — a form of overfitting. Solutions: (1) **Hold out a separate test set** that is never used during tuning — report final performance on this; (2) **K-fold cross-validation** in the objective function — more expensive but more reliable; (3) **Limit the number of trials** to prevent extensive overfitting to validation noise; (4) **Use larger validation sets** so performance estimates are more reliable.
</details>

---

## 🔗 References
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [CampusX Video 10](https://www.youtube.com/watch?v=Y3s-wBBLj_o)
