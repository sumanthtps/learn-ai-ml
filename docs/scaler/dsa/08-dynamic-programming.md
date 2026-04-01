---
title: Dynamic Programming
sidebar_position: 8
description: Dynamic programming fundamentals, state design, patterns, and examples.
---

# Dynamic Programming

Dynamic programming solves problems with overlapping subproblems by reusing earlier results.

## Signals that DP may apply

- same subproblem appears again
- the best answer depends on best answers to smaller subproblems

## Two styles

- memoization: top-down recursion plus caching
- tabulation: bottom-up iteration

## Example: Fibonacci

Memoization:

```python
def fib(n: int, memo=None) -> int:
    if memo is None:
        memo = {}
    if n <= 1:
        return n
    if n in memo:
        return memo[n]
    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)
    return memo[n]
```

### In-depth explanation

This is top-down dynamic programming.

Why plain recursion is wasteful:

- `fib(5)` calls `fib(4)` and `fib(3)`
- `fib(4)` again calls `fib(3)` and `fib(2)`
- the same subproblems repeat many times

How memoization fixes it:

- `memo` stores previously computed answers
- before solving a subproblem, check whether it already exists

Walkthrough:

- initialize `memo` only once if not provided
- base case returns `n` for `0` and `1`
- if answer already exists, reuse it
- otherwise compute recursively, store it, and return it

Complexity:

- time: `O(n)`
- space: `O(n)`

Tabulation:

```python
def fib_tab(n: int) -> int:
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
```

### In-depth explanation

This is bottom-up DP.

Meaning of state:

- `dp[i]` = Fibonacci value at index `i`

Why it is called tabulation:

- we fill a table from smaller answers to larger answers

Walkthrough:

- base values are known for `0` and `1`
- loop from `2` upward
- each state depends only on the two previous states

Why this is often easier to debug:

- computation order is explicit
- no recursion stack is involved

Complexity:

- time: `O(n)`
- space: `O(n)` though it can be optimized to `O(1)`

## DP workflow

1. Define the state.
2. Define the transition.
3. Set base cases.
4. Decide computation order.

## Example: coin change

```python
def coin_change(coins: list[int], amount: int) -> int:
    INF = amount + 1
    dp = [INF] * (amount + 1)
    dp[0] = 0
    for x in range(1, amount + 1):
        for coin in coins:
            if x - coin >= 0:
                dp[x] = min(dp[x], dp[x - coin] + 1)
    return -1 if dp[amount] == INF else dp[amount]
```

### Explanation

- `dp[x]` means minimum coins needed to make amount `x`
- transition checks all usable previous states

### In-depth code walkthrough

State meaning:

- `dp[x]` = minimum number of coins required to make amount `x`

Why `INF = amount + 1`:

- no valid solution can require more than `amount` coins if coin `1` exists
- so `amount + 1` is a safe "impossible for now" sentinel

Initialization:

- all states start as impossible
- `dp[0] = 0` because zero coins are needed to make amount zero

Nested loops:

- outer loop builds answers from small amounts upward
- inner loop tries every coin as the last coin used

Transition:

```python
dp[x] = min(dp[x], dp[x - coin] + 1)
```

means:

- if we use `coin` last, then we must already know the best way to form `x - coin`
- add one more coin for the current choice
- keep the better option

Final check:

- if `dp[amount]` is still `INF`, no solution exists

Complexity:

- time: `O(amount * len(coins))`
- space: `O(amount)`

## Pattern families

- one-dimensional DP
- grid DP
- knapsack
- subsequence DP
- interval DP
- tree DP
- bitmask DP

## Space optimization

Sometimes only previous states are needed, so a full table is unnecessary.

## Common mistakes

- writing loops before defining state meaning
- bad base cases
- storing more state than necessary
- not checking whether greedy already solves the problem

## Practice prompts

- house robber
- longest increasing subsequence
- edit distance
- unique paths
- partition equal subset sum

## Quick revision

- DP is state design plus reuse
- memoization and tabulation solve the same recurrence differently
- if the state is unclear, the solution will stay unclear
