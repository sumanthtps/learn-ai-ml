---
title: Backtracking
sidebar_position: 7
description: Backtracking concepts, recursion templates, pruning, and examples.
---

# Backtracking

Backtracking explores possible choices step by step and abandons paths that cannot lead to a valid solution.

## Core template

- choose
- recurse
- undo

## Example: subsets

```python
def subsets(nums: list[int]) -> list[list[int]]:
    result = []
    path = []

    def dfs(i: int) -> None:
        if i == len(nums):
            result.append(path[:])
            return

        path.append(nums[i])
        dfs(i + 1)
        path.pop()

        dfs(i + 1)

    dfs(0)
    return result
```

### In-depth explanation

This code generates all subsets using the include-exclude pattern.

State meaning:

- `i` = current index being decided
- `path` = subset currently being built
- `result` = all finished subsets

At each index, there are exactly two choices:

- include `nums[i]`
- exclude `nums[i]`

Walkthrough:

- if `i == len(nums)`, all decisions are done, so append a copy of `path`
- `path.append(nums[i])` chooses inclusion
- `dfs(i + 1)` explores all subsets containing that element
- `path.pop()` undoes the choice
- second `dfs(i + 1)` explores all subsets without that element

Why `path[:]` is necessary:

- `path` is mutable
- without copying, all stored answers would point to the same changing list

Complexity:

- time: `O(2^n * n)` if output construction is counted
- space: `O(n)` recursion depth excluding output

## Example: permutations

```python
def permutations(nums: list[int]) -> list[list[int]]:
    result = []
    used = [False] * len(nums)
    path = []

    def dfs():
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            dfs()
            path.pop()
            used[i] = False

    dfs()
    return result
```

### In-depth explanation

Permutations are harder than subsets because position matters.

State meaning:

- `path` = current permutation being built
- `used[i]` = whether `nums[i]` is already placed

At each recursion level:

- try every element not already used
- choose it
- recurse
- undo the choice

Walkthrough:

- if `len(path) == len(nums)`, one full permutation is complete
- loop through indices
- skip used items
- mark chosen element as used
- append it to path
- recurse deeper
- after return, undo both path and used-state changes

Why undo is essential:

- the same shared arrays are reused across branches
- without undo, later branches inherit wrong state

Complexity:

- time: `O(n! * n)`
- space: `O(n)` recursion stack excluding output

## Pruning

Pruning means stopping exploration early.

Examples:

- stop when sum already exceeds target
- stop when a row or column is invalid
- skip duplicates when they would create repeated answers

## Common problem families

- subsets
- permutations
- combination sum
- n-queens
- sudoku

## How to think clearly

For any backtracking problem, define:

- current state
- available choices
- stopping condition
- undo step

## Common mistakes

- forgetting to undo state
- appending mutable lists without copying
- exploring duplicate branches

## Practice prompts

- combination sum
- letter combinations of phone number
- n-queens
- word search

## Quick revision

- backtracking is structured search
- pruning is what makes it practical
- state design is more important than syntax
