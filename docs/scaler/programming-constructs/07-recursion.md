---
title: Recursion
sidebar_position: 7
description: Complete notes on recursion, recursion trees, base cases, and code examples.
---

# Recursion

Recursion solves a problem by solving smaller versions of the same problem.

## Why recursion matters

- natural for trees and divide-and-conquer
- foundational for backtracking and dynamic programming
- common in interviews

## Three parts of recursion

- base case
- recursive call
- progress toward termination

## Simple example: factorial

```python
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

### Explanation

- base case: `n <= 1`
- recursive relation: `n * factorial(n - 1)`
- each call reduces `n`

## Thinking with the call stack

For `factorial(4)`:

```text
factorial(4)
 -> 4 * factorial(3)
 -> 4 * 3 * factorial(2)
 -> 4 * 3 * 2 * factorial(1)
 -> 4 * 3 * 2 * 1
 = 24
```

## Another example: sum of array

```python
def array_sum(nums: list[int], i: int = 0) -> int:
    if i == len(nums):
        return 0
    return nums[i] + array_sum(nums, i + 1)
```

## Recursion tree idea

When a function branches into multiple recursive calls, imagine a tree of work.

This helps analyze:

- correctness
- total number of calls
- time complexity

## Divide and conquer

Recursion often appears in divide-and-conquer algorithms:

- merge sort
- quicksort
- binary search

Binary search:

```python
def binary_search(nums: list[int], target: int, left: int, right: int) -> int:
    if left > right:
        return -1
    mid = (left + right) // 2
    if nums[mid] == target:
        return mid
    if nums[mid] < target:
        return binary_search(nums, target, mid + 1, right)
    return binary_search(nums, target, left, mid - 1)
```

## Recursion vs iteration

Recursion is often cleaner when:

- the structure is hierarchical
- the problem is naturally defined in smaller subproblems

Iteration is often safer when:

- recursion depth may be too high
- the process is simple and linear

## Tail recursion

Tail recursion means the recursive call is the last operation.

Some languages optimize this. Python does not.

## Common mistakes

- missing base case
- not reducing the problem
- changing shared state carelessly
- ignoring stack overflow risk

## Practice prompts

- reverse a string recursively
- check palindrome recursively
- compute power recursively
- implement merge sort recursively

## Quick revision

- recursion needs a base case and progress
- call-stack thinking is essential
- recursion is a modeling tool, not just a coding style
