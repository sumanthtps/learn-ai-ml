---
title: Time Complexity
sidebar_position: 3
description: Complete notes on Big-O, growth rates, amortized analysis, and complexity reasoning.
---

# Time Complexity

Time complexity measures how runtime grows as input size grows. It is a comparison tool, not a stopwatch.

## Why it matters

- helps choose between solutions before coding
- allows reasoned optimization
- is required in interviews and performance discussions

## Common classes

- `O(1)` constant
- `O(log n)` logarithmic
- `O(n)` linear
- `O(n log n)` divide-and-conquer scale
- `O(n^2)` quadratic
- `O(2^n)` exponential
- `O(n!)` factorial

## Reading loops

Single loop:

```python
for i in range(n):
    print(i)
```

Complexity:

```text
O(n)
```

Nested loops:

```python
for i in range(n):
    for j in range(n):
        print(i, j)
```

Complexity:

```text
O(n^2)
```

Halving pattern:

```python
while n > 1:
    n //= 2
```

Complexity:

```text
O(log n)
```

## Rules of thumb

- sequential independent blocks usually add
- nested dependent loops usually multiply
- drop constant multipliers in asymptotic notation
- keep the dominant term only

## Example analysis

```python
def example(nums: list[int]) -> int:
    total = 0
    for x in nums:
        total += x
    for i in range(100):
        total += i
    return total
```

Analysis:

- first loop: `O(n)`
- second loop: `O(100)` which becomes `O(1)`
- total: `O(n + 1) = O(n)`

## Space complexity

Time is not enough. Also ask how much extra memory is used.

```python
def squares(nums: list[int]) -> list[int]:
    out = []
    for x in nums:
        out.append(x * x)
    return out
```

- time: `O(n)`
- extra space: `O(n)`

## Recursion and recurrence

Factorial:

```python
def fact(n: int) -> int:
    if n <= 1:
        return 1
    return n * fact(n - 1)
```

- time: `O(n)`
- stack space: `O(n)`

Merge sort:

```text
T(n) = 2T(n/2) + O(n)
```

Result:

```text
O(n log n)
```

## Amortized analysis

Dynamic arrays occasionally resize and copy elements, but average append cost is still `O(1)` amortized.

That means:

- not every operation is cheap
- but the average over many operations is cheap

## Best, average, and worst case

Linear search:

- best case: target is first item, `O(1)`
- worst case: target is last or absent, `O(n)`
- average case: still grows linearly

## Comparison vs real performance

Asymptotic complexity is necessary but incomplete.

Real performance also depends on:

- constants
- memory layout
- cache locality
- input distribution
- language runtime overhead

## Code example: comparing approaches

Two-sum brute force:

```python
def two_sum_bruteforce(nums: list[int], target: int) -> tuple[int, int] | None:
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return i, j
    return None
```

- time: `O(n^2)`
- space: `O(1)`

Hash map approach:

```python
def two_sum_hash(nums: list[int], target: int) -> tuple[int, int] | None:
    seen = {}
    for i, num in enumerate(nums):
        need = target - num
        if need in seen:
            return seen[need], i
        seen[num] = i
    return None
```

- time: `O(n)`
- space: `O(n)`

## Common mistakes

- saying nested loops always mean `O(n^2)` even when inner work shrinks
- ignoring auxiliary space
- mixing input size with data values
- forgetting recursion stack cost

## Practice prompts

- Analyze binary search.
- Compare sorting then searching vs hashing.
- Explain amortized `O(1)` append.

## Quick revision

- Big-O is about growth, not exact runtime
- dominant term matters most
- always mention space complexity too
- optimization is meaningful only when complexity reasoning is sound
