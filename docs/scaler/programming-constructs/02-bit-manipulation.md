---
title: Bit Manipulation
sidebar_position: 2
description: Complete notes on bitwise operators, common bit tricks, masks, and coding patterns.
---

# Bit Manipulation

Bit manipulation means operating directly on the binary representation of numbers. It is especially useful when state can be stored compactly or when a mathematical shortcut exists in binary form.

## Core operators

- `&` bitwise AND
- `|` bitwise OR
- `^` bitwise XOR
- `~` bitwise NOT
- `<<` left shift
- `>>` right shift

## Intuition for each operator

```text
1 & 1 = 1
1 & 0 = 0
1 | 0 = 1
1 ^ 1 = 0
1 ^ 0 = 1
```

## Common operations

Check if number is even:

```python
def is_even(n: int) -> bool:
    return (n & 1) == 0
```

Set the `i`th bit:

```python
def set_bit(n: int, i: int) -> int:
    return n | (1 << i)
```

Clear the `i`th bit:

```python
def clear_bit(n: int, i: int) -> int:
    return n & ~(1 << i)
```

Toggle the `i`th bit:

```python
def toggle_bit(n: int, i: int) -> int:
    return n ^ (1 << i)
```

Check the `i`th bit:

```python
def is_set(n: int, i: int) -> bool:
    return (n & (1 << i)) != 0
```

## Very important identities

Remove the lowest set bit:

```python
n & (n - 1)
```

Isolate the lowest set bit:

```python
n & -n
```

Check if power of two:

```python
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
```

## Worked example

Let `n = 10`.

```text
10 = 1010₂
```

Toggle bit `1`:

```text
1010
0010
----
1000 = 8
```

Set bit `0`:

```text
1010
0001
----
1011 = 11
```

## Counting set bits

Naive:

```python
def count_set_bits(n: int) -> int:
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count
```

Optimized with Brian Kernighan's trick:

```python
def count_set_bits_fast(n: int) -> int:
    count = 0
    while n:
        n = n & (n - 1)
        count += 1
    return count
```

### Why it works

Every time you do `n & (n - 1)`, the rightmost set bit disappears. So the loop runs only as many times as there are set bits.

## XOR patterns

Useful XOR facts:

- `a ^ a = 0`
- `a ^ 0 = a`
- XOR is commutative and associative

Find the unique element where every other element appears twice:

```python
def single_number(nums: list[int]) -> int:
    answer = 0
    for num in nums:
        answer ^= num
    return answer
```

## Bitmasking for subsets

If a set has `n` elements, there are `2^n` subsets. A mask from `0` to `2^n - 1` can represent which elements are included.

```python
def all_subsets(nums: list[int]) -> list[list[int]]:
    result = []
    n = len(nums)
    for mask in range(1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        result.append(subset)
    return result
```

## Where bit manipulation appears

- permissions and flags
- subset generation
- optimization problems
- low-level systems code
- compression and encoding
- bitmask dynamic programming

## Common mistakes

- using 1-based instead of 0-based bit positions
- forgetting operator precedence
- ignoring signed right-shift behavior in some languages
- using bit tricks when clarity matters more than cleverness

## Practice prompts

- Count set bits in `29`
- Check whether `64` is a power of two
- Generate all subsets of `[1, 2, 3]`

## Quick revision

- `n & 1` checks odd or even
- `n & (n - 1)` removes the lowest set bit
- XOR is excellent for canceling duplicates
- bitmasks are compact set representations
