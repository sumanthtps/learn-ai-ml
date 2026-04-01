---
title: Arrays Patterns
sidebar_position: 1
description: Array problem-solving patterns, examples, and code for interviews.
---

# Arrays Patterns

At the DSA level, array questions are mostly pattern-recognition questions.

## High-value patterns

- prefix sums
- sliding window
- two pointers
- monotonic stack
- binary search on answer
- Kadane's algorithm

## Prefix sums

Useful when many range sums are needed.

```python
def build_prefix(nums: list[int]) -> list[int]:
    pref = [0]
    for x in nums:
        pref.append(pref[-1] + x)
    return pref
```

### In-depth explanation

The idea of prefix sums is to precompute cumulative totals so that later range queries become constant time.

Line by line:

- `pref = [0]` creates a dummy starting sum before any element is included
- for each `x`, the code appends the previous cumulative sum plus the new value
- after the loop, `pref[i]` means "sum of the first `i` elements"

Why the dummy `0` helps:

- it avoids special cases for ranges starting at index `0`
- it lets us write one clean subtraction formula for every range

Example:

If `nums = [2, 5, 1, 4]`, then:

- `pref[0] = 0`
- `pref[1] = 2`
- `pref[2] = 7`
- `pref[3] = 8`
- `pref[4] = 12`

So `pref` becomes `[0, 2, 7, 8, 12]`.

Complexity:

- build time: `O(n)`
- extra space: `O(n)`

Range sum from `l` to `r`:

```python
def query(pref: list[int], l: int, r: int) -> int:
    return pref[r + 1] - pref[l]
```

### In-depth explanation

This function uses the prefix-array meaning directly.

- `pref[r + 1]` is the sum of elements from index `0` through `r`
- `pref[l]` is the sum of elements before index `l`
- subtracting removes the unwanted left portion

So the answer is:

```text
sum(l..r) = prefix(up to r) - prefix(before l)
```

Example:

For `nums = [2, 5, 1, 4]`, range sum from `1` to `3` is:

- `pref[4] = 12`
- `pref[1] = 2`
- answer = `12 - 2 = 10`

That matches `5 + 1 + 4 = 10`.

Complexity:

- per query: `O(1)`

## Sliding window

Used for contiguous segments.

Example: longest subarray with sum at most `k` for positive numbers.

```python
def longest_window(nums: list[int], k: int) -> int:
    left = 0
    total = 0
    best = 0
    for right, x in enumerate(nums):
        total += x
        while total > k:
            total -= nums[left]
            left += 1
        best = max(best, right - left + 1)
    return best
```

### In-depth explanation

This is a classic variable-size sliding-window algorithm, and it works because all numbers are assumed to be positive.

Key invariant:

- the current window is `nums[left:right+1]`
- `total` always stores the sum of exactly that window

Why positivity matters:

- when the sum becomes too large, moving `left` forward always decreases the sum
- that makes the shrinking step valid and predictable

Walkthrough:

- `left = 0` starts the window at the beginning
- `for right, x in enumerate(nums)` expands the window one element at a time
- `total += x` includes the new rightmost element
- `while total > k` keeps shrinking from the left until the constraint is satisfied again
- `best = max(best, right - left + 1)` records the longest valid window seen so far

Example with `nums = [1, 2, 1, 1, 3]` and `k = 4`:

- expand to `[1,2,1]`, sum `4`, valid
- expand to `[1,2,1,1]`, sum `5`, too large
- shrink left, new window `[2,1,1]`, sum `4`, valid

Complexity:

- time: `O(n)` because each pointer moves at most `n` times
- space: `O(1)`

## Kadane's algorithm

Maximum subarray sum:

```python
def max_subarray(nums: list[int]) -> int:
    best = current = nums[0]
    for x in nums[1:]:
        current = max(x, current + x)
        best = max(best, current)
    return best
```

### In-depth explanation

This is Kadane's algorithm. The crucial idea is that for each position, the best subarray ending at that position has only two possibilities:

- start fresh from the current value `x`
- extend the previous best-ending-here subarray with `x`

Meaning of variables:

- `current` = best subarray sum that must end at the current index
- `best` = best subarray sum seen anywhere so far

Why `current = max(x, current + x)` works:

- if the old running sum is hurting us, drop it and restart at `x`
- if the old running sum helps, extend it

Example for `[-2, 1, -3, 4, -1, 2, 1, -5, 4]`:

- at `4`, restarting is better than extending the negative history
- then `4 + (-1) + 2 + 1 = 6` becomes the global best

Complexity:

- time: `O(n)`
- space: `O(1)`

## Two pointers

Best when the array is sorted or a window is expanding and shrinking.

Example: pair with target sum.

```python
def pair_sum_sorted(nums: list[int], target: int) -> bool:
    i, j = 0, len(nums) - 1
    while i < j:
        s = nums[i] + nums[j]
        if s == target:
            return True
        if s < target:
            i += 1
        else:
            j -= 1
    return False
```

### In-depth explanation

This two-pointer method depends on the array being sorted.

Why sorting order helps:

- if the sum is too small, moving the left pointer right is the only move that can increase it
- if the sum is too large, moving the right pointer left is the only move that can decrease it

Variables:

- `i` points to the current smallest candidate
- `j` points to the current largest candidate

At each step:

- compute `s = nums[i] + nums[j]`
- if `s == target`, we found a pair
- if `s < target`, the smaller number is too small, so increment `i`
- if `s > target`, the larger number is too large, so decrement `j`

This avoids checking every pair.

Complexity:

- time: `O(n)` after sorting, or `O(n)` directly if input is already sorted
- space: `O(1)`

## Monotonic stack

Used for next greater or smaller element style problems.

```python
def next_greater(nums: list[int]) -> list[int]:
    ans = [-1] * len(nums)
    stack = []
    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] < x:
            ans[stack.pop()] = x
        stack.append(i)
    return ans
```

### In-depth explanation

This uses a monotonic decreasing stack of indices.

What the stack represents:

- indices whose next greater element has not been found yet
- values at those indices are kept in decreasing order

Walkthrough:

- `ans` starts as all `-1`, meaning "not found yet"
- for each new value `x`, the code checks whether `x` is greater than the values indexed on top of the stack
- while that is true, `x` is the answer for those smaller earlier elements
- after resolving all such indices, the current index is pushed because its own answer is still unknown

Example for `[2, 1, 4, 3]`:

- stack has indices for `2`, then `1`
- when `4` arrives, it becomes the next greater element for both `1` and `2`

Why indices are stored instead of values:

- we need to write the answer into the correct position of `ans`

Complexity:

- time: `O(n)` because each index is pushed and popped at most once
- space: `O(n)`

## Binary search on answer

Sometimes the answer is not an index but a value range.

Example use cases:

- minimum eating speed
- minimum capacity
- maximize minimum distance

## Common mistakes

- using sliding window when negative values break the invariant
- sorting too early and losing original index meaning
- missing monotonic-stack opportunities

## Practice prompts

- product of array except self
- trapping rain water
- longest consecutive sequence
- maximum circular subarray

## Quick revision

- arrays are about patterns
- preprocessing often turns slow brute force into fast solutions
- recognize whether the problem is about range, order, window, or extremum
