---
title: Arrays
sidebar_position: 4
description: Complete notes on arrays, memory layout, common operations, patterns, and code examples.
---

# Arrays

An array stores values in contiguous memory. This gives fast indexing and predictable traversal.

## Why arrays matter

- simplest and most common data structure
- foundation for strings, matrices, heaps, and dynamic arrays
- many interview problems are array pattern problems

## Core properties

- random access: usually `O(1)`
- append at end of dynamic array: amortized `O(1)`
- insertion in middle: `O(n)`
- deletion in middle: `O(n)`

## Contiguous memory

If each element has fixed size, the address of `arr[i]` can be computed directly.

That is why indexed lookup is fast.

## Common operations

```python
arr = [10, 20, 30, 40]

print(arr[2])     # 30
arr.append(50)
arr[1] = 25
```

## Core patterns

### Prefix sum

Used when repeated range sum queries are needed.

```python
def prefix_sums(nums: list[int]) -> list[int]:
    pref = [0]
    for num in nums:
        pref.append(pref[-1] + num)
    return pref


def range_sum(pref: list[int], left: int, right: int) -> int:
    return pref[right + 1] - pref[left]
```

### Two pointers

Used for sorted arrays, partitioning, and pair problems.

```python
def has_pair_with_sum(nums: list[int], target: int) -> bool:
    nums.sort()
    left, right = 0, len(nums) - 1
    while left < right:
        total = nums[left] + nums[right]
        if total == target:
            return True
        if total < target:
            left += 1
        else:
            right -= 1
    return False
```

### Sliding window

Useful for contiguous subarray constraints.

```python
def max_sum_subarray_k(nums: list[int], k: int) -> int:
    window_sum = sum(nums[:k])
    best = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        best = max(best, window_sum)
    return best
```

## 2D arrays

Matrices are arrays of arrays.

```python
grid = [
    [1, 2, 3],
    [4, 5, 6],
]
```

Traversal:

```python
for row in range(len(grid)):
    for col in range(len(grid[0])):
        print(grid[row][col])
```

## Dynamic arrays

Languages often provide arrays that grow automatically.

Examples:

- Python `list`
- Java `ArrayList`
- C++ `vector`

They internally resize when capacity is exceeded.

## Worked example

Find the maximum sum of any contiguous subarray.

Brute force checks all subarrays in `O(n^2)` or worse.

Kadane's algorithm reduces it to `O(n)`.

```python
def max_subarray(nums: list[int]) -> int:
    current = best = nums[0]
    for x in nums[1:]:
        current = max(x, current + x)
        best = max(best, current)
    return best
```

### Explanation

- either extend the previous subarray or start fresh at current element
- keep the best answer seen so far

## Common mistakes

- forgetting boundary conditions on empty arrays
- mixing fixed window and variable window logic
- using nested loops when prefix or window patterns apply
- sorting when original order must be preserved

## Practice prompts

- move zeros to the end in-place
- find longest subarray with sum `k`
- rotate array by `k`
- merge overlapping intervals

## Quick revision

- arrays are contiguous and indexable
- many array problems reduce to a small set of patterns
- knowing the pattern matters more than memorizing solutions
