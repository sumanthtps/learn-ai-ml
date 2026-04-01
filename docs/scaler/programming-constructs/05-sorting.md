---
title: Sorting
sidebar_position: 5
description: Complete notes on sorting algorithms, tradeoffs, stability, and code examples.
---

# Sorting

Sorting arranges data according to an order. It is often used not just as a final goal, but as a preprocessing step that makes later logic simpler.

## Why sorting matters

- enables binary search
- simplifies interval problems
- helps grouping and deduplication
- is a recurring interview topic

## Simple sorting algorithms

### Bubble sort

Repeatedly swaps adjacent out-of-order elements.

- time: `O(n^2)`
- space: `O(1)`
- stable: yes

### Selection sort

Selects the smallest remaining element each round.

- time: `O(n^2)`
- space: `O(1)`
- stable: no

### Insertion sort

Inserts each new element into its correct position in the sorted prefix.

- time: `O(n^2)` worst case
- space: `O(1)`
- stable: yes
- good for small or nearly sorted input

## Efficient sorting algorithms

### Merge sort

- divide array into halves
- sort both halves
- merge them

- time: `O(n log n)`
- space: `O(n)`
- stable: yes

```python
def merge_sort(nums: list[int]) -> list[int]:
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])
    return merge(left, right)


def merge(left: list[int], right: list[int]) -> list[int]:
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out
```

### Quicksort

- choose pivot
- partition into smaller and larger elements
- recursively sort both sides

- average: `O(n log n)`
- worst: `O(n^2)`
- space: recursion-dependent
- stable: usually no

### Heapsort

- build heap
- repeatedly extract largest or smallest

- time: `O(n log n)`
- space: `O(1)` extra
- stable: no

## Stability

A stable sort preserves the relative order of equal elements.

This matters when:

- sorting records by multiple fields
- preserving prior ordering logic

## Comparison sorting lower bound

General comparison-based sorting cannot do better than `O(n log n)` in the worst case.

That is why merge sort, heapsort, and average-case quicksort are so important.

## Non-comparison sorting

When values have constraints, faster approaches may exist.

- counting sort
- radix sort
- bucket sort

These can beat `O(n log n)` in special cases.

## Worked example

Sort student records by score while preserving original order for equal scores.

A stable sort is preferred.

```python
students = [("A", 90), ("B", 75), ("C", 90)]
print(sorted(students, key=lambda x: x[1]))
```

Output:

```text
[('B', 75), ('A', 90), ('C', 90)]
```

`A` stays before `C` because Python's built-in sort is stable.

## Common mistakes

- sorting without checking whether order must be preserved
- forgetting sorting adds `O(n log n)` cost
- choosing quicksort blindly when worst-case guarantees matter

## Practice prompts

- implement merge sort
- sort intervals by start time
- sort by frequency, then by value

## Quick revision

- insertion sort is good for small nearly sorted data
- merge sort is stable and predictable
- quicksort is fast in practice
- sorting is often a preprocessing trick, not just an end goal
