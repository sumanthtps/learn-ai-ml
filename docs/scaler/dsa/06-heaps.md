---
title: Heaps
sidebar_position: 6
description: Heap fundamentals, priority queues, top-k patterns, and code examples.
---

# Heaps

A heap is a complete binary tree usually implemented using an array. It supports efficient access to the smallest or largest element.

## Types

- min-heap
- max-heap

## Array representation

For index `i`:

- left child: `2*i + 1`
- right child: `2*i + 2`
- parent: `(i - 1) // 2`

## Python example

```python
import heapq

nums = [5, 1, 9, 3]
heapq.heapify(nums)
print(heapq.heappop(nums))  # 1
```

### In-depth explanation

Python's `heapq` implements a min-heap.

Walkthrough:

- `nums` starts as a regular list
- `heapq.heapify(nums)` rearranges it in-place so the heap property holds
- after heapify, the smallest element is guaranteed to be at index `0`
- `heapq.heappop(nums)` removes and returns that smallest element

Important note:

- the internal array is heap-ordered, not fully sorted
- only the minimum-at-root guarantee exists

## Top-k largest elements

```python
import heapq


def top_k(nums: list[int], k: int) -> list[int]:
    heap = []
    for x in nums:
        heapq.heappush(heap, x)
        if len(heap) > k:
            heapq.heappop(heap)
    return sorted(heap, reverse=True)
```

### In-depth explanation

This keeps a min-heap of size at most `k`.

Why a min-heap:

- among the current top `k` largest values, the smallest of them is the easiest one to discard
- the heap root gives that smallest-of-the-top-k value efficiently

Step by step:

- push every number into the heap
- if heap grows beyond size `k`, pop once
- that pop removes the smallest number currently in the heap
- by the end, only the `k` largest numbers remain

Why final sorting is needed:

- a heap is not fully sorted
- the question returns the top `k` elements in descending order for readability

Complexity:

- time: `O(n log k)`
- space: `O(k)`

This is better than full sorting when `k` is much smaller than `n`.

## Merge k sorted lists

Heaps are great when repeated minimum extraction is needed.

## Streaming median idea

Use:

- max-heap for lower half
- min-heap for upper half

This supports efficient median maintenance.

## Heapify

Building a heap from an array can be done in linear time.

That is a common interview detail people miss.

## Common mistakes

- sorting everything when only top `k` is needed
- forgetting heap direction
- assuming heap gives fully sorted order without repeated pops

## Practice prompts

- kth largest element
- merge k sorted arrays
- task scheduler
- running median

## Quick revision

- heaps support repeated extreme-value access
- priority queues are often heap-backed
- if only top `k` matters, a heap is often cleaner than full sort
