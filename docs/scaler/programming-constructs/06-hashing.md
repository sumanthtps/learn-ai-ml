---
title: Hashing
sidebar_position: 6
description: Complete notes on hash maps, hash sets, collisions, use cases, and code examples.
---

# Hashing

Hashing maps a key to a bucket or index using a hash function. It gives very fast average-case lookup, insert, and delete.

## Why hashing matters

- converts many nested-loop problems into one-pass problems
- powers dictionaries, sets, indexes, and caches
- is one of the most common interview optimizations

## Core structures

- hash map: key to value
- hash set: unique keys only

## Collision handling

Two keys can map to the same bucket. Common strategies:

- chaining
- open addressing

In practice, languages hide these details, but you should know collisions exist.

## Common uses

- membership check
- frequency count
- complement lookup
- duplicate removal
- grouping by key

## Code examples

Frequency counter:

```python
def char_frequency(s: str) -> dict[str, int]:
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    return freq
```

Two-sum:

```python
def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    seen = {}
    for i, num in enumerate(nums):
        need = target - num
        if need in seen:
            return seen[need], i
        seen[num] = i
    return None
```

Group anagrams:

```python
def group_anagrams(words: list[str]) -> dict[str, list[str]]:
    groups = {}
    for word in words:
        key = "".join(sorted(word))
        groups.setdefault(key, []).append(word)
    return groups
```

## Load factor and resizing

If too many elements share too few buckets, performance drops.

That is why hash tables:

- maintain a load factor
- resize when needed
- rehash elements into the new table

## Average vs worst case

- average lookup: `O(1)`
- worst case: can degrade, depending on collisions

For interviews, it is fine to say hash maps are average-case `O(1)`.

## Advanced ideas

- rolling hash for strings
- consistent hashing in distributed systems
- bloom filters for approximate membership checks

## Worked example

Count how many numbers appear more than once.

```python
def count_duplicates(nums: list[int]) -> int:
    freq = {}
    for x in nums:
        freq[x] = freq.get(x, 0) + 1
    return sum(1 for count in freq.values() if count > 1)
```

### Explanation

- one pass builds frequency counts
- second pass counts repeated keys

## Common mistakes

- assuming hash maps preserve order in every language
- forgetting worst-case behavior exists
- using mutable or unstable objects as keys without care
- sorting first when hashing gives linear-time logic

## Practice prompts

- longest consecutive sequence
- first non-repeating character
- subarray sum equals `k`
- valid anagram

## Quick revision

- hashing is a fast lookup strategy
- collisions are normal and must be handled
- the main interview skill is recognizing when hashing removes repeated scanning
