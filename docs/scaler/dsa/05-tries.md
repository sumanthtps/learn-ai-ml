---
title: Tries
sidebar_position: 5
description: Trie data structure, prefix search, implementations, and use cases.
---

# Tries

A trie stores strings by following character paths from a root node. It is ideal for prefix-based operations.

## Where tries are useful

- autocomplete
- dictionary lookup
- prefix counts
- word search
- XOR maximization with bitwise tries

## Basic structure

Each node stores:

- children
- word-end marker

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True
```

### In-depth explanation

This implementation separates the trie into:

- `TrieNode`, which stores local node state
- `Trie`, which exposes operations on the whole structure

For `TrieNode`:

- `children` maps a character to the next node
- `is_end` tells us whether a complete word ends here

Why `is_end` matters:

- without it, `"app"` and `"apple"` would be indistinguishable as full words

`insert` walkthrough:

- start at root
- for each character, move to that child
- if the child does not exist, create it
- after last character, mark `is_end = True`

`search` walkthrough:

- follow characters one by one
- if any character path is missing, word is absent
- after consuming all characters, return `node.is_end`
- this prevents prefixes from being mistaken for full words

`starts_with` walkthrough:

- same traversal logic as `search`
- but it only checks whether the prefix path exists
- it does not require `is_end = True`

Complexity:

- insert: `O(L)`
- search: `O(L)`
- prefix check: `O(L)`

where `L` is string length.

## Why a trie helps

If you have many prefix queries, a trie avoids scanning every word.

## Tradeoffs

Benefits:

- fast prefix search
- natural hierarchical representation

Costs:

- higher memory usage
- more implementation overhead than hashing

## Advanced ideas

- compressed trie
- radix tree
- bitwise trie

## Common mistakes

- forgetting end-of-word marker
- forcing a trie where hashing is enough
- ignoring memory cost

## Practice prompts

- implement autocomplete suggestions
- count words with a given prefix
- maximum XOR of two numbers

## Quick revision

- trie is for prefix-heavy workloads
- each level usually corresponds to one character or bit
- not every string problem needs a trie
