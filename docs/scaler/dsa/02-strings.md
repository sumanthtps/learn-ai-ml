---
title: Strings
sidebar_position: 2
description: String problem-solving patterns, matching, parsing, and code examples.
---

# Strings

Strings are sequences, but they often come with extra structure like characters, frequency, substrings, and lexicographic order.

## Common patterns

- frequency counting
- sliding window
- palindrome expansion
- parsing
- hashing
- pattern matching

## Frequency map

```python
def is_anagram(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    freq = {}
    for ch in a:
        freq[ch] = freq.get(ch, 0) + 1
    for ch in b:
        if ch not in freq:
            return False
        freq[ch] -= 1
        if freq[ch] == 0:
            del freq[ch]
    return not freq
```

### In-depth explanation

Two strings are anagrams if they contain exactly the same characters with exactly the same counts.

How this code works:

- if lengths differ, they cannot be anagrams
- the first loop builds a frequency table for string `a`
- the second loop consumes characters from `b`
- if a character from `b` is missing in the table, the strings differ
- after decrementing, counts that reach zero are removed to keep the map clean
- `return not freq` means all counts matched perfectly

Why deleting zero-count entries is useful:

- it makes the final emptiness check simple
- it avoids clutter from zero values

Example:

- `a = "listen"`
- `b = "silent"`

After processing `a`, frequencies count each letter once.
After processing `b`, every count returns to zero, so the map becomes empty.

Complexity:

- time: `O(n)`
- space: `O(k)` where `k` is number of distinct characters

## Sliding window

Longest substring without repeating characters:

```python
def longest_unique_substring(s: str) -> int:
    last_seen = {}
    left = 0
    best = 0
    for right, ch in enumerate(s):
        if ch in last_seen and last_seen[ch] >= left:
            left = last_seen[ch] + 1
        last_seen[ch] = right
        best = max(best, right - left + 1)
    return best
```

### In-depth explanation

This sliding-window algorithm maintains a window with no repeated characters.

Meaning of variables:

- `left` = start index of the current valid window
- `right` = end index as we scan forward
- `last_seen[ch]` = most recent index where character `ch` appeared
- `best` = longest valid window length seen so far

Why the `if` condition matters:

- if a repeated character was seen before `left`, it does not affect the current window
- only repeats inside the current window matter

So:

```python
if ch in last_seen and last_seen[ch] >= left:
```

means "this character repeats inside the active window."

When that happens:

- move `left` just past the previous occurrence

This is efficient because `left` only moves forward.

Example with `"abba"`:

- `a` window is `"a"`
- `b` window is `"ab"`
- next `b` repeats inside window, so move `left` to after the old `b`
- continue from there

Complexity:

- time: `O(n)`
- space: `O(k)`

## Palindrome expansion

```python
def longest_palindrome(s: str) -> str:
    def expand(l: int, r: int) -> tuple[int, int]:
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return l + 1, r

    best = (0, 1)
    for i in range(len(s)):
        best = max(best, expand(i, i), expand(i, i + 1), key=lambda p: p[1] - p[0])
    return s[best[0]:best[1]]
```

### In-depth explanation

This solution uses the "expand around center" technique.

Why centers matter:

- every palindrome has a center
- odd-length palindromes have one center character
- even-length palindromes have a center gap between two characters

The helper `expand(l, r)`:

- starts from a proposed center
- moves outward while both ends match
- returns the maximal valid palindrome boundaries

The returned pair uses Python slice convention:

- start index inclusive
- end index exclusive

Inside the main loop:

- `expand(i, i)` checks odd-length palindrome centered at `i`
- `expand(i, i + 1)` checks even-length palindrome between `i` and `i + 1`
- `max(..., key=lambda p: p[1] - p[0])` keeps the longer one

Example with `"babad"`:

- center at `a` gives `"bab"` or `"aba"` depending on expansion

Complexity:

- time: `O(n^2)` in worst case
- space: `O(1)` extra

## Parsing

String problems often need disciplined parsing.

Example:

```python
def count_words(s: str) -> int:
    return len([word for word in s.strip().split() if word])
```

### In-depth explanation

This is a compact parsing example.

Step by step:

- `s.strip()` removes leading and trailing whitespace
- `.split()` breaks the string on whitespace
- `if word` filters out empty fragments if present
- `len(...)` counts resulting words

Why parsing matters in string problems:

- many bugs happen not in the main logic but in whitespace, punctuation, or malformed-input handling

Complexity:

- time: `O(n)`
- space: proportional to the number of split pieces

## Pattern matching

### Naive

Check every starting position.

### KMP

Uses prefix information to avoid re-checking characters.

You should know the intuition even if you do not memorize every detail immediately:

- matched prefix information can be reused
- mismatch does not always require restarting from scratch

## Rolling hash

Useful for substring matching and duplicate substring problems.

## Common mistakes

- confusing substring with subsequence
- forgetting string immutability cost in some languages
- using `O(n^2)` concatenation patterns

## Practice prompts

- minimum window substring
- group anagrams
- valid palindrome after one deletion
- string to integer parser

## Quick revision

- sliding window is a major string pattern
- substrings are contiguous, subsequences are not
- parsing questions reward careful edge-case handling
