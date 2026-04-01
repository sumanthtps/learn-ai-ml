---
title: Number Systems
sidebar_position: 1
description: Complete notes on decimal, binary, octal, hexadecimal, signed representation, and numeric storage.
---

# Number Systems

Number systems explain how values are represented. This matters because computers do not "understand" decimal digits the way humans do. They store bit patterns and interpret them according to rules.

## Introduction

Humans are comfortable with base `10` because we grow up counting that way. Computers, however, are built from tiny physical switches that are best modeled as `on` and `off`, which maps naturally to base `2`. That is why number systems are not just a math topic in programming. They are the bridge between human-readable values and machine-readable representation.

If this topic feels abstract at first, think of it this way:

- decimal is how we usually write values
- binary is how computers usually store values
- hexadecimal is how engineers often inspect binary values without writing extremely long strings of `0`s and `1`s

## Visual intuition

![Binary positional value diagram](https://commons.wikimedia.org/wiki/Special:Redirect/file/Value_of_digits_in_the_Binary_numeral_system.svg)

Image source: [Wikimedia Commons - Value of digits in the Binary numeral system](https://commons.wikimedia.org/wiki/File:Value_of_digits_in_the_Binary_numeral_system.svg)

## Why this topic matters

- helps you understand binary and memory
- makes bit manipulation much easier
- prevents mistakes with overflow and signed numbers
- builds intuition for low-level systems and performance

## The four common bases

- Decimal: base `10`
- Binary: base `2`
- Octal: base `8`
- Hexadecimal: base `16`

Positional value means each digit contributes:

```text
digit * base^position
```

Example in decimal:

```text
572 = 5*10^2 + 7*10^1 + 2*10^0
```

Example in binary:

```text
1011 = 1*2^3 + 0*2^2 + 1*2^1 + 1*2^0 = 11
```

## Mental model

Every number system answers the same question:

`How much does each position contribute?`

In base `10`, moving one place left multiplies contribution by `10`.
In base `2`, moving one place left multiplies contribution by `2`.

That is why:

```text
1011₂
```

means:

- `1` group of `8`
- `0` groups of `4`
- `1` group of `2`
- `1` group of `1`

Once this positional idea clicks, conversions become much easier.

## Decimal to binary

Repeatedly divide by `2` and record remainders.

Example for `13`:

```text
13 / 2 = 6 remainder 1
6 / 2  = 3 remainder 0
3 / 2  = 1 remainder 1
1 / 2  = 0 remainder 1
```

Read bottom to top:

```text
13 = 1101₂
```

## Binary to decimal

Expand with powers of `2`.

```text
1101₂ = 1*2^3 + 1*2^2 + 0*2^1 + 1*2^0
      = 8 + 4 + 0 + 1
      = 13
```

## Octal and hexadecimal

These are compact ways to write binary.

- `1` octal digit = `3` bits
- `1` hexadecimal digit = `4` bits

Example:

```text
1111 1010₂ = FA₁₆
```

## Bits, bytes, and memory units

- `1 bit` stores `0` or `1`
- `8 bits = 1 byte`
- `1024 bytes = 1 KB`
- `1024 KB = 1 MB`

## Signed and unsigned integers

For `n` bits:

- unsigned range is `0` to `2^n - 1`
- signed range in two's complement is `-2^(n-1)` to `2^(n-1) - 1`

For `8` bits:

- unsigned: `0` to `255`
- signed: `-128` to `127`

## Two's complement

Two's complement is how negative integers are commonly represented.

To represent `-5` in `8` bits:

1. Write `5` in binary: `00000101`
2. Invert bits: `11111010`
3. Add `1`: `11111011`

So `-5` is:

```text
11111011
```

Why two's complement is useful:

- addition and subtraction use the same circuitry
- there is only one representation of zero

## Overflow and underflow

Overflow happens when the result exceeds the representable range.

Example with signed 8-bit integers:

```text
127 + 1 = -128
```

That looks strange mathematically, but it is expected in fixed-width storage.

## Floating-point numbers

Integers are exact inside their range. Floating-point numbers are approximate.

This is one of the first places where many beginners realize that "the value I typed" and "the value the machine stored" are not always identical. Floating-point representation is designed for a huge range of values, not perfect decimal precision.

That is why:

```python
print(0.1 + 0.2)
```

may produce:

```text
0.30000000000000004
```

This happens because many decimal fractions do not have exact finite binary representations.

## Worked example

Convert `45` to binary and hexadecimal.

Binary:

```text
45 / 2 = 22 r1
22 / 2 = 11 r0
11 / 2 = 5  r1
5 / 2  = 2  r1
2 / 2  = 1  r0
1 / 2  = 0  r1
```

So:

```text
45 = 101101₂
```

Now group into 4 bits:

```text
0010 1101 = 2D₁₆
```

Why grouping works:

- hexadecimal is base `16`
- `16 = 2^4`
- so each hex digit exactly matches `4` binary digits

That is why engineers often inspect machine values in hex instead of decimal or raw binary.

## Code example

```python
def decimal_to_binary(n: int) -> str:
    if n == 0:
        return "0"
    bits = []
    while n > 0:
        bits.append(str(n % 2))
        n //= 2
    return "".join(reversed(bits))


def binary_to_decimal(bits: str) -> int:
    value = 0
    for ch in bits:
        value = value * 2 + int(ch)
    return value


print(decimal_to_binary(45))   # 101101
print(binary_to_decimal("101101"))  # 45
```

### Explanation

- `decimal_to_binary` repeatedly collects remainders
- `binary_to_decimal` processes digits left to right like Horner's rule

## Common mistakes

- forgetting to read remainders bottom to top
- confusing signed and unsigned range
- assuming floating-point values are exact
- ignoring bit width while discussing overflow
- treating binary as a special case instead of "just another positional system"

## Interview questions to expect

- Convert a value between bases.
- Explain two's complement.
- What range does a 32-bit signed integer support.
- Why does floating-point arithmetic sometimes look incorrect.

## Practice prompts

- Convert `255` to binary and hexadecimal.
- Represent `-18` in two's complement using `8` bits.
- Explain why `2^10` is close to `10^3`.

## Quick revision

- Binary is base `2`
- Hex maps cleanly to binary in groups of `4`
- Two's complement is the standard negative integer representation
- Overflow is about limited storage width, not bad arithmetic
