---
title: OOP
sidebar_position: 8
description: Complete notes on object-oriented programming, classes, encapsulation, inheritance, and composition.
---

# OOP

Object-oriented programming organizes software around objects that hold state and behavior.

## Why OOP matters

- useful for modeling domains
- common in application and backend code
- important for low-level design interviews

## Core pillars

- encapsulation
- abstraction
- inheritance
- polymorphism

## Class and object

```python
class BankAccount:
    def __init__(self, owner: str, balance: float = 0.0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float) -> None:
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount


account = BankAccount("Sumanth", 1000)
account.deposit(250)
print(account.balance)
```

## Encapsulation

Encapsulation means keeping related data and behavior together and limiting direct unsafe access.

Benefit:

- easier invariants
- fewer accidental bugs

## Abstraction

Abstraction means exposing only what the user of a class needs.

Example:

- users call `withdraw`
- they do not need to know internal bookkeeping details

## Inheritance

Inheritance allows one class to derive behavior from another.

```python
class Animal:
    def speak(self) -> str:
        return "..."


class Dog(Animal):
    def speak(self) -> str:
        return "Woof"
```

## Polymorphism

Different objects can respond to the same method in different ways.

```python
def make_sound(animal: Animal) -> None:
    print(animal.speak())
```

## Composition vs inheritance

Composition means building objects out of smaller objects instead of extending large parent hierarchies.

Prefer composition when:

- behavior should be mixed flexibly
- inheritance creates tight coupling
- "is-a" relationship is weak

## Example with composition

```python
class Engine:
    def start(self) -> str:
        return "Engine started"


class Car:
    def __init__(self):
        self.engine = Engine()

    def drive(self) -> str:
        return self.engine.start() + " and car moves"
```

## Why OOP can go wrong

- too many tiny classes
- deep inheritance chains
- mixing unrelated responsibilities
- exposing too much internal state

## Interview use

You may be asked to design:

- parking lot
- library system
- elevator system
- payment workflow

The goal is usually:

- identify entities
- define responsibilities
- show extensibility

## Practice prompts

- model a library with books and members
- model a shopping cart and pricing rule
- model an employee hierarchy carefully without overusing inheritance

## Quick revision

- objects combine state and behavior
- composition is often safer than deep inheritance
- good OOP is about changeability and clarity, not just creating classes
