---
title: SOLID
sidebar_position: 3
description: SOLID principles explained with practical examples.
---

# SOLID

SOLID is a set of design principles that help object-oriented code stay easier to extend and maintain.

## S: Single Responsibility Principle

A class should have one reason to change.

Bad:

- one class validates orders, sends emails, and writes invoices

Better:

- order validation service
- invoice service
- notification service

## O: Open/Closed Principle

Software should be open for extension, closed for modification.

Idea:

- add new behavior through extension instead of editing fragile existing logic

## L: Liskov Substitution Principle

Subtypes should behave correctly wherever the base type is expected.

If a subclass breaks assumptions, inheritance is probably wrong.

## I: Interface Segregation Principle

Clients should not depend on methods they do not use.

Prefer smaller focused interfaces over giant "god interfaces."

## D: Dependency Inversion Principle

High-level modules should depend on abstractions, not directly on concrete low-level details.

## Example

```python
class EmailSender:
    def send(self, message: str) -> None:
        print("Sending email:", message)


class NotificationService:
    def __init__(self, sender):
        self.sender = sender

    def notify(self, message: str) -> None:
        self.sender.send(message)
```

Here `NotificationService` depends on a sender contract, not on one hardcoded implementation style.

## Common misuse

- creating too many interfaces too early
- overengineering tiny programs
- applying SOLID mechanically instead of thoughtfully

## Quick revision

- SOLID is about changeability
- use it to reduce coupling and responsibility overload
- do not let principles make simple code harder than necessary
