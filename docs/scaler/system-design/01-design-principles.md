---
title: Design Principles
sidebar_position: 1
description: Core software and system design principles with examples and tradeoffs.
---

# Design Principles

Design principles help you build systems that remain understandable and changeable.

## Core principles

- separation of concerns
- loose coupling
- high cohesion
- clear interfaces
- simplicity
- observability

## Separation of concerns

Each component should focus on one kind of responsibility.

Example:

- API layer handles HTTP
- service layer handles business logic
- repository layer handles persistence

## Loose coupling

Components should depend on stable contracts, not on hidden details.

Benefit:

- easier testing
- easier replacement
- less breakage from changes

## High cohesion

Code that belongs together should stay together.

Bad example:

- one class handles email sending, SQL querying, and payment logic

Good example:

- each module owns one clear area of behavior

## Design for failure

In real systems, dependencies fail.

So design should consider:

- retries
- timeouts
- fallbacks
- circuit breakers

## Design for observability

You should be able to answer:

- what failed
- where it failed
- how often it fails
- how long it takes

That means:

- logs
- metrics
- traces

## Code example: layered design

```python
class UserRepository:
    def __init__(self, db):
        self.db = db

    def get_by_id(self, user_id: int):
        return self.db.get(user_id)


class UserService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

    def fetch_user_profile(self, user_id: int):
        user = self.repo.get_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        return {"id": user["id"], "name": user["name"]}
```

## Common mistakes

- adding abstractions before real variation exists
- mixing transport, business logic, and storage concerns
- designing for every future possibility instead of current constraints

## Quick revision

- good design reduces future change cost
- modularity is about responsibility clarity
- simplicity beats unnecessary cleverness
