---
title: Monoliths and Services
sidebar_position: 6
description: Monoliths, modular monoliths, microservices, and decomposition tradeoffs.
---

# Monoliths and Services

People often rush to microservices, but a monolith is frequently the right starting point.

## Monolith

One deployable unit containing multiple capabilities.

Benefits:

- simple deployment
- easy local development
- easy debugging
- fewer distributed-system failures

Costs:

- tighter coupling over time
- risky deployments as size grows
- harder independent scaling

## Modular monolith

A monolith with strong internal boundaries.

This is often a very healthy architecture.

## Microservices

Separate deployable services around business capabilities.

Benefits:

- independent scaling
- team autonomy
- independent release cycles

Costs:

- network failures
- distributed tracing complexity
- data consistency complexity
- deployment and ops overhead

## Rule of thumb

Start simple. Split only when:

- scale demands it
- team boundaries demand it
- release independence matters

## Example decomposition

From monolith:

- auth module
- orders module
- payments module

To services only if needed:

- auth service
- orders service
- payments service

## Common mistakes

- choosing microservices for resume value
- splitting before domain boundaries are understood
- creating synchronous dependency chains everywhere

## Quick revision

- monolith is not bad architecture
- modular monolith is often a strong default
- microservices solve some problems but create many new ones
