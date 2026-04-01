---
title: Operating Systems
sidebar_position: 9
description: "Operating systems notes for interviews: processes, threads, memory, synchronization, and scheduling."
---

# Operating Systems

Operating systems manage hardware resources and provide abstractions that let programs run safely and efficiently.

## Why OS matters in this curriculum

- interviewers ask OS frequently
- concurrency and memory behavior affect real software
- backend and systems work depend on these concepts

## Process vs thread

Process:

- running program with its own address space

Thread:

- execution unit inside a process

Multiple threads in the same process share memory, which enables communication but also creates synchronization risk.

## Memory regions

- code segment
- stack
- heap

Stack:

- function calls
- local variables
- usually automatically managed

Heap:

- dynamic allocation
- longer-lived objects

## Virtual memory

Programs see a virtual address space, not raw physical memory directly.

Benefits:

- isolation
- abstraction
- efficient memory use

## Paging

Memory is divided into fixed-size pages. The OS maps virtual pages to physical frames.

## Scheduling

The CPU scheduler decides which process or thread runs next.

Common goals:

- fairness
- responsiveness
- throughput

## Synchronization

Important terms:

- mutex
- semaphore
- critical section
- race condition

Race condition:

- outcome depends on timing of concurrent execution

## Deadlocks

Deadlock can happen when processes wait forever for resources held by one another.

Four classic conditions:

- mutual exclusion
- hold and wait
- no preemption
- circular wait

## Simple thread-safe example

```python
from threading import Lock

counter = 0
lock = Lock()


def increment():
    global counter
    with lock:
        counter += 1
```

### Explanation

- `with lock` ensures only one thread updates `counter` at a time
- without the lock, a race condition may appear

### In-depth code walkthrough

This is a minimal example of protecting shared state.

Why `counter` is risky:

- multiple threads may read the same old value at the same time
- both may compute a new value
- one update may overwrite the other

That means `counter += 1` is not "magically atomic" at the conceptual level.

Role of each line:

- `Lock()` creates a mutual-exclusion primitive
- `global counter` tells Python we are modifying the module-level variable
- `with lock:` acquires the lock before entering the block and releases it automatically when leaving
- `counter += 1` is now protected so only one thread can perform that critical section at a time

Why the context manager style is good:

- safer than manual acquire and release
- avoids forgetting to release the lock when errors happen

Complexity idea:

- lock usage improves correctness, not speed
- synchronization can reduce parallel throughput, but it prevents corrupted state

## Context switching

Switching CPU execution from one thread or process to another has overhead.

Too many threads can hurt performance because switching is not free.

## Common interview questions

- difference between process and thread
- stack vs heap
- mutex vs semaphore
- what is deadlock
- what is virtual memory

## Quick revision

- OS provides abstraction, isolation, and scheduling
- threads share memory, processes usually do not
- synchronization prevents race conditions
- deadlocks are about resource wait cycles
