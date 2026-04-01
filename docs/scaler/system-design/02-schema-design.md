---
title: Schema Design
sidebar_position: 2
description: Schema design fundamentals, normalization, indexing, and tradeoffs.
---

# Schema Design

Schema design defines how data is structured in storage systems. It strongly affects performance, correctness, and maintainability.

## Core concepts

- entities
- attributes
- primary keys
- foreign keys
- relationships
- indexes

## Example: task management app

Entities:

- users
- projects
- tasks
- comments

Example relational schema:

```sql
CREATE TABLE users (
  id BIGSERIAL PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL
);

CREATE TABLE tasks (
  id BIGSERIAL PRIMARY KEY,
  project_id BIGINT NOT NULL,
  assignee_id BIGINT,
  title TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

## Normalization

Normalization reduces redundancy.

Benefits:

- less repeated data
- easier consistency
- cleaner updates

Costs:

- more joins
- sometimes slower reads

## Denormalization

Useful when read performance matters more than perfectly normalized storage.

Examples:

- storing aggregate counts
- materialized views
- duplicated display-ready fields

## Indexing

Indexes speed up reads but add write overhead.

If you often query:

```sql
SELECT * FROM tasks WHERE assignee_id = 42 AND status = 'OPEN';
```

then an index on `(assignee_id, status)` may help.

## Choosing relational vs document

Relational is strong when:

- data is structured
- relationships matter
- transactions matter

Document storage is strong when:

- schema is flexible
- nested data is natural
- join-heavy modeling is unnecessary

## Schema evolution

Real systems change. Plan for:

- migrations
- backward compatibility
- nullable transitional columns

## Common mistakes

- designing tables without thinking about query patterns
- adding indexes blindly
- using denormalization without a consistency strategy

## Quick revision

- schema is not just about storage, it is about access patterns
- keys and indexes are central
- normalize for correctness, denormalize carefully for performance
