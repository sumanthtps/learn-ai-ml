---
id: project-mlops-pipeline-problem
title: "Project 3 Problem · MLOps Pipeline Management API"
sidebar_label: "📝 P3 · MLOps Problem"
sidebar_position: 28
tags: [project, mlops, problem-statement, platform, model-registry, advanced]
---

# Project 3 Problem · MLOps Pipeline Management API

> **Problem Statement** — Build the control-plane backend for an internal MLOps platform. This statement defines the real-world scope without prescribing the exact architecture or answer.

---

## Business Context

A company has multiple ML teams training and deploying models, but the operational workflow is chaotic:

- no central model registry
- unclear deployment history
- weak rollback process
- no safe promotion workflow
- no standardized experiment tracking API

The company needs an internal backend platform that manages ML lifecycle operations.

---

## Product Goal

Build a FastAPI application that acts as the operational backend for an ML platform.

The API should make it possible to:

- register experiments
- track model versions
- promote or retire models
- support staged release workflows
- expose operational status and history

---

## Core Functional Requirements

Your system should support all of the following:

1. record training or experiment runs
2. register new model artifacts and metadata
3. list and inspect model versions
4. promote a model to staging or production
5. support rollback to a previous production version
6. expose model deployment history
7. protect sensitive actions with strong authorization

---

## Domain Concepts the API Should Model

- experiments
- model versions
- deployment stages
- active production version
- rollback target
- optional A/B or canary state

---

## Non-Functional Requirements

The platform should feel operationally safe:

- auditable
- permissioned
- traceable
- explicit in state transitions
- observable

---

## Technical Expectations

Your project should make use of concepts such as:

- FastAPI route design
- persistent storage
- auth and RBAC
- background jobs where appropriate
- monitoring/logging
- good API modeling

---

## API Surface to Define

At minimum:

- experiment creation/list endpoint
- model registration endpoint
- model promotion endpoint
- rollback endpoint
- health/admin endpoints

Optional but strong:

- canary rollout controls
- drift signal or alert endpoint
- deployment metrics endpoint

---

## Deliverables

- functioning FastAPI backend
- clear resource model
- robust role-based permissions
- documentation with examples
- containerization support

---

## Evaluation Criteria

- quality of domain modeling
- safety of promotion and rollback logic
- operational realism
- clarity of API contracts
- maintainability of code structure

---

## Important Constraint

Do not reduce this project to a shallow CRUD app with `status` fields only. The value is in lifecycle management and operational safety, not just storing rows.
