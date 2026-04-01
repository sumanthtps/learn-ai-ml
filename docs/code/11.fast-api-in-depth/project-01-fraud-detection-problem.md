---
id: project-fraud-detection-problem
title: "Project 1 Problem · Real-Time Fraud Detection API"
sidebar_label: "📝 P1 · Fraud Problem"
sidebar_position: 24
tags: [project, fraud, problem-statement, api, ml, advanced]
---

# Project 1 Problem · Real-Time Fraud Detection API

> **Problem Statement** — Design and build a production-style fraud detection API for a fintech company. This page defines the challenge completely, but it intentionally does not provide the implementation path or architecture answer.

---

## Business Context

A fintech company processes card transactions in real time. Every swipe, tap, or online payment must be evaluated in milliseconds to decide whether the payment should be:

- approved
- flagged for review
- blocked immediately

The business wants an internal ML-powered API that other systems can call during transaction processing.

---

## Product Goal

Build a backend service that accepts transaction data, validates it, scores fraud risk, stores audit records, and exposes controlled access for operations and fraud analysts.

Your solution should feel like a realistic internal ML API, not a notebook wrapped in a route.

---

## Users of the System

- payment-processing service calling the prediction endpoint
- fraud analysts reviewing suspicious transactions
- admins managing operational controls
- background systems retraining or refreshing fraud models

---

## Core Functional Requirements

Your system must support all of the following:

1. Accept a transaction payload and return a fraud decision.
2. Log every scored transaction for audit and traceability.
3. Distinguish between low-risk, medium-risk, and high-risk outcomes.
4. Expose a way for fraud analysts to retrieve flagged transactions.
5. Protect privileged functionality with proper authentication and authorization.
6. Support a background workflow for retraining or model-refresh operations.
7. Provide operational endpoints for health and observability.

---

## Suggested Domain Fields

You may refine the schema, but the API should be capable of handling fields like:

- transaction ID
- user ID
- amount
- merchant category
- card present or not
- geo-location or region
- device fingerprint or channel
- timestamp
- historical risk signals or engineered features

---

## Functional Output Expectations

The prediction response should communicate:

- decision
- fraud score or risk score
- confidence or probability signal
- model version or scoring metadata if appropriate

Do not make the output vague. The caller should be able to act on it immediately.

---

## Non-Functional Requirements

Your implementation should target the following qualities:

- low latency
- safe validation
- good logging
- clear error handling
- secure access
- scalability under repeated traffic
- observability for production debugging

---

## Technical Requirements

This project should use concepts from the FastAPI in-depth module, including where appropriate:

- FastAPI routing and validation
- request and response schemas
- persistence layer
- ML model serving
- authentication and authorization
- caching
- background processing
- monitoring and metrics
- Docker packaging

You are free to choose exact design details, but the final system must clearly demonstrate these capabilities.

---

## API Surface to Define

At minimum, your project should include:

- one prediction endpoint
- one or more health or readiness endpoints
- one analyst-facing read endpoint
- one admin-only or operational endpoint

Optional but valuable:

- batch scoring
- model refresh endpoint
- websocket or event stream for flagged alerts

---

## Data and Model Expectations

You do not need a state-of-the-art fraud model. But your project must behave like a real ML system:

- load a trained model or scoring artifact
- preprocess inputs consistently
- produce deterministic API responses
- separate model logic from route logic

---

## Security Expectations

Your project should enforce access boundaries. Different actors should not have identical permissions.

Think about:

- API clients
- analysts
- admins

The project should make it clear which actions are public, authenticated, or restricted.

---

## Observability Expectations

At minimum, your project should make it possible to answer:

- how many requests are coming in
- how many are failing
- how long scoring takes
- which model version is serving traffic

---

## Deliverables

Your submission should include:

- running FastAPI application
- clear project structure
- README with setup instructions
- example request and response payloads
- evidence of validation and auth
- evidence of monitoring or logging
- Docker support

---

## Evaluation Criteria

You should judge the solution on:

- correctness of API behavior
- clarity of schemas and contracts
- realism of the ML-serving flow
- separation of concerns
- security awareness
- production readiness
- documentation quality

---

## Important Constraint

Do not turn this into a pure CRUD app with a fake `predict` route that just returns hardcoded labels. The project should feel like a real inference service with operational concerns.

---

## Extension Ideas

If you finish early, you can add:

- streaming analyst alerts
- cache invalidation on model refresh
- canary rollout for a new model version
- drift monitoring sketch

---

## What This Problem Is Testing

This project is testing whether you can combine:

- API engineering
- ML model integration
- system design thinking
- production concerns

into one coherent backend service.
