---
id: project-car-price-prediction-problem
title: "Project 4 Problem · Car Price Prediction ML API"
sidebar_label: "📝 P4 · Car Price Problem"
sidebar_position: 30
tags: [project, car-price, problem-statement, capstone, ml-api, advanced]
---

# Project 4 Problem · Car Price Prediction ML API

> **Capstone Problem Statement** — Build a complete car price prediction API with auth, caching, logging, monitoring, Docker, and deployment. This page defines the assignment in detail without providing the answer path.

---

## Business Context

A used-car marketplace wants an internal pricing API that helps:

- sales teams estimate listing prices
- operations teams validate suspiciously low or high listings
- partner apps query predicted market value in real time

The product team wants the service exposed as a reliable ML API rather than a notebook or offline script.

---

## Product Goal

Build a production-style FastAPI service that predicts used car prices from structured vehicle features and exposes the service with proper operational controls.

---

## Required Features

Your capstone must include all of the following:

1. A prediction endpoint that accepts vehicle features and returns estimated price.
2. Input validation with clear error responses.
3. Authentication on protected endpoints.
4. Caching for repeated or equivalent requests.
5. Structured logging.
6. Monitoring support.
7. Docker packaging.
8. Deployment suitable for a platform such as Render.

---

## Suggested Input Features

Your request schema should support features such as:

- car brand
- car model
- manufacturing year
- kilometers driven
- fuel type
- transmission
- owner count
- engine size or similar numeric vehicle attributes

You may extend the feature set, but your API should stay coherent and realistic.

---

## Required Output

Your response should contain:

- predicted price
- currency
- enough metadata to be useful operationally

Optional but strong:

- model version
- confidence band
- cache hit metadata

---

## Non-Functional Requirements

The capstone should aim for:

- safe request validation
- predictable response contract
- clear logging
- measurable performance
- deployability
- maintainable structure

---

## Monitoring Expectations

You do not need a giant enterprise observability stack, but your service should make it possible to answer:

- how many prediction requests are arriving
- what the latency looks like
- whether cache is helping
- whether errors are rising

---

## Authentication Expectations

There must be a distinction between:

- public or unauthenticated operational info, if any
- normal prediction access
- privileged actions such as model reload or admin inspection

You may choose API key or JWT depending on your design justification.

---

## Deployment Expectations

The project should be runnable:

- locally
- inside Docker
- on a simple cloud deployment target such as Render

Your documentation should explain how.

---

## Deliverables

- working FastAPI codebase
- trained or mocked ML artifact integrated into the API
- clear README
- example requests and responses
- Dockerfile
- deployment notes

---

## Evaluation Criteria

- realism of the ML-serving workflow
- quality of the API contract
- use of auth, caching, logging, and monitoring
- quality of packaging and deployment readiness
- documentation and developer experience

---

## Important Constraint

Do not treat the capstone as only "create `/predict` and return one float." The point is to show end-to-end engineering maturity around an ML service.

---

## Stretch Goals

If you want to push beyond the base assignment:

- batch prediction endpoint
- model version selection
- request tracing IDs
- rate limiting
- shadow testing for a new model
