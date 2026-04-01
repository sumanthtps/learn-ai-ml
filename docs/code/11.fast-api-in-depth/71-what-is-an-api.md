---
title: "What is an API?"
sidebar_position: 71
---

## Source

- Playlist: FastAPI for Machine Learning | CampusX
- Original video title: 1. What is an API? | Introduction to APIs | FAST API for Machine Learning | CampusX
- YouTube video ID: `WJKsPchji0Q`
- Transcript pages in the uploaded PDF: 2-27

## Why this lesson matters

This lesson builds the core mental model for the entire playlist. If you do not clearly understand what an API is, FastAPI will look like a collection of decorators and URLs. Once the API idea is clear, every later topic becomes easier:

- why clients send requests
- why servers return structured responses
- why JSON is used so often
- why machine learning models are usually served behind APIs
- why frontend, backend, mobile apps, and third-party tools can all talk to the same business logic

## What the transcript covers

The transcript explains:

- why FastAPI matters for AI and machine learning engineers
- what an API is in simple language
- the connection between frontend and backend
- how monolithic systems differ from API-based systems
- why mobile apps and third-party integrations made APIs more important
- why HTTP and JSON became common in web systems
- how the same API idea applies to machine learning model serving

## API: the precise definition

An API, or Application Programming Interface, is a defined way for one software system to ask another software system to do something.

The most important words in that definition are:

### Interface
An interface is the visible surface through which interaction happens. In the web world, that usually means endpoints such as:

- `GET /patients`
- `POST /predict`
- `DELETE /patients/12`

### Defined
An API is not random communication. It follows agreed rules:

- what URL to call
- which HTTP method to use
- what input format is accepted
- what output format is returned
- what errors look like

### Software to software communication
Humans may trigger the action through a UI, but the real interaction usually happens between systems:

- frontend to backend
- mobile app to backend
- backend to another backend
- analytics service to internal platform
- client app to machine learning inference service

## The simplest mental model

The transcript uses a waiter analogy. That analogy is useful because it captures the idea of controlled communication.

### Restaurant analogy

- Customer = client
- Waiter = API layer
- Kitchen = backend logic
- Food = response

The customer does not directly enter the kitchen and start using the stove. The customer places a request through a controlled interface. The waiter carries the request, the kitchen processes it, and the waiter brings the result back.

That is exactly how most web APIs work.

## Request-response flow in a real system

A typical API flow looks like this:

1. A client sends a request.
2. The server receives the request.
3. Input is validated.
4. Business logic runs.
5. The backend may talk to a database, a model, or another service.
6. The result is formatted.
7. A response is returned.

### Example: course search website

Suppose a user searches for `AI agents` on an edtech platform.

Flow:

1. User types `AI agents` in the search bar.
2. Frontend sends a request to the server.
3. API receives the request.
4. Backend searches the database.
5. Matching results are prepared.
6. Response is returned as JSON.
7. Frontend renders the list to the user.

The user only sees a search box and results. The API is the controlled path connecting the visible system to the hidden system.

## API is not the database

A beginner mistake is to think that the API is just another name for the data source. That is not correct.

The API is the contract layer.

The data or business logic behind it could be:

- a relational database
- a NoSQL database
- a machine learning model
- a payment gateway
- another internal microservice
- file storage
- a rules engine

This distinction matters because the API must stay stable even if the internal implementation changes.

### Example
Today your endpoint `/predict` may use a scikit-learn model. Tomorrow it may use an XGBoost model or a neural network. The client ideally should not need to change, as long as the API contract remains the same.

## Monolith vs API-first architecture

The transcript discusses how older systems were often tightly coupled.

### Monolith
In a monolithic system, frontend rendering, business logic, and data access are often bundled together in one application.

This can work for small products, but problems appear when:

- a mobile app needs the same functionality
- a third party wants controlled access
- different teams need independent release cycles
- scaling requirements differ between UI and backend

### API-first or API-based architecture
In API-based design, the backend capability is exposed through well-defined endpoints. Different clients can use the same backend contract.

One backend can power:

- a website
- an Android app
- an iOS app
- an admin dashboard
- partner integrations
- ML experimentation tools

This is one reason APIs became central to modern engineering.

## Why mobile apps increased API importance

When companies only had websites, they could sometimes keep backend logic tightly attached to server-rendered pages.

When mobile apps arrived, that approach started to break down.

Now the same business capability had to be consumed by:

- web frontend
- Android app
- iPhone app

Instead of rewriting the same business logic multiple times, companies exposed that logic through APIs.

That made the API the reusable middle layer.

## Why HTTP matters

On the web, the most common protocol for API communication is HTTP.

HTTP gives us a standard way to describe:

- the target resource using URLs
- the operation using methods such as GET or POST
- headers for metadata
- request bodies for input data
- status codes for result signaling

### Common HTTP methods

- `GET` - fetch data
- `POST` - create or trigger processing
- `PUT` - replace or update
- `PATCH` - partial update
- `DELETE` - remove a resource

You will study these in more detail later, but from this first lesson, remember that HTTP gives the conversation structure.

## Why JSON matters

JSON became the default exchange format because it is:

- lightweight
- human-readable
- easy to parse
- language-independent

A frontend in JavaScript, a mobile app in Swift, and a backend in Python can all exchange JSON.

### Example JSON response

```json
{
  "patient_id": 3,
  "name": "Rahul",
  "city": "Bengaluru",
  "bmi": 24.1
}
```

Without a common format, each client would need custom translation logic.

## API from the machine learning perspective

This is where the lesson becomes important for ML engineers.

When you build a model in Jupyter, you usually test it locally with Python variables. But real users cannot interact with your notebook directly.

You need a serving layer.

That serving layer is often an API.

### Traditional backend example

```text
request -> backend -> database -> response
```

### ML backend example

```text
request -> API -> preprocessing -> model inference -> postprocessing -> response
```

The database is no longer the only backend actor. The model becomes a computational component behind the API.

## Why ML models are usually served behind APIs

A machine learning model by itself is just an artifact plus some code.

To make it useful to the outside world, you need:

- a way to send inputs
- a way to validate them
- a way to run inference safely
- a way to return outputs predictably

An API solves that.

### Example: insurance premium prediction

Input from client:

```json
{
  "age": 35,
  "bmi": 27.4,
  "smoker": true,
  "city": "Mumbai"
}
```

The API can:

1. validate the fields
2. convert categorical values into model-ready form
3. call the model
4. map numeric output into a human-friendly result
5. return structured JSON

Response:

```json
{
  "predicted_premium": 18432.55,
  "risk_band": "high"
}
```

## Internal, partner, and public APIs

The transcript focuses on the general idea of software-to-software communication. In day-to-day engineering, you also need to know that not all APIs are exposed in the same way.

### Internal API
Used only within the company.

Example:
- billing service calling user service

### Partner API
Used by approved external organizations.

Example:
- payment provider integration

### Public API
Available to outside developers under controlled rules.

Example:
- weather API or map API

This matters because authentication, rate limits, documentation quality, and change management differ by API type.

## API contract: the most important production idea

A good API is not only code. It is a contract.

That contract usually includes:

- endpoint names
- accepted input fields
- required vs optional data
- return structure
- status codes
- error format
- versioning expectations

Clients build against that contract. Once other systems depend on it, changing it carelessly can break production.

### Bad mindset
"I changed the response field name, but my code still works locally."

### Correct mindset
"If any client depends on the old field name, this is a breaking change."

## Worked examples

### Example 1: weather app

Client request:

```text
GET /weather?city=Bengaluru
```

API responsibility:

- receive city name
- validate it
- call weather provider or internal service
- normalize the output
- return consistent JSON

### Example 2: hospital patient record lookup

Client request:

```text
GET /patients/42
```

API responsibility:

- identify patient 42
- fetch from database
- enforce permission rules
- return patient data or a 404 error

### Example 3: ML sentiment analysis

Client request:

```json
{
  "text": "The service was excellent."
}
```

API responsibility:

- validate text exists
- preprocess text
- run model
- return sentiment label and score

## Common mistakes beginners make

### 1. Thinking API means only web frontend calls
APIs are broader than that. Backend-to-backend communication is extremely common.

### 2. Thinking API is just a URL
A URL is only part of the contract. Method, body, response shape, and status codes also matter.

### 3. Exposing internal implementation details
Clients should depend on stable API behavior, not on how your internals are coded.

### 4. Ignoring errors
A real API must clearly define what happens when input is invalid, data is missing, or the backend fails.

### 5. Treating model serving like notebook execution
Notebook code is usually not safe for uncontrolled external input. APIs add structure and safety.

## Daily engineering additions beyond the transcript

The transcript gives the motivation and the architecture. In real projects, you must also think about:

- authentication and authorization
- request validation
- response schema design
- logging and monitoring
- rate limiting
- retries and timeouts
- versioning
- backward compatibility
- caching where appropriate

These are the operational concerns that separate a demo API from a production API.

## Important Q&A

### 1. What is the simplest definition of an API?
An API is a defined interface that allows one software system to request work from another software system.

### 2. Why are APIs so important in modern software?
Because they allow different clients and services to reuse the same backend capability without duplicating logic.

### 3. Why is an API useful for machine learning?
Because a model becomes usable to websites, apps, and other systems only when it is exposed through a stable serving interface.

### 4. Why do we need standard formats like JSON?
Because clients may be built in different languages and still need a common representation for data exchange.

### 5. Why should clients not directly access the database?
Because the API layer provides validation, permissions, business rules, transformation, and a stable contract.

## Quick revision

- API means controlled software-to-software communication.
- It is a contract, not just a URL.
- HTTP structures requests and responses.
- JSON is a common data exchange format.
- APIs became more important when multiple clients needed the same backend.
- ML models are commonly served behind APIs.
- A production API needs validation, errors, security, and versioning, not just working code.
