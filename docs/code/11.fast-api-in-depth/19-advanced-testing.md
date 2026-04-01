---
id: advanced-testing
title: "19 · Advanced Testing — pytest, Fixtures, Mocks, Load Tests"
sidebar_label: "19 · Advanced Testing"
sidebar_position: 19
tags: [testing, pytest, mocking, integration, load-testing, ci, intermediate]
---

# Advanced Testing — pytest, Fixtures, Mocks, Load Tests

> **Advanced Topic** — Building a complete, maintainable test suite for production ML APIs.

---

## The Testing Philosophy for ML APIs

Tests for an ML API have a unique challenge: the ML model itself is non-deterministic and slow. The solution is **test isolation**: replace the real model with a predictable mock so tests run in milliseconds and produce the same result every time.

```
Test pyramid for ML APIs:

              ┌──────────────┐
              │  Load Tests  │  ← Few, slow, real infrastructure (Locust)
            ┌─┴──────────────┴─┐
            │ Integration Tests │  ← Some, medium speed, real DB + mocked model
          ┌─┴──────────────────┴─┐
          │    Unit Tests         │  ← Many, fast, everything mocked
          └───────────────────────┘
```

**Unit tests**: Test a single function in isolation. The model is mocked. Zero database. Run in milliseconds.

**Integration tests**: Test entire API flows (HTTP request → validation → DB write → response). Use a real test database, mocked model. Run in seconds.

**Load tests**: Send thousands of requests to measure throughput and latency. Run on a test server.

---

## Setting Up the Test Environment

```bash
pip install pytest pytest-asyncio httpx pytest-cov aiosqlite
```

```ini title="pytest.ini"
[pytest]
# asyncio_mode=auto: all async test functions run automatically (no @pytest.mark.asyncio needed)
asyncio_mode = auto

# testpaths: where to look for tests
testpaths = tests

# addopts: default options for every pytest run
addopts =
    -v                      # verbose output (show each test name)
    --tb=short             # short traceback format
    --cov=app              # measure coverage for the 'app' directory
    --cov-report=term      # show coverage in terminal
    --cov-report=html      # also generate HTML report

filterwarnings = ignore::DeprecationWarning
```

---

## The `conftest.py` — Your Test Foundation

```python title="tests/conftest.py"
"""
pytest automatically discovers and loads conftest.py files.
Fixtures defined here are available to ALL test files.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np

from app.main import app, model_store


@pytest.fixture(scope="session")
def client():
    """
    TestClient wraps your FastAPI app for testing.
    
    scope="session" means: create ONE client for all tests in the session.
    This is efficient — no need to create a new client for each test.
    
    TestClient handles the full request cycle:
    - Parses your route decorators
    - Runs your Pydantic validators
    - Executes your endpoint functions
    - Returns a response object
    
    No actual HTTP network is involved — it's all in-process.
    """
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_model():
    """
    autouse=True: this fixture runs for EVERY test automatically.
    
    We replace the real ML model with a MagicMock.
    
    Why mock? The real model:
    - Takes 200ms to load from disk (slow tests)
    - Returns different results each run (non-deterministic tests)
    - Must be trained first (complex setup)
    
    The mock:
    - Loads instantly
    - Returns exactly what we tell it to (predictable)
    - No training needed
    """
    mock = MagicMock()
    
    # Configure what the mock returns when called
    mock.predict.return_value = np.array(["medium"])
    mock.predict_proba.return_value = np.array([[0.1, 0.7, 0.2]])
    mock.classes_ = np.array(["high", "low", "medium"])
    
    # Temporarily replace the real model with our mock
    original_store = dict(model_store)
    model_store.update({
        "model": mock,
        "classes": ["high", "low", "medium"]
    })
    
    yield mock  # tests run here with the mock in place
    
    # After each test: restore original state
    model_store.clear()
    model_store.update(original_store)


@pytest.fixture
def valid_insurance_input():
    """Reusable valid input data for insurance prediction tests."""
    return {
        "age": 35,
        "sex": "male",
        "bmi": 27.9,
        "children": 2,
        "smoker": "no",
        "region": "southeast"
    }
```

---

## Unit Tests — Testing One Thing at a Time

```python title="tests/unit/test_schemas.py"
"""
Test Pydantic schemas directly without HTTP.
These run in microseconds — no HTTP overhead, no database.
"""
import pytest
from pydantic import ValidationError
from app.schemas.insurance import InsuranceInput


class TestInsuranceInputSchema:
    """Group related tests together in a class for organization."""
    
    VALID_DATA = {
        "age": 35, "sex": "male", "bmi": 27.9,
        "children": 2, "smoker": "no", "region": "southeast"
    }
    
    def test_valid_input_creates_model(self):
        """Happy path: valid data should create the model without error."""
        input_model = InsuranceInput(**self.VALID_DATA)
        assert input_model.age == 35
        assert input_model.sex == "male"
        assert input_model.bmi == 27.9
    
    def test_age_below_minimum_fails(self):
        """Age must be >= 18. Values below this should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            InsuranceInput(**{**self.VALID_DATA, "age": 5})
        
        # Verify the error message mentions "age" — not some other field
        errors = exc_info.value.errors()
        assert any(err["loc"][-1] == "age" for err in errors)
    
    def test_age_above_maximum_fails(self):
        """Age must be <= 100."""
        with pytest.raises(ValidationError):
            InsuranceInput(**{**self.VALID_DATA, "age": 150})
    
    def test_invalid_sex_fails(self):
        """sex must be 'male' or 'female' — nothing else."""
        with pytest.raises(ValidationError):
            InsuranceInput(**{**self.VALID_DATA, "sex": "unknown"})
    
    def test_invalid_smoker_fails(self):
        """smoker must be 'yes' or 'no' — not 'sometimes'."""
        with pytest.raises(ValidationError):
            InsuranceInput(**{**self.VALID_DATA, "smoker": "sometimes"})
    
    def test_invalid_region_fails(self):
        """region must be one of the four defined regions."""
        with pytest.raises(ValidationError):
            InsuranceInput(**{**self.VALID_DATA, "region": "india"})
    
    @pytest.mark.parametrize("bmi,should_fail", [
        (9.9, True),    # too low
        (10.0, False),  # minimum valid
        (65.0, False),  # maximum valid
        (65.1, True),   # too high
    ])
    def test_bmi_boundary_values(self, bmi, should_fail):
        """
        Parameterized test: runs this same test with multiple inputs.
        This tests boundary values (edge cases at the limits of valid range).
        """
        if should_fail:
            with pytest.raises(ValidationError):
                InsuranceInput(**{**self.VALID_DATA, "bmi": bmi})
        else:
            model = InsuranceInput(**{**self.VALID_DATA, "bmi": bmi})
            assert model.bmi == bmi
```

---

## Integration Tests — Testing the Full Request Cycle

```python title="tests/integration/test_predict_api.py"
"""
Integration tests use the TestClient to make real HTTP requests
through your entire FastAPI application stack.
"""
import pytest

VALID_INPUT = {
    "age": 35, "sex": "male", "bmi": 27.9,
    "children": 2, "smoker": "no", "region": "southeast"
}

class TestPredictEndpoint:
    
    def test_valid_prediction_returns_200(self, client):
        """
        Full integration test: 
        HTTP POST → Pydantic validation → mock model → response
        """
        response = client.post("/predict", json=VALID_INPUT)
        
        assert response.status_code == 200
        
        data = response.json()
        # Check structure
        assert "prediction" in data
        assert "confidence" in data
        assert "model_version" in data
        assert "probabilities" in data
        
        # Check value constraints
        assert data["prediction"] in ["low", "medium", "high"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["probabilities"], dict)
    
    def test_missing_required_field_returns_422(self, client):
        """When required field is absent, FastAPI returns 422 with details."""
        incomplete = {"age": 35, "sex": "male"}  # missing bmi, children, smoker, region
        
        response = client.post("/predict", json=incomplete)
        
        assert response.status_code == 422
        detail = response.json()["detail"]
        
        # Verify which fields are reported as missing
        missing_fields = {err["loc"][-1] for err in detail if err["type"] == "missing"}
        assert "bmi" in missing_fields
        assert "smoker" in missing_fields
        assert "region" in missing_fields
    
    def test_invalid_field_value_returns_422(self, client):
        """Wrong type causes 422."""
        invalid = {**VALID_INPUT, "age": "thirty-five"}
        response = client.post("/predict", json=invalid)
        assert response.status_code == 422
    
    def test_out_of_range_bmi_returns_422(self, client):
        """BMI below 10 is invalid."""
        invalid = {**VALID_INPUT, "bmi": 5.0}
        response = client.post("/predict", json=invalid)
        assert response.status_code == 422
    
    def test_model_not_loaded_returns_503(self, client, mock_model):
        """
        When the model store is empty (model not yet loaded),
        the endpoint should return 503 Service Unavailable.
        """
        # Override mock_model fixture to simulate no model
        mock_model.__bool__ = lambda self: False  # model evaluates as falsy
        model_store["model"] = None
        
        response = client.post("/predict", json=VALID_INPUT)
        assert response.status_code == 503
    
    def test_model_inference_error_returns_500(self, client, mock_model):
        """
        When the model throws an exception,
        the API should return 500 (not crash and not expose internals).
        """
        mock_model.predict.side_effect = RuntimeError("Internal model error")
        
        response = client.post("/predict", json=VALID_INPUT)
        
        assert response.status_code == 500
        # Verify we didn't leak the internal error message
        response_body = response.json()
        assert "Internal model error" not in str(response_body)
    
    def test_response_does_not_contain_sensitive_fields(self, client):
        """Ensure the response doesn't accidentally expose internal data."""
        response = client.post("/predict", json=VALID_INPUT)
        data = response.json()
        
        # These fields should never appear in the response
        forbidden_fields = {"model_path", "api_key", "database_url", "secret"}
        for field in forbidden_fields:
            assert field not in data, f"Sensitive field '{field}' found in response!"


class TestHealthEndpoints:
    def test_health_returns_ok_when_running(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_ready_with_model_loaded(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True
    
    def test_ready_without_model_returns_503(self, client):
        model_store["model"] = None
        response = client.get("/ready")
        assert response.status_code == 503
```

---

## Load Testing with Locust

```python title="tests/load/locustfile.py"
"""
Load test: simulates many concurrent users hitting your API.
Measures throughput (requests/second) and latency (P50, P95, P99).

Run: locust -f tests/load/locustfile.py --host=http://localhost:8000
Open: http://localhost:8089 → set number of users → start
"""
from locust import HttpUser, task, between
import random

class MLAPIUser(HttpUser):
    """
    Simulates a user of the ML API.
    
    wait_time = between(0.5, 2): waits 0.5-2 seconds between requests.
    Simulates real user behavior (not hammering as fast as possible).
    """
    wait_time = between(0.5, 2)
    
    def on_start(self):
        """Called when a simulated user starts. Login here."""
        resp = self.client.post("/auth/login", json={
            "username": "loadtest_user",
            "password": "loadtestpass"
        })
        if resp.status_code == 200:
            self.token = resp.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = None
            self.headers = {}
    
    @task(weight=10)
    def predict_insurance(self):
        """
        weight=10: this task is 10x more common than weight=1 tasks.
        Simulates the most common API usage.
        """
        payload = {
            "age": random.randint(18, 65),
            "sex": random.choice(["male", "female"]),
            "bmi": round(random.uniform(18, 45), 1),
            "children": random.randint(0, 4),
            "smoker": random.choice(["yes", "no"]),
            "region": random.choice(["northeast", "northwest", "southeast", "southwest"])
        }
        self.client.post(
            "/predict",
            json=payload,
            headers=self.headers,
            name="/predict",  # groups all predict requests under one label
        )
    
    @task(weight=2)
    def check_health(self):
        """Health check — less frequent."""
        self.client.get("/health")
    
    @task(weight=1)
    def get_model_info(self):
        """Model info — least frequent."""
        self.client.get("/model/info", headers=self.headers)
```

---

## Running Tests in CI with GitHub Actions

```yaml title=".github/workflows/test.yml"
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-test.txt

      - name: Run tests
        run: pytest --cov=app --cov-report=xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
```

Every pull request now automatically runs all tests. Merging is blocked if tests fail.

---

## Q&A

**Q: My test passes locally but fails in CI. Why?**

Common causes: (1) Environment variables not set in CI (add them to GitHub Secrets). (2) Test depends on local file that isn't in the repo. (3) Port conflict — something already using port 8000 in CI. (4) Different Python version in CI.

**Q: How do I test file upload endpoints?**

```python
def test_upload_model(client):
    import io
    fake_model_file = io.BytesIO(b"fake pkl content")
    response = client.post(
        "/models/upload",
        files={"model_file": ("model.pkl", fake_model_file, "application/octet-stream")},
    )
    assert response.status_code == 201
```

**Q: What's the right coverage percentage to target?**

80% is a widely accepted minimum. Don't try to reach 100% — testing every edge case of error handlers is expensive and often tests Python internals more than your code. Focus on the critical paths: prediction logic, validation, authentication.
