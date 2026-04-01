---
id: serving-ml-models
title: "08 · Serving ML Models with FastAPI"
sidebar_label: "08 · Serving ML Models"
sidebar_position: 8
tags: [ml, scikit-learn, joblib, prediction, inference, fastapi]
---

# Serving ML Models with FastAPI

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=JdDoMi_vqbM) · **Series:** FastAPI for ML – CampusX

---

## Project: Insurance Premium Prediction API

We build an end-to-end ML API:
1. Train a Random Forest model on insurance data
2. Save the model with `joblib`
3. Build a `/predict` endpoint with FastAPI
4. Connect a Streamlit frontend to the API

---

## Step 1: Train and Save the Model

```python title="train.py"
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load data
df = pd.read_csv("insurance.csv")

# Feature engineering
# Target: premium_category (low / medium / high)
X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["premium_category"]

# Preprocessing pipeline
categorical_features = ["sex", "smoker", "region"]
numerical_features = ["age", "bmi", "children"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", "passthrough", numerical_features)
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)

print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

# Save the model
joblib.dump(model, "artifacts/insurance_model.pkl")
print("Model saved to artifacts/insurance_model.pkl")
```

---

## Step 2: Define the Pydantic Schema

```python title="schemas.py"
from pydantic import BaseModel, Field
from typing import Literal

class InsuranceInput(BaseModel):
    """Features required for insurance premium prediction."""
    age: int = Field(ge=18, le=100, description="Age of the insured")
    sex: Literal["male", "female"]
    bmi: float = Field(ge=10.0, le=60.0, description="Body Mass Index")
    children: int = Field(ge=0, le=10)
    smoker: Literal["yes", "no"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "age": 30,
                "sex": "male",
                "bmi": 27.5,
                "children": 2,
                "smoker": "no",
                "region": "southeast"
            }]
        }
    }

class PredictionOutput(BaseModel):
    """Response from the prediction endpoint."""
    prediction: Literal["low", "medium", "high"]
    confidence: float
    model_version: str = "1.0.0"
```

---

## Step 3: Build the FastAPI App

```python title="main.py"
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from schemas import InsuranceInput, PredictionOutput

# ─── Load Model at Startup ───────────────────────────────────────

model_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup, clean up on shutdown."""
    try:
        model_store["model"] = joblib.load("artifacts/insurance_model.pkl")
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model file not found — run train.py first")
    yield
    model_store.clear()

app = FastAPI(
    title="Insurance Premium Prediction API",
    description="Predicts insurance premium category (low/medium/high)",
    version="1.0.0",
    lifespan=lifespan
)

# ─── Endpoints ───────────────────────────────────────────────────

@app.get("/", tags=["info"])
def root():
    return {"message": "Insurance Premium Prediction API", "version": "1.0.0"}

@app.get("/health", tags=["info"])
def health():
    model_loaded = "model" in model_store
    return {"status": "ok" if model_loaded else "degraded", "model_loaded": model_loaded}

@app.post("/predict", response_model=PredictionOutput, tags=["prediction"])
def predict(data: InsuranceInput):
    if "model" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert input to DataFrame (matching training format)
    input_df = pd.DataFrame([{
        "age": data.age,
        "sex": data.sex,
        "bmi": data.bmi,
        "children": data.children,
        "smoker": data.smoker,
        "region": data.region
    }])
    
    model = model_store["model"]
    
    # Get prediction and probability
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    confidence = float(max(probabilities))
    
    return PredictionOutput(
        prediction=prediction,
        confidence=round(confidence, 4),
        model_version="1.0.0"
    )
```

---

## Step 4: Streamlit Frontend (Calling the API)

```python title="frontend.py"
import streamlit as st
import requests

st.title("Insurance Premium Predictor")
st.markdown("Enter customer details to predict their insurance premium category.")

with st.form("prediction_form"):
    age = st.slider("Age", 18, 100, 30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.5)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    
    submitted = st.form_submit_button("Predict Premium")

if submitted:
    payload = {
        "age": age, "sex": sex, "bmi": bmi,
        "children": children, "smoker": smoker, "region": region
    }
    
    try:
        response = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            color = {"low": "green", "medium": "orange", "high": "red"}[result["prediction"]]
            st.markdown(f"### Premium Category: :{color}[{result['prediction'].upper()}]")
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
        else:
            st.error(f"Prediction failed: {response.json().get('detail')}")
    except requests.ConnectionError:
        st.error("Cannot connect to the API. Is it running?")
```

---

## Topics Not Covered in the Video

### Batch Prediction Endpoint

```python
from typing import List

class BatchInsuranceInput(BaseModel):
    inputs: List[InsuranceInput]

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    total: int

@app.post("/batch-predict", response_model=BatchPredictionOutput, tags=["prediction"])
def batch_predict(batch: BatchInsuranceInput):
    if not batch.inputs:
        raise HTTPException(400, "Empty batch")
    
    model = model_store["model"]
    
    df = pd.DataFrame([inp.model_dump() for inp in batch.inputs])
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)
    
    results = [
        PredictionOutput(
            prediction=pred,
            confidence=round(float(max(prob)), 4)
        )
        for pred, prob in zip(predictions, probabilities)
    ]
    
    return BatchPredictionOutput(predictions=results, total=len(results))
```

### Model Metadata Endpoint

```python
@app.get("/model/info", tags=["model"])
def model_info():
    return {
        "name": "insurance_premium_classifier",
        "version": "1.0.0",
        "algorithm": "RandomForestClassifier",
        "features": ["age", "sex", "bmi", "children", "smoker", "region"],
        "classes": ["low", "medium", "high"],
        "training_date": "2024-01-15",
        "metrics": {
            "test_accuracy": 0.87,
            "f1_score": 0.86
        }
    }
```

### Model Hot-Reload (Without Restart)

```python
import threading

model_lock = threading.Lock()

@app.post("/model/reload", tags=["admin"])
def reload_model(api_key: str = Header(...)):
    if api_key != settings.ADMIN_API_KEY:
        raise HTTPException(403, "Unauthorized")
    
    with model_lock:
        try:
            new_model = joblib.load("artifacts/insurance_model.pkl")
            model_store["model"] = new_model
            return {"message": "Model reloaded successfully"}
        except Exception as e:
            raise HTTPException(500, f"Reload failed: {str(e)}")
```

---

## Q&A

**Q: Why load the model in `lifespan` and not inside the endpoint function?**
> Loading inside the endpoint runs on every single request — potentially reloading a 500MB model thousands of times per minute. Using `lifespan` loads it once at startup and reuses it. The difference is microseconds vs. seconds per request.

**Q: My model needs preprocessing. Should I save it separately?**
> Always save your preprocessing pipeline **together with** the model using `joblib`. Use `sklearn.pipeline.Pipeline` to wrap preprocessing + model into one object. This prevents the "training/serving skew" bug where your API uses different preprocessing than training.

**Q: What if the model raises an exception during inference?**
> Wrap inference in `try/except`:
> ```python
> try:
>     prediction = model.predict(input_df)[0]
> except Exception as e:
>     raise HTTPException(500, f"Inference failed: {str(e)}")
> ```

**Q: How do I handle model versioning?**
> Tag your model artifacts by version: `model_v1.pkl`, `model_v2.pkl`. Load the version specified in config. Return the version in the response. This allows canary deployments and rollbacks.

**Q: Can I run inference asynchronously?**
> scikit-learn's `predict()` is synchronous and CPU-bound. Running it with `async def` won't help — Python's GIL still blocks. Use `run_in_executor` to run CPU-bound inference in a thread pool without blocking the async event loop:
> ```python
> import asyncio
> from concurrent.futures import ThreadPoolExecutor
> executor = ThreadPoolExecutor(max_workers=4)
> 
> @app.post("/predict")
> async def predict_async(data: InsuranceInput):
>     loop = asyncio.get_event_loop()
>     result = await loop.run_in_executor(executor, run_inference, data)
>     return result
> ```

**Q: What's the difference between `/predict` and `/v1/predict`?**
> Including the version prefix in the URL is a best practice for production APIs. It allows you to release breaking changes (new schema, new model) under `/v2/predict` without breaking existing clients still using `/v1/predict`.
