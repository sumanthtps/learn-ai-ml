---
id: serving-ml-models
title: "08 · Serving ML Models with FastAPI"
sidebar_label: "08 · Serving ML Models"
sidebar_position: 8
tags: [ml, scikit-learn, joblib, prediction, inference, fastapi, beginner]
---

# Serving ML Models with FastAPI

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=JdDoMi_vqbM) · **Series:** FastAPI for ML – CampusX

---

## Visual Reference

![Scikit-learn logo](https://commons.wikimedia.org/wiki/Special:Redirect/file/Scikit_learn_logo_small.svg)

Source: [Wikimedia Commons - Scikit learn logo small](https://commons.wikimedia.org/wiki/File:Scikit_learn_logo_small.svg)

## The Core Goal of This Video

Every concept we've learned — FastAPI, Pydantic, HTTP methods, schemas — all exists for one purpose: **taking a trained ML model and making it callable by anyone over HTTP**.

This is the moment where ML engineering meets software engineering. You've built a model. Now you serve it.

---

## The Problem We're Solving: Insurance Premium Prediction

**Business context:** An insurance company's website needs to show customers their estimated premium category before they apply. The actuarial team built a Random Forest model in Python. The web team builds in React. They need a bridge.

**The model:** Takes 6 features (age, sex, BMI, children, smoker status, region) and predicts one of three premium categories: **low**, **medium**, **high**.

**Our job:** Wrap this model in a FastAPI endpoint so the React frontend can call `POST /predict` and get a prediction back in JSON.

---

## Step 1: Train the Model (Run This Once)

The model is trained offline and saved to disk. The API loads it at startup — training never happens inside the API.

```python title="train.py"
"""
Run this script once to train and save the model.
Training is separate from serving — never train inside your API.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# ─── Load and prepare data ───────────────────────────────────────
df = pd.read_csv("data/insurance.csv")

# Create target: bin insurance charges into low/medium/high
p33 = df["charges"].quantile(0.33)
p67 = df["charges"].quantile(0.67)
df["premium_category"] = df["charges"].apply(
    lambda x: "low" if x < p33 else ("medium" if x < p67 else "high")
)

X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["premium_category"]

# ─── Build preprocessing + model pipeline ────────────────────────
# KEY INSIGHT: Always save preprocessing WITH the model.
# If you only save the model (not the preprocessor), your API must
# manually preprocess the same way as training — and will inevitably
# do it slightly differently, causing the "training/serving skew" bug.
categorical_features = ["sex", "smoker", "region"]
numerical_features = ["age", "bmi", "children"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ("num", StandardScaler(), numerical_features),
])

# Pipeline automatically applies preprocessor then classifier
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1    # use all CPU cores during training
    ))
])

# ─── Train and evaluate ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y     # ensure each split has the same class distribution
)

pipeline.fit(X_train, y_train)

print("=== Model Evaluation ===")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Test Accuracy: {pipeline.score(X_test, y_test):.3f}")

# ─── Save the complete pipeline ───────────────────────────────────
os.makedirs("artifacts", exist_ok=True)

joblib.dump(pipeline, "artifacts/insurance_model.pkl")
# joblib is better than pickle for sklearn objects:
# - Handles numpy arrays efficiently
# - Better compression
# - More reliable across Python versions
print("\nModel saved to artifacts/insurance_model.pkl")
print("Classes:", pipeline.classes_)
```

Run once: `python train.py`

---

## Step 2: Define the Prediction Schema

Before writing the API, define exactly what it accepts and returns. This is your contract with clients.

```python title="schemas.py"
from pydantic import BaseModel, Field
from typing import Literal

class InsuranceInput(BaseModel):
    """
    Features required for insurance premium prediction.
    
    CRITICAL: Field names, types, and allowed values must EXACTLY MATCH
    what the model was trained on. If the model was trained with
    smoker as "yes"/"no" (strings), you must validate the same values.
    
    Pydantic's Literal type enforces this — any other value causes 422.
    """
    age: int = Field(
        ge=18, le=100,
        description="Age of the primary insured",
        examples=[30, 45, 60]
    )
    sex: Literal["male", "female"] = Field(
        description="Biological sex of the insured"
    )
    bmi: float = Field(
        ge=10.0, le=65.0,
        description="Body Mass Index (kg/m²)"
    )
    children: int = Field(
        default=0, ge=0, le=10,
        description="Number of dependent children"
    )
    smoker: Literal["yes", "no"] = Field(
        description="Whether the insured smokes"
    )
    region: Literal["northeast", "northwest", "southeast", "southwest"] = Field(
        description="Residential area of the insured"
    )
    
    # This example appears in Swagger UI — helps developers know what to send
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "age": 35,
                "sex": "male",
                "bmi": 27.9,
                "children": 2,
                "smoker": "no",
                "region": "southeast"
            }]
        }
    }


class PredictionOutput(BaseModel):
    """
    Response from the prediction endpoint.
    
    Always define a response schema — it:
    1. Documents what clients can expect
    2. Validates your own output (catches bugs where your code returns wrong types)
    3. Prevents accidentally returning internal fields
    """
    prediction: Literal["low", "medium", "high"] = Field(
        description="Predicted premium category"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model's confidence (probability of the predicted class)"
    )
    probabilities: dict[str, float] = Field(
        description="Probability distribution over all classes"
    )
    model_version: str = Field(
        description="Version string for the model that produced this prediction"
    )
```

---

## Step 3: Build the FastAPI Prediction API

```python title="main.py"
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from schemas import InsuranceInput, PredictionOutput

# ─── Model storage ───────────────────────────────────────────────
# Module-level dict to hold the loaded model.
# dict is thread-safe for reading (which is all we do after startup).
# A class-based service would also work; a dict is simpler for learning.
model_store = {}
MODEL_VERSION = "1.0.0"

# ─── Startup: load model once ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    The model loads HERE, once, when the server starts.
    
    Every subsequent prediction request uses model_store["model"]
    — a dict lookup that takes nanoseconds, not the 200-500ms
    of loading from disk.
    
    If model loading fails, the server still starts but /predict
    will return 503. This is better than crashing at startup
    (which prevents /health from responding and confuses orchestrators).
    """
    print("🚀 Server starting...")
    
    try:
        model_store["model"] = joblib.load("artifacts/insurance_model.pkl")
        # Store class labels too — model.classes_ gives ["high", "low", "medium"]
        model_store["classes"] = list(model_store["model"].classes_)
        print(f"✅ Model loaded. Classes: {model_store['classes']}")
    except FileNotFoundError:
        print("⚠️ Model file not found. Run train.py first.")
        print("   Server will start, but /predict will return 503.")
        model_store["model"] = None
        model_store["classes"] = []
    
    yield   # Server handles requests here
    
    # Shutdown cleanup
    model_store.clear()
    print("👋 Server shutting down. Model unloaded.")


app = FastAPI(
    title="Insurance Premium Prediction API",
    description="Predicts insurance premium category (low/medium/high)",
    version=MODEL_VERSION,
    lifespan=lifespan
)

# ─── Operational endpoints ────────────────────────────────────────

@app.get("/", tags=["info"])
def root():
    """API information."""
    return {
        "name": "Insurance Premium Prediction API",
        "version": MODEL_VERSION,
        "endpoints": {
            "predict": "POST /predict",
            "docs": "GET /docs",
            "health": "GET /health"
        }
    }

@app.get("/health", tags=["ops"])
def health():
    """
    Liveness probe — answers "is the process running?"
    Should always return 200 as long as the process is alive.
    Even if the model isn't loaded, the process is still "healthy" (alive).
    """
    return {"status": "ok", "version": MODEL_VERSION}

@app.get("/ready", tags=["ops"])
def readiness():
    """
    Readiness probe — answers "is the API ready to serve predictions?"
    Returns 503 if the model isn't loaded (prediction would fail anyway).
    
    Kubernetes uses this to decide whether to route traffic here.
    """
    model_loaded = model_store.get("model") is not None
    return {
        "ready": model_loaded,
        "status": "ready" if model_loaded else "model_not_loaded",
        "classes": model_store.get("classes", []),
    }

@app.get("/model/info", tags=["model"])
def model_info():
    """Return metadata about the currently loaded model."""
    if not model_store.get("model"):
        raise HTTPException(503, "Model not loaded. Run train.py and restart the server.")
    
    return {
        "version": MODEL_VERSION,
        "algorithm": "RandomForestClassifier",
        "features": ["age", "sex", "bmi", "children", "smoker", "region"],
        "target_classes": model_store.get("classes", []),
        "preprocessing": "OneHotEncoder (categorical) + StandardScaler (numerical)"
    }

# ─── Prediction endpoint ──────────────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["prediction"],
    summary="Predict insurance premium category",
)
def predict(data: InsuranceInput):
    """
    Predict whether the insurance premium is low, medium, or high.
    
    Why regular 'def' and not 'async def'?
    ML inference is CPU-bound — it uses the processor heavily.
    Using 'async def' for CPU-bound work blocks the event loop,
    making the API handle requests sequentially.
    'def' causes FastAPI to run this in a thread pool — better for CPU work.
    """
    # ─── 1. Verify model is loaded ────────────────────────────────
    if not model_store.get("model"):
        raise HTTPException(
            status_code=503,    # 503 Service Unavailable
            detail="Prediction service is unavailable. Model not loaded. Try again in a few moments."
        )
    
    # ─── 2. Prepare input features ───────────────────────────────
    # The sklearn pipeline expects a pandas DataFrame with specific column names.
    # Column order must match what the model was trained on.
    # We pass all 6 feature columns by name — order is handled by ColumnTransformer.
    input_df = pd.DataFrame([{
        "age": data.age,
        "sex": data.sex,
        "bmi": data.bmi,
        "children": data.children,
        "smoker": data.smoker,
        "region": data.region,
    }])
    # input_df looks like:
    # age  sex     bmi   children smoker region
    # 35   male    27.9  2        no     southeast
    
    # ─── 3. Run inference ─────────────────────────────────────────
    model = model_store["model"]
    
    try:
        # predict() returns array of class names: ["medium"]
        prediction_array = model.predict(input_df)
        prediction = str(prediction_array[0])
        
        # predict_proba() returns array of probabilities for each class
        # classes are in alphabetical order: ["high", "low", "medium"]
        proba_array = model.predict_proba(input_df)
        proba_list = proba_array[0].tolist()  # numpy array → Python list
        
    except Exception as e:
        # Catch any sklearn error (shape mismatch, unexpected values, etc.)
        # Don't expose the internal error message — it may contain sensitive info
        print(f"Inference error: {e}")   # log for your debugging
        raise HTTPException(
            status_code=500,
            detail="Inference failed due to an internal error. Please check your input values."
        )
    
    # ─── 4. Build response ────────────────────────────────────────
    classes = model_store["classes"]   # ["high", "low", "medium"]
    
    # Map class label to its probability
    probabilities = {
        cls: round(float(prob), 4)
        for cls, prob in zip(classes, proba_list)
    }
    # e.g., {"high": 0.1, "low": 0.2, "medium": 0.7}
    
    # Confidence = probability of the predicted class
    confidence = probabilities[prediction]
    
    return PredictionOutput(
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        model_version=MODEL_VERSION
    )
```

---

## Step 4: Streamlit Frontend

Connect a web UI to your API:

```python title="frontend.py"
"""
Run separately: streamlit run frontend.py
(API must be running: uvicorn main:app --reload)
"""
import streamlit as st
import requests

st.set_page_config(page_title="Insurance Premium Predictor", page_icon="🏥")
st.title("🏥 Insurance Premium Predictor")
st.markdown("Fill in the details below to predict the insurance premium category.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 35)
        bmi = st.number_input("BMI", 10.0, 65.0, 27.5, step=0.1)
        smoker = st.selectbox("Smoker?", ["no", "yes"])
    with col2:
        sex = st.selectbox("Sex", ["male", "female"])
        children = st.number_input("Children", 0, 10, 0)
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    
    submitted = st.form_submit_button("🔮 Predict Premium", use_container_width=True)

if submitted:
    payload = {
        "age": age, "sex": sex, "bmi": float(bmi),
        "children": children, "smoker": smoker, "region": region
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            color = {"low": "green", "medium": "orange", "high": "red"}[result["prediction"]]
            
            st.markdown(f"### Prediction: :{color}[**{result['prediction'].upper()}**]")
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            
            st.subheader("Class Probabilities")
            for cls, prob in sorted(result["probabilities"].items()):
                st.progress(prob, text=f"{cls.capitalize()}: {prob*100:.1f}%")
        
        elif response.status_code == 422:
            errors = response.json()["detail"]
            for err in errors:
                st.error(f"Field '{err['loc'][-1]}': {err['msg']}")
        
        else:
            st.error(f"API Error {response.status_code}: {response.json().get('detail')}")
    
    except requests.ConnectionError:
        st.error("❌ Cannot connect to the API. Is `uvicorn main:app --reload` running?")
    except requests.Timeout:
        st.error("⏱️ Request timed out.")
```

---

## Running the Complete Stack

```bash
# Step 1: Train the model (once)
python train.py

# Step 2: Start the API (Terminal 1)
uvicorn main:app --reload
# → http://localhost:8000

# Step 3: Start the frontend (Terminal 2)
streamlit run frontend.py
# → http://localhost:8501

# Step 4: Test the API directly
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"sex":"male","bmi":27.9,"children":2,"smoker":"no","region":"southeast"}'

# Expected:
# {"prediction":"medium","confidence":0.72,"probabilities":{"high":0.08,"low":0.2,"medium":0.72},"model_version":"1.0.0"}
```

---

## Q&A

**Q: Why does `model.predict()` need a DataFrame and not just a list?**

The `ColumnTransformer` preprocessor applies different transformations to different columns by name ("age" gets StandardScaler, "sex" gets OneHotEncoder). A plain Python list `[35, "male", 27.9, 2, "no", "southeast"]` has no column names — the preprocessor can't distinguish "age" from "sex". A named DataFrame provides the column information the preprocessor needs.

**Q: My model gives the same prediction whether input is valid or garbage. What's wrong?**

You likely saved the model without the preprocessing pipeline. If `model.pkl` is only the `RandomForestClassifier` (not the full `Pipeline`), raw string values like "male" or "northeast" get silently converted to NaN by sklearn, which the model handles poorly. Always save `Pipeline(steps=[("preprocessor", ...), ("classifier", ...)])` as one object.

**Q: What does `model.classes_` return?**

The unique target class labels that the model was trained on, in alphabetical order. For our three-class problem: `["high", "low", "medium"]`. The order matters because `model.predict_proba()` returns probabilities in this same order. We use `zip(classes, proba_list)` to correctly pair each class with its probability.

**Q: How do I update the model without restarting the server?**

Train a new model, save it to disk, then add a reload endpoint:
```python
@app.post("/model/reload")
def reload_model(admin_key: str = Depends(verify_admin)):
    model_store["model"] = joblib.load("artifacts/insurance_model.pkl")
    model_store["classes"] = list(model_store["model"].classes_)
    return {"message": "Model reloaded"}
```
This avoids the ~1 second downtime of a full server restart.

**Q: How many predictions per second can this handle?**

With 4 Uvicorn workers and this sklearn Random Forest: roughly 200-500 predictions/second, depending on hardware. Bottleneck is usually the sklearn inference (~2-10ms per prediction). For higher throughput: use batch prediction (send multiple inputs in one request), run inference in a thread pool with `run_in_executor`, or switch to a faster model (LightGBM, XGBoost).
