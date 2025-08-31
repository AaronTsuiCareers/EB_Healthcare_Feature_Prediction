from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import boto3
import os
import json
from pathlib import Path
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from prometheus_fastapi_instrumentator import Instrumentator

MODEL_KEY = "total_charges_pipeline.joblib"
META_KEY = "model_metadata.json"
MODEL_DIR = "/home/ubuntu/tcm-service"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_KEY)
META_PATH = os.path.join(MODEL_DIR, META_KEY)
BUCKET = os.getenv("S3_BUCKET", "tcm-service")
REGION = os.getenv("AWS_REGION", "us-east-1")

PREDICTION_COUNTER = Counter('model_predictions_total', 'Total number of predictions made', ['model_version'])
PREDICTION_ERRORS = Counter('model_prediction_errors_total', 'Total number of prediction errors', ['error_type'])
PREDICTION_DURATION = Histogram('model_prediction_duration_seconds', 'Time spent making predictions in seconds')
MODEL_LOAD_DURATION = Histogram('model_load_duration_seconds', 'Time spent loading the model from disk')
MODEL_LOAD_COUNTER = Counter('model_load_operations_total', 'Total number of model load operations', ['status'])
MODEL_AGE = Gauge('model_age_seconds', 'Age of the loaded model in seconds since last modification')
MODEL_LOAD_TIMESTAMP = Gauge('model_load_timestamp_seconds', 'Timestamp when model was last loaded')

class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    predictions: List[float]

class HealthResponse(BaseModel):
    loaded: bool
    load_time: float = None
    model_version: str = "unknown"
    features: List[str] = []

app = FastAPI(title="TCM Model Service", version="1.0.0")

model = None
metadata: Dict[str, Any] = {}
model_load_time = None

def download_from_s3(s3_key: str, local_path: str):
    """Download a file from S3 to the specified local path"""
    s3 = boto3.client("s3", region_name=REGION)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(BUCKET, s3_key, local_path)
    print(f"Downloaded {s3_key} to {local_path}")

def load_model():
    global model, metadata, model_load_time
    
    start_time = time.time()
    
    try:
        if not Path(MODEL_PATH).exists():
            print(f"Downloading model from S3: {MODEL_KEY}")
            download_from_s3(MODEL_KEY, MODEL_PATH)
        
        if not Path(META_PATH).exists():
            print(f"Downloading metadata from S3: {META_KEY}")
            download_from_s3(META_KEY, META_PATH)
        
        print("Loading model...")
        model = joblib.load(MODEL_PATH)
        
        print("Loading metadata...")
        with open(META_PATH) as f:
            metadata.update(json.load(f))
        
        model_load_time = time.time()
        MODEL_LOAD_TIMESTAMP.set(model_load_time)
        model_file_mtime = os.path.getmtime(MODEL_PATH)
        MODEL_AGE.set(time.time() - model_file_mtime)
        MODEL_LOAD_COUNTER.labels(status='success').inc()
        
        print(f"Model loaded successfully with version: {metadata.get('model_version', 'unknown')}")
        print(f"Features: {list(metadata.get('features', []))}")
        
    except Exception as e:
        MODEL_LOAD_COUNTER.labels(status='error').inc()
        print(f"Error loading model: {e}")
        raise
    
    finally:
        load_duration = time.time() - start_time
        MODEL_LOAD_DURATION.observe(load_duration)

@app.on_event("startup")
async def startup_event():
    """Load model when the application starts"""
    load_model()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        import pandas as pd
        df = pd.DataFrame(request.records)
        preds = model.predict(df)
        predictions = preds.tolist()

        model_version = metadata.get('model_version', 'unknown')
        PREDICTION_COUNTER.labels(model_version=model_version).inc(len(predictions))
        prediction_duration = time.time() - start_time
        PREDICTION_DURATION.observe(prediction_duration)
        
        return PredictionResponse(predictions=predictions)
    
    except Exception as e:
        error_type = type(e).__name__
        PREDICTION_ERRORS.labels(error_type=error_type).inc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(REGISTRY)

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        loaded=model is not None,
        load_time=model_load_time,
        model_version=metadata.get('model_version', 'unknown'),
        features=list(metadata.get('features', []))
    )

@app.get("/ready")
async def ready():
    """Readiness check endpoint"""
    if model is not None:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Model not loaded")

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)