# Healthcare Feature Prediction

## Overview

The TCM (Total Charges Model) Service is a production-ready machine learning API built with FastAPI for predicting hospital total charges based on patient admission, demographic, and clinical attributes.

The service loads a trained ML pipeline from Amazon S3, serves predictions through REST endpoints, and provides observability through Prometheus metrics and health checks.

This project demonstrates a cloud-native machine learning deployment workflow with model serving, monitoring, health validation, and infrastructure-ready API endpoints for production environments.

---

## Problem Statement

Healthcare cost prediction is a critical problem for hospitals, insurers, and healthcare administrators. Estimating total patient charges accurately helps improve:

- financial planning
- patient billing transparency
- insurance claim forecasting
- hospital resource allocation
- operational decision-making

This project focuses on predicting **Total Charges** using patient admission records and categorical healthcare features such as diagnosis severity, admission type, demographic information, and payment typology.

---

## Dataset

The dataset consists of hospital inpatient discharge records containing patient demographics, admission details, severity indicators, and billing information.

The target variable is:

**Total Charges**

This value represents the total billed hospital charges for a patient encounter.

### Dataset Characteristics

- Tabular structured healthcare dataset
- Primarily categorical features
- Supervised regression problem
- Real-world hospital billing prediction use case

---

## Scope

This project includes:

- training a machine learning regression pipeline
- saving the trained pipeline using `joblib`
- storing model artifacts in Amazon S3
- automatic model retrieval during service startup
- serving predictions using FastAPI
- production health checks
- readiness endpoints
- observability with Prometheus metrics
- deployment-ready architecture for cloud environments

This project does not include:

- frontend dashboard UI
- real-time retraining pipeline
- streaming inference workflows

---

## Key Variables

### Target Variable

- **Total Charges** → predicted output variable

### Categorical Features

- Hospital County
- Age Group
- Gender
- Race
- Length of Stay
- Type of Admission
- Patient Disposition
- APR DRG Description
- APR Severity of Illness Description
- APR Risk of Mortality
- Payment Typology 1
- Payment Typology 2
- Payment Typology 3

### Dropped Variable

- Total Costs (Total Costs are what the hospital internally incurs, while Total Charges are what the hospital bills the patient/insurer. Many values of Total Costs were also missing, so the variable as a whole was dropped as imputation was not a viable option for this model.)

This model is primarily driven by categorical feature engineering.

---

## Tech Stack

### Backend

- Python
- FastAPI
- Pydantic

### Machine Learning

- Scikit-learn
- Pandas
- Joblib

### Cloud & Storage

- AWS
- Amazon S3
- Boto3

### Monitoring

- Prometheus
- prometheus-fastapi-instrumentator

### Deployment

- Linux-based production environment
- API-first microservice deployment

---

## Methodology

### 1. Data Preparation

- cleaned structured hospital admission records
- removed target leakage columns
- separated categorical and target variables
- prepared regression-ready training data

### 2. Model Training

A machine learning pipeline was trained to predict hospital total charges using categorical feature transformations and regression modeling.

The trained model was serialized using `joblib`.

### 3. Metadata Tracking

A metadata file stores:

- model version
- feature list
- evaluation metrics
- schema expectations

This supports production monitoring and reproducibility.

### 4. Cloud Deployment

Artifacts are uploaded to Amazon S3 and automatically downloaded during service startup.

### 5. API Serving

Prediction requests are sent to:

`POST /predict`

The API converts incoming JSON records into a DataFrame and returns prediction results.

### 6. Monitoring & Observability

The service exposes:

- `/health`
- `/ready`
- `/metrics`

for production readiness and monitoring through Prometheus.

---

## Results

### Model Performance

| Metric | Value |
|---|---:|
| MAE | 72,169.83 |
| RMSE | 211,134.73 |
| R² Score | 0.5083 |

### Interpretation

- The model explains approximately **50.8%** of the variance in hospital total charges
- Performance is reasonable given the complexity and variability of healthcare billing
- Additional feature engineering and richer numeric variables could further improve accuracy

---

## How to Run

```bash
# Clone the repository
git clone <your-repository-url>

# Navigate into project folder
cd tcm-model-service

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export S3_BUCKET=tcm-service
export AWS_REGION=us-east-1

# Run the FastAPI application
uvicorn main:app --reload
```

### Access Endpoints

- `POST /predict`
- `GET /health`
- `GET /ready`
- `GET /metrics`

Example:

```bash
http://localhost:8000/health
```

---

## Project Structure

```text
EB_Healthcare_Feature_Prediction/
│
├── app.py
├── requirements.txt
├── total_charges_pipeline.joblib
├── model_metadata.json
├── README.md
└── prometheus.yml
```
---

## Architecture Diagram

                           AWS Cloud

┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│                    Amazon S3                                               │
│          ┌───────────────────────────────┐                                 │
│          │ Model Artifacts               │                                 │
│          │ • pipeline.joblib             │                                 │
│          │ • metadata.json               │                                 │
│          └──────────────┬────────────────┘                                 │
│                         │ Download at Startup                              │
└─────────────────────────┼──────────────────────────────────────────────────┘
                          │
                          ▼
                 ┌──────────────────────┐
                 │    FastAPI Service   │
                 │──────────────────────│
                 │ Pydantic Validation  │
                 │ Prediction Endpoint  │
                 │ Health Checks        │
                 │ Prometheus Metrics   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Scikit-learn Model   │
                 │ Pipeline             │
                 └──────────┬───────────┘
                            │
                            ▼
                 Predicted Total Charges

        ▲
        │
        │ REST API
        │
┌──────────────────────┐
│      API Clients     │
│ Postman              │
│ curl                 │
│ Applications         │
└──────────────────────┘

---

## Author

### Aaron Tsui

Master of Science in Data Science  
Northwestern University 

Specializing in machine learning, deep learning, and applied AI systems for healthcare and large-scale predictive modeling.

- Email: aaron.tsui.careers@gmail.com  
- LinkedIn: https://www.linkedin.com/in/aaron-tsui/

