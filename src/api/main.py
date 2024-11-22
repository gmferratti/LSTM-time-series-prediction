import os
from fastapi import FastAPI
from mlflow.pyfunc import load_model
import mlflow
import pandas as pd

# Exemplo: runs:/134ba4849a6749e4bffedc3bd21f4ec1/mlp_model
MODEL_URI = os.getenv("MODEL_URI")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI()
model = load_model(MODEL_URI)

@app.post('/predict')
def predict(data: dict):
    # Preprocess data
    # Generate predictions
    return {'predictions': model.predict(pd.DataFrame(data))}
