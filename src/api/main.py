from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import joblib
from google.cloud import storage
from typing import List
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class StockData(BaseModel):
    close_prices: List[float]
    volumes: List[float]


class PredictionResponse(BaseModel):
    predicted_price: float


# Inicializar FastAPI
app = FastAPI(title="Stock Price Prediction API")

# Configurações
BUCKET_NAME = "tcc-fiap-mlet-models"
MODEL_PATH = "model.pth"
SCALER_PATH = "scaler.pkl"
SEQUENCE_LENGTH = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 2


def load_from_gcs():
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        # Download modelo
        model_blob = bucket.blob(MODEL_PATH)
        model_blob.download_to_filename(MODEL_PATH)

        # Download scaler
        scaler_blob = bucket.blob(SCALER_PATH)
        scaler_blob.download_to_filename(SCALER_PATH)

        # Carregar modelo
        model = LSTMModel(input_size=2, hidden_size=HIDDEN_SIZE,
                          num_layers=NUM_LAYERS, output_size=1)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

        # Carregar scaler
        scaler = joblib.load(SCALER_PATH)

        return model, scaler
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao carregar modelo")


# Carregar modelo e scaler na inicialização
model, scaler = load_from_gcs()


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: StockData):
    try:
        if len(data.close_prices) != SEQUENCE_LENGTH or len(data.volumes) != SEQUENCE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Entrada deve conter exatamente {
                    SEQUENCE_LENGTH} pontos de dados"
            )

        # Preparar dados
        input_data = np.column_stack((data.close_prices, data.volumes))
        scaled_data = scaler.transform(input_data)

        # Converter para tensor
        input_tensor = torch.FloatTensor(scaled_data).unsqueeze(0)

        # Fazer predição
        with torch.no_grad():
            scaled_prediction = model(input_tensor)

        # Reverter scaling
        prediction = scaler.inverse_transform(
            np.column_stack([scaled_prediction.numpy(),
                            np.zeros_like(scaled_prediction.numpy())])
        )[0, 0]

        return PredictionResponse(predicted_price=float(prediction))

    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
