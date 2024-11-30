import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from typing import Tuple
import logging
import joblib
from google.cloud import storage
import mlflow
import yfinance as yf

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Dataset personalizado para séries temporais."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """Modelo LSTM para previsão de séries temporais."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def get_stock_data(start_date: str = '2018-01-01') -> pd.DataFrame:
    """Baixa os dados históricos de preços da ação."""
    end_date = datetime.today().strftime('%Y-%m-%d')

    df = (
        yf.download('VIVT3.SA', start=start_date,
                    end=end_date, group_by='column')
        .reset_index()
        .droplevel(level=1, axis=1)
    )
    df = df[['Date', 'Close', 'Volume']]

    return df


def prepare_sequences(data: pd.DataFrame, seq_length: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Prepara sequências de dados para treino do modelo LSTM."""
    # Selecionar features
    features = data[['Close', 'Volume']].values

    # Dividir em treino e validação
    train_size = int(len(features) * 0.8)
    train_data = features[:train_size]
    val_data = features[train_size:]

    # Escalonar dados
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_data)
    scaled_val = scaler.transform(val_data)

    # Preparar sequências
    X_train, y_train = [], []
    for i in range(len(scaled_train) - seq_length):
        X_train.append(scaled_train[i:(i + seq_length)])
        y_train.append(scaled_train[i + seq_length, 0])

    X_val, y_val = [], []
    for i in range(len(scaled_val) - seq_length):
        X_val.append(scaled_val[i:(i + seq_length)])
        y_val.append(scaled_val[i + seq_length, 0])

    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), scaler


def save_to_gcs(bucket_name: str, model: nn.Module, scaler) -> None:
    """Salva modelo e scaler no Google Cloud Storage."""
    # Salvar localmente primeiro
    torch.save(model.state_dict(), 'model.pth')
    joblib.dump(scaler, 'scaler.pkl')

    # Upload para GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Upload modelo
    model_blob = bucket.blob('model.pth')
    model_blob.upload_from_filename('model.pth')

    # Upload scaler
    scaler_blob = bucket.blob('scaler.pkl')
    scaler_blob.upload_from_filename('scaler.pkl')

    logger.info(f"Modelo e scaler salvos no bucket: {bucket_name}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device
) -> None:
    """Treina o modelo LSTM."""
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validação
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs.squeeze(), batch_y).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            logger.info(f"""
                        Epoch [{epoch+1}/{num_epochs}],
                        Train Loss: {train_loss:.4f},
                        Val Loss: {val_loss:.4f}""")

            # Log métricas no MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss
            }, step=epoch)


def main():
    # Configurações
    BUCKET_NAME = "tcc-fiap-mlet-models"
    SEQUENCE_LENGTH = 10
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001

    # MLflow tracking
    mlflow.set_experiment("stock-prediction-lstm")

    with mlflow.start_run():
        # Log parâmetros
        mlflow.log_params({
            "sequence_length": SEQUENCE_LENGTH,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE
        })

        # Preparar dados
        df = get_stock_data()
        X_train, y_train, X_val, y_val, scaler = prepare_sequences(
            df, SEQUENCE_LENGTH)

        # Criar dataloaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Configurar dispositivo
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Inicializar modelo
        model = LSTMModel(
            input_size=X_train.shape[2],
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            output_size=1
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Treinar modelo
        train_model(
            model, train_loader, val_loader,
            criterion, optimizer, NUM_EPOCHS, device
        )

        # Salvar modelo e scaler no GCS
        save_to_gcs(BUCKET_NAME, model, scaler)


if __name__ == "__main__":
    main()
