import logging
import pandas as pd
import numpy as np

from src.config import setup_logging

# Logging
setup_logging()
logger = logging.getLogger(__name__)

def log_params(
    FEAT_COLS: list,
    TARGET_COL: str,
    TEST_SIZE: float,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
):
    """
    Loga os parâmetros e informações do dataset após a divisão em treino e teste.
    """
    logger.info("Test size ratio: %.2f", TEST_SIZE)
    logger.info(f"Features: {FEAT_COLS}, Target: {TARGET_COL}")
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas de erro para avaliar a performance de modelos preditivos.

    Parâmetros:
    y_true (array-like): Valores reais observados (ground truth).
    y_pred (array-like): Valores preditos pelo modelo.

    Retorna:
    tuple: Uma tupla contendo as seguintes métricas:
        - mae (float): Mean Absolute Error (Erro Médio Absoluto).
        - rmse (float): Root Mean Square Error (Raiz do Erro Quadrático Médio).
        - mape (float): Mean Absolute Percentage Error (Erro Médio Absoluto Percentual em %).
    
    Fórmulas:
    - MAE: mean(|y_true - y_pred|)
    - RMSE: sqrt(mean((y_true - y_pred)^2))
    - MAPE: mean(|(y_true - y_pred) / y_true|) * 100
    """
    # Mean Absolute Error (MAE)
    # Mede o erro médio absoluto entre os valores reais e preditos.
    mae = np.mean(np.abs(y_true - y_pred))

    # Root Mean Square Error (RMSE)
    # Mede o erro quadrático médio, mas é mais sensível a outliers devido ao quadrado das diferenças.
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Mean Absolute Percentage Error (MAPE)
    # Mede o erro percentual médio absoluto, útil para interpretar o erro em relação ao tamanho dos valores reais.
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Retorna as métricas calculadas
    return mae, rmse, mape
