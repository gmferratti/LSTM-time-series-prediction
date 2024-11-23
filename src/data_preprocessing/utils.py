import logging
import pandas as pd
import numpy as np

from src.config import setup_logging

# Logging
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