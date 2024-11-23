""" Preprocess data for training."""

import pandas as pd
import logging
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from src.config import setup_logging
from .utils import log_params
from src.data_preprocessing.custom_classes import TimeSeriesDataset
from src.data_preprocessing.constants import (
    INGESTION_FILENAME,
    INGESTION_FILETYPE,
    FEAT_COLS,
    TARGET_COL,
    TEST_SIZE,
    SEQUENCE_LENGTH,
    BATCH_SIZE,
)

# Logging
logger = logging.getLogger(__name__)

def main():

    PP_PATH = "data/data_preprocessing"

    logger.info("Preprocessing data...")

    # Load data
    df = pd.read_csv(f"data/data_ingestion/{INGESTION_FILENAME}.{INGESTION_FILETYPE}", parse_dates=['Date'], index_col='Date')

    # Splitting feat and target
    X = df[FEAT_COLS]
    y = df[TARGET_COL]

    # Splitting train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
    log_params(FEAT_COLS, TARGET_COL, TEST_SIZE, X_train, X_test)

    # Scaling data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    # Train and test datasets
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH)

    # Batches using DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Save batches
    train_batches = [(x, y) for x, y in train_loader]
    test_batches = [(x, y) for x, y in test_loader]

    torch.save(train_batches, f"{PP_PATH}/train_batches.pt")
    torch.save(test_batches, f"{PP_PATH}/test_batches.pt")

    logger.info("Data preprocessing completed.")

if __name__ == "__main__":
    main()
