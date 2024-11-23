"""Constants for data preprocessing."""

INGESTION_FILENAME = 'vivo_ingested'
INGESTION_FILETYPE = 'csv'

FEAT_COLS = ["Open", "High", "Low", "Volume"]
TARGET_COL = "Close"

TEST_SIZE = 0.2
SEQUENCE_LENGTH = 10
BATCH_SIZE = 32