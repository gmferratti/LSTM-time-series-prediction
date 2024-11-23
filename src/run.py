from src.data_ingestion.ingest_data import main as ingest_data
from src.data_preprocessing.preprocess import main as preprocess_data
#from src.modeling.train_evaluate import main as train_evaluate

def run():
    ingest_data()
    preprocess_data()
    # train_evaluate()

if __name__ == "__main__":
    run()