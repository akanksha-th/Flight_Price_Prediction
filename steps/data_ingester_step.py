import pandas as pd
from src.data_ingester import get_data_ingester
from zenml import step

@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    data_ingester = get_data_ingester(file_path)
    df = data_ingester.ingest()
    return df