import pandas as pd
from src.data_splitter import SimpleDataSplitter
from typing import Tuple
from zenml import step

@step
def split_data(file_path: str, target_column: str, test_size: float=0.2)-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(file_path)
    splitter = SimpleDataSplitter(test_size=test_size)
    X_train, X_test, y_train, y_test = splitter.split_data(df, target_column)
    return X_train, X_test, y_train, y_test