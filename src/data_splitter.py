import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
import logging

RANDOM_STATE = 37

class DataSplitter(ABC):
    def __init__(self, test_size: float=0.2, random_state=RANDOM_STATE):
        self.test_size = test_size
        self._random_state = random_state
        super().__init__(test_size, random_state)

    @abstractmethod
    def split_data(self, df, target_col):
        pass


class SimpleDataSplitter(DataSplitter):
    def split_data(self, df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self._random_state
            )
        logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
if __name__ == "__main__":
    # df = pd.read_csv("path-to-the-csv-file")
    # splitter = SimpleDataSplitter()
    # X_train, X_test, y_train, y_test = splitter.split_data(df, 'target_column')
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    pass