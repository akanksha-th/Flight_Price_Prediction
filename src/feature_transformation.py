import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from abc import ABC, abstractmethod
from dataclasses import dataclass

# ---------------------- Missing Value Imputer ----------------------
class Missing(ABC):
    def __init__(self, df: pd.DataFrame) -> pd.DataFrame:
        self._df = df.copy()
        self.imputer = None

    @abstractmethod
    def run(self) -> pd.DataFrame:
        "Run the Imputation and return the cleaned DataFrame."
        pass


class SimpleMeanMedianImputer(BaseImputer):
    def __init__(self, df: pd.DataFrame, strategy: str = "median"):
        """
        Strategy: 'mean', 'median', 'most_frequent'
        """
        super().__init__(df)
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=strategy)

    def run(self) -> pd.DataFrame:
        numeric_col = self._df.select_dtypes(include=np.number).columns
        skew_score = [pd.Series.skew(col) for col in numeric_col]
        #for col, score in enumerate(numeric_col, skew_score):
            #if abs(score) > 1:
                #self._df[col] = np.log1p(self._df[col])
        return self._df


# ---------------------- Fix Skewness ----------------------


# ---------------------- Encoding Categorical Features ----------------------



# ---------------------- Scaling ----------------------



# ---------------------- Date-time feature extraction ----------------------
