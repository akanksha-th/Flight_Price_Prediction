# encoding.py

from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from abc import ABC, abstractmethod

@dataclass
class EncoderConfig:
    method: str = "onehot"            # 'label', 'onehot', 'target', 'frequency'
    columns: Optional[List[str]] = None
    target_col: Optional[str] = None  # required for target encoding

# ---------------------- Base Encoder ----------------------
class BaseEncoder(ABC):
    def __init__(self, df: pd.DataFrame, config: EncoderConfig):
        self._df = df.copy()
        self.config = config
        self.columns = config.columns or self._df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.encoders = {}  # store fitted encoders for later use

    @abstractmethod
    def fit_transform(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# ---------------------- Label Encoder ----------------------
class LabelEncoderClass(BaseEncoder):
    def fit_transform(self):
        for col in self.columns:
            le = LabelEncoder()
            self._df[col] = le.fit_transform(self._df[col].astype(str))
            self.encoders[col] = le
        return self._df

    def transform(self, df: pd.DataFrame):
        df_copy = df.copy()
        for col in self.columns:
            le = self.encoders[col]
            df_copy[col] = le.transform(df_copy[col].astype(str))
        return df_copy

# ---------------------- One-Hot Encoder ----------------------
class OneHotEncoderClass(BaseEncoder):
    def fit_transform(self):
        self._df = pd.get_dummies(self._df, columns=self.columns, drop_first=False)
        return self._df

    def transform(self, df: pd.DataFrame):
        df_copy = pd.get_dummies(df, columns=self.columns, drop_first=False)
        # Ensure same columns as train
        missing_cols = set(self._df.columns) - set(df_copy.columns)
        for col in missing_cols:
            df_copy[col] = 0
        df_copy = df_copy[self._df.columns]
        return df_copy

# ---------------------- Target Encoder ----------------------
class TargetEncoderClass(BaseEncoder):
    def fit_transform(self):
        if not self.config.target_col:
            raise ValueError("Target column must be specified for target encoding")
        for col in self.columns:
            mapping = self._df.groupby(col)[self.config.target_col].mean()
            self._df[col] = self._df[col].map(mapping)
            self.encoders[col] = mapping
        return self._df

    def transform(self, df: pd.DataFrame):
        df_copy = df.copy()
        for col in self.columns:
            mapping = self.encoders[col]
            df_copy[col] = df_copy[col].map(mapping).fillna(mapping.mean())
        return df_copy

# ---------------------- Frequency Encoder ----------------------
class FrequencyEncoderClass(BaseEncoder):
    def fit_transform(self):
        for col in self.columns:
            mapping = self._df[col].value_counts(normalize=True)
            self._df[col] = self._df[col].map(mapping)
            self.encoders[col] = mapping
        return self._df

    def transform(self, df: pd.DataFrame):
        df_copy = df.copy()
        for col in self.columns:
            mapping = self.encoders[col]
            df_copy[col] = df_copy[col].map(mapping).fillna(0)
        return df_copy

# ---------------------- Factory ----------------------
class EncoderFactory:
    @staticmethod
    def get_encoder(df: pd.DataFrame, config: EncoderConfig) -> BaseEncoder:
        method_map = {
            'label': LabelEncoderClass,
            'onehot': OneHotEncoderClass,
            'target': TargetEncoderClass,
            'frequency': FrequencyEncoderClass
        }
        if config.method not in method_map:
            raise ValueError(f"Unknown encoding method: {config.method}")
        return method_map[config.method](df, config)

# ---------------------- Example Usage ----------------------
if __name__ == "__main__":
    df = pd.DataFrame({
        'airline': ['A', 'B', 'A', 'C'],
        'source': ['X', 'Y', 'X', 'Z'],
        'target': [100, 200, 150, 300]
    })

    # Example: Target encoding
    config = EncoderConfig(method="target", columns=['airline', 'source'], target_col='target')
    encoder = EncoderFactory.get_encoder(df, config)
    df_encoded = encoder.fit_transform()
    print(df_encoded)

    # Transform test set
    df_test = pd.DataFrame({
        'airline': ['A', 'B', 'C'],
        'source': ['X', 'Z', 'Y'],
        'target': [120, 180, 250]
    })
    df_test_encoded = encoder.transform(df_test)
    print(df_test_encoded)
