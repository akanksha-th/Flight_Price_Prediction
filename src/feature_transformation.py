import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from abc import ABC, abstractmethod
from dataclasses import dataclass


class MissingValueImputation:
    def __init__(self, df: pd.DataFrame) -> pd.DataFrame:
        self._df = df

    pass