import pandas as pd
import numpy as np
from skimpy import skim
import re
from abc import ABC, abstractmethod
import logging


class DataCleaner(ABC):
    """
    Abstract base class for all data cleaners.
    Enforces the implementation of the `run` method.
    """
    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        self._original_df = df.copy()

    def reset(self):
        self._df = self._original_df.copy()
        return self

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Run the cleaning process and return cleaned DataFrame."""
        pass


class PreEDALightCleaner(DataCleaner):
    """
    Performs 'basic cleaning' on the raw datasets to make them usable for EDA.
    Does NOT perform imputations, transformations or outlier removal that might bias EDA results.
    """

    def run(self) -> pd.DataFrame:
        if self._df is None:
            raise ValueError("DataFrame not found")
        
        self._standardize_column_names()
        self._convert_obvious_dtypes()
        self._drop_empty_columns()
        self._drop_duplicates()
        PreEDALightCleaner.info_summary(self._df)
            
        return self._df
    
    # ---------- Internal cleaning steps ---------- 
    def _standardize_column_names(self):
        self._df.columns = [
            re.sub(r'[^\w\s]', '', col.strip().lower()).replace(' ', '_')
            for col in self._df.columns
        ]
        logging.info(f"Standardized {len(self._df.columns)} column names")
        return self

    def _convert_obvious_dtypes(self):
        for col in self._df.columns:
            if self._df[col].dtype == object:
                try:
                    self._df[col] = pd.to_numeric(self._df[col])
                except ValueError:
                    sample_vals = self._df[col].dropna().astype(str).head(5).tolist()
                    if any(re.match(r"^\d{4}-\d{2}-\d{2}", v) for v in sample_vals):
                        self._df[col] = pd.to_datetime(self._df[col], errors='coerce')
        return self

    def _drop_empty_columns(self):
        self._df.dropna(axis=1, how='all', inplace=True)
        logging.info(f"Dropped empty columns, now shape={self._df.shape}")
        return self

    def _drop_duplicates(self):
        self._df.drop_duplicates(inplace=True)
        return self

    # ---------- Utility Methods ---------- 
    @staticmethod
    def info_summary(df: pd.DataFrame):
        """Quick skimpy summary of the DataFrame."""
        return skim(df)
    
logging.info("Data Cleaning done successfully. Data is ready for EDA.")

if __name__ == "__main__":
    # df = pd.read_csv("path-to-the-csv-data-file")
    # PreEDALightCleaner.info_summary(df)
    # cleaner = PreEDALightCleaner(df)
    # cleaned_df = cleaner.run()
    pass