import pandas as pd
import numpy as np
from skimpy import skim
import re
from abc import ABC, abstractmethod
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler


class DataCleaner(ABC):
    """
    Abstract base class for all data cleaners.
    Enforces the implementation of the `run` method.
    """
    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        self._original_df = df.copy()

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
        return (
            self
            ._standardize_column_names()
            ._convert_obvious_dtypes()
            ._drop_empty_columns()
            ._drop_duplicates()
            ._df
        )
    
    # ---------- Internal cleaning steps ---------- 
    def _standardize_column_names(self):
        self._df.columns = [
            re.sub(r'[^\w\s]', '', col.strip().lower()).replace(' ', '_')
            for col in self._df.columns
        ]
        return self

    def _convert_obvious_dtypes(self):
        for col in self._df.columns:
            if self._df[col].dtype == object:
                # Try datetime
                self._df[col] = pd.to_datetime(self._df[col], errors='ignore')
                # Try numeric
                if not pd.api.types.is_datetime64_any_dtype(self._df[col]):
                    self._df[col] = pd.to_numeric(self._df[col], errors='ignore')
        return self

    def _drop_empty_columns(self):
        self._df.dropna(axis=1, how='all', inplace=True)
        return self

    def _drop_duplicates(self):
        self._df.drop_duplicates(inplace=True)
        return self

    # ---------- Utility Methods ---------- 
    @staticmethod
    def info_summary(df: pd.DataFrame):
        """Quick skimpy summary of the DataFrame."""
        return skim(df)
    

class PostEDAFullCleaner(DataCleaner):
    """
    Performs final cleaning after EDA: imputation, outlier handling, encoding
    Also, generates transformation diagnostics comparing original vs transformed data.
    """

    def __init__(self, df: pd.DataFrame,
                 impute_strategy_num='mean',
                 impute_strategy_cat='most frequent',
                 encoding='onehot',
                 scaling='standard',
                 outlier_method='zscore',
                 z_thresh=3):
        super().__init__(df)
        self.impute_strategy_num = impute_strategy_num
        self.impute_strategy_cat = impute_strategy_cat
        self.encoding = encoding
        self.scaling = scaling
        self.outlier_method = outlier_method
        self.z_thresh = z_thresh

    def run(self):
        self._handle_missing_values()
        self._encode_categoricals()
        self._scale_features()
        self._handle_outliers()
        report = self._generate_diagnostics_report()
        return self._df, report
    
    # ---------- Cleaning Methods ---------- 
    def _handle_missing_values(self):
        num_cols = self._df.select_dtypes(include='number').columns
        cat_cols = self._df.select_dtypes(exclude='number').columns

        if len(num_cols)>0:
            num_imputer = SimpleImputer(strategy=self.impute_strategy_num)
            self._df[num_cols] = num_imputer.fit_transform(self._df[num_cols])

        elif len(cat_cols)>0:
            cat_imputer = SimpleImputer(strategy=self.impute_strategy_cat)
            self._df[cat_cols] = cat_imputer.fit_transform(self._df[cat_cols])

    def _encode_categoricals(self):
        pass

    def _scale_features(self):
        pass

    def _handle_outliers(self):
        pass

    # ---------- Diagnostics ---------- 
    def _generate_diagnostics_report(self):
        pass



if __name__ == "__main__":
    df = pd.read_csv("path-to-the-csv-data-file")
    PreEDALightCleaner.info_summary(df)
    cleaner = PreEDALightCleaner(df)
    cleaned_df = cleaner.run()