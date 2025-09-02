import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from fancyimpute import SoftImpute
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ImputerConfig:
    method: str = "simple"
    strategy: str = "median"
    n_neighbours: int = 5   # for KNN Imputer
    max_iter: int = 10    # for Iterative Imputer
    random_state: int = 17
    add_missing_indicators: bool = True
    numeric_cols: Optional[List[str]] = None
    categorical_cols: Optional[List[str]] = None
    drop_threshold: float = 0.5  # Drop columns with missing ratio above 50%


class BaseImputer(ABC):
    def __init__(self, df: pd.DataFrame, config: ImputerConfig):
        self._df = df
        self.config = config
        self._imputer = None
        self._drop_highmissing_columns()

    @abstractmethod
    def run(self) -> pd.DataFrame:
        pass

    def _get_numeric_columns(self):
        return self.config.numeric_cols or self._df.select_dtypes(include=np.number).columns.tolist()
    
    def _get_categorical_columns(self):
        return self.config.categorical_cols or self._df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def _drop_highmissing_columns(self):
        missing_pct = self._df.isnull().mean()
        cols_to_drop = missing_pct[missing_pct > self.config.drop_threshold].index.tolist()
        if cols_to_drop:
            print(f"Dropping columns with > {self.config.drop_threshold*100:.0f}% missing values: {cols_to_drop}")
            self._df.drop(columns=cols_to_drop, inplace=True)


class SimpleImputerClass(BaseImputer):
    def __init__(self, df: pd.DataFrame, config: ImputerConfig):
        super().__init__(df, config)
        
    def run(self) -> pd.DataFrame:
        numeric_cols = self._get_numeric_columns()
        if numeric_cols:
            num_imputer = SimpleImputer(strategy=self.config.strategy, add_indicator=self.config.add_missing_indicators)
            imputed_numeric = num_imputer.fit_transform(self._df[numeric_cols])
            imputed_numeric_df = pd.DataFrame(imputed_numeric, 
                                              columns=numeric_cols, 
                                              index=self._df.index)
            self._df[numeric_cols] = imputed_numeric_df[numeric_cols]

        categorical_cols = self._get_categorical_columns()
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent", add_indicator=self.config.add_missing_indicators)
            self._df[categorical_cols] = pd.DataFrame(
                cat_imputer.fit_transform(self._df[categorical_cols]),
                columns=categorical_cols,
                index=self._df.index
            )
        return self._df
    

class KnnImputer(BaseImputer):
    def __init__(self, df: pd.DataFrame, config: ImputerConfig):
        super().__init__(df, config)
        self._imputer = KNNImputer(n_neighbors=self.config.n_neighbours, add_indicator=self.config.add_missing_indicators)

    def run(self) -> pd.DataFrame:
        numeric_cols = self._get_numeric_columns()
        if numeric_cols:
            self._df[numeric_cols] = self._imputer.fit_transform(self._df[numeric_cols])
        return self._df
    

class IterativeImputerClass(BaseImputer):
    def __init__(self, df, config):
        super().__init__(df, config)
        self._imputer = IterativeImputer(max_iter=self.config.max_iter, random_state=self.config.random_state, add_indicator=self.config.add_missing_indicators)

    def run(self) -> pd.DataFrame:
        numeric_cols = self._get_numeric_columns()
        categorical_cols = self._get_categorical_columns()
        
        encoders = {}
        if categorical_cols:
            for col in categorical_cols:
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                self._df[[col]] = encoder.fit_transform(self._df[[col]])
                encoders[col] = encoder

        iterative_imputer = IterativeImputer(max_iter=self.config.max_iter, random_state=self.config.random_state)
        self._df[numeric_cols + categorical_cols] = iterative_imputer.fit_transform(self._df[numeric_cols + categorical_cols])

        # Decode categorical back to original
        if categorical_cols:
            for col in categorical_cols:
                self._df[col] = encoders[col].inverse_transform(self._df[[col]])
        return self._df


class MatrixCompletionImputer(BaseImputer):
    def run(self) -> pd.DataFrame:
        numeric_cols = self._get_numeric_columns()
        if numeric_cols:
            mat = self._df[numeric_cols].values
            mat_filled = SoftImpute().fit_transform(mat)
            self._df[numeric_cols] = mat_filled
        return self._df


class ImputeFactory:
    @staticmethod
    def get_imputer(df:pd.DataFrame, config: ImputerConfig) -> BaseImputer:
        if config.method == "simple":
            return SimpleImputerClass(df, config)
        elif config.method == "knn":
            return KnnImputer(df, config)
        elif config.method == "iterative":
            return IterativeImputerClass(df, config)
        elif config.method == "matrix_completion":
            return MatrixCompletionImputer(df, config)
        else:
            raise ValueError(f"Unknown imputation method: {config.method}")
        


if __name__ == "__main__":
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': ['a', 'b', 'c', np.nan],
        'C': [np.nan, 2, 3, 4]
    })

    config = ImputerConfig(method="simple", strategy="mean")
    imputer = ImputeFactory.get_imputer(df, config)
    df_imputed = imputer.run()
    print(df_imputed)