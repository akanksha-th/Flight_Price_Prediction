import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, 
    Normalizer, PowerTransformer, QuantileTransformer
    )
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, List

@dataclass
class ScalerConfig:
    method: str = "standard"
    columns: Optional[List[str]] = None
    norm: str = "l2"    # for Normalizer: l1, l2, max
    power_method: str = 'yeo-johnson'   # For PowerTransformer
    quantile_output_distribution: str = "uniform"   # for QuantileTransformer
    copy: bool = True

class BaseScaler(ABC):
    def __init__(self, df: pd.DataFrame, config: ScalerConfig):
        self._df = df.copy()
        self.config = config
        self._scaler = None
        self.columns = self.config.columns or self._df.select_dtypes(include=np.number).columns.tolist()

    @abstractmethod
    def run(self):
        self._df[self.columns] = self._scaler.fit_transform(self._df[self.columns])
        return self._df
    

class StandardScalerClass(BaseScaler):
    def __init__(self, df, config: ScalerConfig):
        super().__init__(df, config)
        self._scaler = StandardScaler(copy=self.config.copy)

    def run(self):
        self._df[self.columns] = self._scaler.fit_transform(self._df[self.columns])
        return self._df
    
class MinMaxScalerClass(BaseScaler):
    def __init__(self, df, config: ScalerConfig):
        super().__init__(df, config)
        self._scaler = MinMaxScaler(copy=config.copy)

    def run(self):
        self._df[self.columns] = self._scaler.fit_transform(self._df[self.columns])
        return self._df

class RobustScalerClass(BaseScaler):
    def __init__(self, df, config: ScalerConfig):
        super().__init__(df, config)
        self._scaler = RobustScaler(copy=config.copy)

    def run(self):
        self._df[self.columns] = self._scaler.fit_transform(self._df[self.columns])
        return self._df

class MaxAbsScalerClass(BaseScaler):
    def __init__(self, df, config: ScalerConfig):
        super().__init__(df, config)
        self._scaler = MaxAbsScaler(copy=config.copy)

    def run(self):
        self._df[self.columns] = self._scaler.fit_transform(self._df[self.columns])
        return self._df

class NormalizerClass(BaseScaler):
    def __init__(self, df, config: ScalerConfig):
        super().__init__(df, config)
        self._scaler = Normalizer(norm=config.norm, copy=config.copy)

    def run(self):
        self._df[self.columns] = self._scaler.fit_transform(self._df[self.columns])
        return self._df

class PowerTransformerClass(BaseScaler):
    def __init__(self, df, config: ScalerConfig):
        super().__init__(df, config)
        self._scaler = PowerTransformer(method=config.power_method, copy=config.copy)

    def run(self):
        self._df[self.columns] = self._scaler.fit_transform(self._df[self.columns])
        return self._df

class QuantileTransformerClass(BaseScaler):
    def __init__(self, df, config: ScalerConfig):
        super().__init__(df, config)
        self._scaler = QuantileTransformer(output_distribution=config.quantile_output_distribution,
                                          copy=config.copy)

    def run(self):
        self._df[self.columns] = self._scaler.fit_transform(self._df[self.columns])
        return self._df
    

class ScalerFactory:
    @staticmethod
    def get_scaler(df:pd.DataFrame, config: ScalerConfig) -> BaseScaler:
        method_map = {
            "standard": StandardScalerClass,
            "minmax": MinMaxScalerClass,
            "robust": RobustScalerClass,
            "maxabs": MaxAbsScalerClass,
            "normalize": NormalizerClass,
            "power": PowerTransformerClass,
            "quantile": QuantileTransformerClass
        }
        if config.method not in method_map:
            raise ValueError(f"Unknown scaling method: {config.method}")
        return method_map[config.method](df, config)

if __name__ == "__main__":
    df = pd.DataFrame({
        'distance': [100, 200, 300, 400],
        'price': [50, 80, 60, 90],
        'delay': [5, 15, 10, 20]
    })

    config = ScalerConfig(method="normalize")
    scaler = ScalerFactory.get_scaler(df, config)
    df_scaled = scaler.run()
    print(df_scaled)