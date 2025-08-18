import numpy as np
import pandas as pd

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, List, Iterable, Any

from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.compose import TransformedTargetRegressor


# ------------------ Config ------------------
@dataclass
class ModelConfig:
    random_state: int = 37
    target_log_transform: bool = True

    # Feature Overrides
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None

    # Cross-validation strategy
    cv_strategy: str = 'kfold'
    cv_splits: int = 5
    groups: Optional[Iterable[Any]] = None

    # Search/ optimization
    enable_model_selection: bool = True
    n_iter_Search: int = 30
    scoring: str = "neg_mean_absolute_error"
    n_jobs: int = -1

    model_path: str = "Trained_flight_price_model.joblib"


class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, config=None) -> RegressorMixin:
        pass


# ------------------ Strategies ------------------
class LinearModelsStrategy(ModelBuildingStrategy):
    def __init__(self, model: str = 'ridge'):
        assert model in {"linear", "ridge", "elasticnet"}, "Invalid model type"
        self.model_name = model

    def _get_estimator(self, config: ModelConfig):
        if self.model_name == 'linear':
            return LinearRegression()
        elif self.model_name == 'ridge':
            return Ridge(random_state=config.random_state, max_iter=10000)
        elif self.model_name == 'elasticnet':
            return ElasticNet(random_state=config.random_state, max_iter=10000)

    def build_and_train_model(self, X_train, y_train, config=None):
        cfg = config or ModelConfig()
        estimator = self._get_estimator(cfg)
        if cfg.target_log_transform:
            estimator = TransformedTargetRegressor(
                regressor=estimator, func=np.log1p, inverse_func=np.expm1
            )
        estimator.fit(X_train, y_train)
        return estimator


class TreeEnsembleStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train, y_train, config=None):
        pass


class ModelSeclectionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train, y_train, config=None):
        pass


# ------------------ Facade ------------------
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def build_model(self, X_train, y_train, config=None):
        return self._strategy.build_and_train_model(X_train, y_train, config)


# ------------------ Evaluation ------------------
def evaluate(model, X_val, y_val):
    pass

def save_model():
    pass

def load_model():
    pass


if __name__ == "__main__":
    # df = pd.read_csv("path-to-the-csv-file")
    # X = training features
    # y = target feature
    # builder = ModelBuilder(LinearModelsStrategy())
    # model = builder.build.model(X, y)
    pass