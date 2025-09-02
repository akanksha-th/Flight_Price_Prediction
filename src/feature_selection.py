from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import shap
import lime
import lime.lime_tabular

# ---------------------- Config ----------------------
@dataclass
class FeatureSelectionConfig:
    method: str = "variance"  # 'variance', 'correlation', 'model'
    threshold: float = 0.01   # for variance or correlation
    model_type: str = "random_forest"  # for model-based selection
    n_estimators: int = 100
    random_state: int = 42

# ---------------------- Base Class ----------------------
class BaseFeatureSelector(ABC):
    def __init__(self, X: pd.DataFrame, y: pd.Series, config: FeatureSelectionConfig):
        self.X = X.copy()
        self.y = y.copy()
        self.config = config
        self.selected_features = None

    @abstractmethod
    def select_features(self) -> pd.DataFrame:
        pass

# ---------------------- Variance Threshold ----------------------
class VarianceSelector(BaseFeatureSelector):
    def select_features(self):
        selector = VarianceThreshold(threshold=self.config.threshold)
        selector.fit(self.X)
        self.selected_features = self.X.columns[selector.get_support()].tolist()
        return self.X[self.selected_features]

# ---------------------- Correlation-based ----------------------
class CorrelationSelector(BaseFeatureSelector):
    def select_features(self):
        corr_matrix = self.X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self.config.threshold)]
        self.selected_features = [col for col in self.X.columns if col not in to_drop]
        return self.X[self.selected_features]

# ---------------------- Model-based ----------------------
class ModelBasedSelector(BaseFeatureSelector):
    def select_features(self):
        if self.config.model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=self.config.n_estimators,
                                          random_state=self.config.random_state)
        elif self.config.model_type == "gradient_boost":
            model = GradientBoostingRegressor(n_estimators=self.config.n_estimators,
                                              random_state=self.config.random_state)
        else:
            raise ValueError("Unsupported model type")

        model.fit(self.X, self.y)
        importance = pd.Series(model.feature_importances_, index=self.X.columns)
        self.selected_features = importance[importance > importance.mean()].index.tolist()
        return self.X[self.selected_features]

# ---------------------- Factory ----------------------
class FeatureSelectorFactory:
    @staticmethod
    def get_selector(X, y, config: FeatureSelectionConfig):
        method_map = {
            "variance": VarianceSelector,
            "correlation": CorrelationSelector,
            "model": ModelBasedSelector
        }
        if config.method not in method_map:
            raise ValueError(f"Unknown selection method: {config.method}")
        return method_map[config.method](X, y, config)

# ---------------------- SHAP + LIME Explanation ----------------------
class ModelExplainer:
    def __init__(self, model, X_train: pd.DataFrame):
        self.model = model
        self.X_train = X_train

    def shap_summary(self):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_train)
        shap.summary_plot(shap_values, self.X_train)
        return shap_values

    def lime_local(self, X_row: pd.DataFrame):
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.X_train.columns.tolist(),
            class_names=['target'],
            verbose=True,
            mode='regression'
        )
        exp = explainer.explain_instance(X_row.values[0], self.model.predict)
        exp.show_in_notebook(show_table=True)
        return exp

# ---------------------- Example ----------------------
if __name__ == "__main__":
    df = pd.DataFrame({
        'distance': [100, 200, 300, 400, 500],
        'price': [50, 80, 60, 90, 100],
        'delay': [5, 15, 10, 20, 25],
        'airline_factor': [1, 0, 1, 2, 1]
    })
    y = pd.Series([1, 0, 1, 0, 1])

    # Feature selection
    config = FeatureSelectionConfig(method='model', model_type='random_forest')
    selector = FeatureSelectorFactory.get_selector(df, y, config)
    df_selected = selector.select_features()
    print("Selected Features:", selector.selected_features)

    # Model explanation
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(df_selected, y)
    explainer = ModelExplainer(model, df_selected)
    shap_values = explainer.shap_summary()  # global
    explainer.lime_local(df_selected.iloc[[0]])  # local explanation
