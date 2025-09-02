from zenml import step
import pandas as pd
from typing import Dict, Any
from src.model_building import ModelConfig, ModelBuilderFactory


@step
def model_building(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    ensemble_method: str = "voting",   # "bagging", "boosting", "stacking"
    baseline_models: list = None
) -> Dict[str, Any]:
    """
    ZenML step for building baseline and ensemble models.

    Args:
        X_train: Training features.
        y_train: Training labels.
        ensemble_method: Type of ensemble to use.
        baseline_models: List of baseline model names.

    Returns:
        Dictionary with models {"baselines": ..., "ensemble": ...}
    """
    config = ModelConfig(
        ensemble_method=ensemble_method,
        baseline_models=baseline_models or ["logistic", "random_forest", "xgboost"]
    )

    builder = ModelBuilderFactory(config)
    models = builder.build_models(X_train, y_train)

    return models
