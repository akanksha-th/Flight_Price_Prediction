from zenml import step
import pandas as pd
from typing import Dict, Any, Tuple
from src.model_training import ModelTrainer, TrainingConfig


@step
def model_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    models: Dict[str, Any],
    metric: str = "f1",
    n_trials: int = 20
) -> Tuple[Any, Dict[str, float]]:
    """
    ZenML step for model training with hyperparameter tuning.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        models: Dictionary of models from model_building step.
        metric: Optimization metric.
        n_trials: Number of hyperparameter tuning trials.

    Returns:
        - Best trained model
        - Dict of evaluation scores
    """
    config = TrainingConfig(
        metric=metric,
        n_trials=n_trials
    )

    trainer = ModelTrainer(config)
    best_model, scores = trainer.train_and_select(X_train, y_train, X_val, y_val, models)

    return best_model, scores
