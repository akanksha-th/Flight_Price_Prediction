from zenml import step
import pandas as pd
from typing import Tuple

from src.feature_selection import (
    FeatureSelectionConfig,
    FeatureSelectorFactory
)

@step
def feature_selection(
    df: pd.DataFrame,
    target_col: str,
    method: str = "lasso",
    n_features: int = 10,
    correlation_threshold: float = 0.9,
    model_based_estimator: str = "random_forest",
    explainability: bool = True
) -> Tuple[pd.DataFrame, list]:
    """
    ZenML step for feature selection with explainability support.

    Args:
        df: Input dataframe (already transformed).
        target_col: Name of the target column.
        method: Feature selection method: 
            "variance", "correlation", "rfe", "lasso", "random_forest", "xgboost".
        n_features: Number of features to keep.
        correlation_threshold: Threshold for correlation-based selection.
        model_based_estimator: Estimator for model-based selection.
        explainability: If True, run SHAP/LIME to validate feature importance.

    Returns:
        Tuple[pd.DataFrame, list]: 
            - Reduced dataframe with selected features.
            - List of selected feature names.
    """

    config = FeatureSelectionConfig(
        method=method,
        n_features=n_features,
        correlation_threshold=correlation_threshold,
        model_based_estimator=model_based_estimator,
        explainability=explainability
    )

    selector = FeatureSelectorFactory.get_selector(df, target_col, config)
    df_selected, selected_features = selector.run()

    return df_selected, selected_features
