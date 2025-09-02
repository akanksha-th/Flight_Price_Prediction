from zenml import step
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


@step
def model_evaluator(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    ZenML step for evaluating a trained model with standard metrics.

    Args:
        model: Trained model to evaluate.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    # If probabilities available, compute AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except Exception:
            pass

    return metrics
