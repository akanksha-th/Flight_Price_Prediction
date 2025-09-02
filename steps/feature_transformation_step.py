from zenml import step
import pandas as pd

# Import your transformation modules
from src.feature_transformations.handling_missing_data import ImputerConfig, ImputeFactory
from src.feature_transformations.scaling import ScalerConfig, ScalerFactory
from src.feature_transformations.encoding_cat import EncoderConfig, EncoderFactory
from src.feature_transformations.dimensionality_reduction import DimReducerConfig, DimReducerFactory


@step
def feature_transformation(
    df: pd.DataFrame,
    impute_method: str = "simple",
    impute_strategy: str = "mean",
    scaling_method: str = "standard",
    encoding_method: str = "onehot",
    dimred_method: str = None,
    drop_threshold: float = 0.5,
    n_components: int = 2
) -> pd.DataFrame:
    """
    ZenML step for complete feature transformation pipeline:
    1. Missing value imputation
    2. Scaling numeric features
    3. Encoding categorical features
    4. Dimensionality reduction (optional)

    Args:
        df: Input dataset (train or raw).
        impute_method: "simple", "knn", "iterative", "matrix_completion".
        impute_strategy: "mean", "median", "most_frequent" (for simple).
        scaling_method: "standard", "minmax", "robust", "quantile", etc.
        encoding_method: "onehot", "ordinal", "target", etc.
        dimred_method: "pca", "umap", "tsne", "autoencoder", or None.
        drop_threshold: Drop columns with missing ratio above this threshold.
        n_components: Number of dimensions to reduce to (if dimred_method is set).

    Returns:
        Transformed DataFrame.
    """

    # ---- Missing Value Handling ----
    impute_config = ImputerConfig(method=impute_method, strategy=impute_strategy, drop_threshold=drop_threshold)
    imputer = ImputeFactory.get_imputer(df, impute_config)
    df = imputer.run()

    # ---- Scaling ----
    scale_config = ScalerConfig(method=scaling_method)
    scaler = ScalerFactory.get_scaler(df, scale_config)
    df = scaler.run()

    # ---- Encoding ----
    encode_config = EncoderConfig(method=encoding_method)
    encoder = EncoderFactory.get_encoder(df, encode_config)
    df = encoder.run()

    # ---- Dimensionality Reduction ----
    if dimred_method:
        dimred_config = DimReducerConfig(method=dimred_method, n_components=n_components)
        reducer = DimReducerFactory.get_reducer(df, dimred_config)
        df = reducer.run()

    return df
