from zenml import Model, pipeline

from steps import (
    data_ingestion,
    basic_data_cleaning,
    feature_transformation,
    feature_selection,
    split_data,
    model_building,
    model_training,
    model_evaluator,
)


@pipeline(
    model=Model(
        name="flight_price_predictor"  # Uniquely identifies this model in ZenML
    ),
)
def ml_pipeline():
    raw_data = data_ingestion(
        file_path="C:/Users/aktkr/Flight_Price_Prediction/data/flight-price-prediction/Clean_Dataset.csv"
    )

    cleaned_data = basic_data_cleaning(raw_data)
    
    transformed_data = feature_transformation(cleaned_data)

    selected_data, selected_features = feature_selection(
        transformed_data, target_col="Price"   # <--- replace with your actual target column
    )

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        selected_data, target_col="Price"
    )

    models = model_building(X_train, y_train)

    best_model, best_params = model_training(
        X_train, y_train, X_val, y_val, models
    )
    metrics = model_evaluator(best_model, X_test, y_test)

    return best_model, best_params, metrics


if __name__ == "__main__":
    run = ml_pipeline()
