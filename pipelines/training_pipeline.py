from zenml import Model, pipeline

from steps import (
    data_ingestion,
    basic_data_cleaning,
    final_data_cleaning,
    feature_transformation,
    split_data,
    model_building,
    model_evaluator,
)

@pipeline(
    model=Model(
        name="flight_price_predictor"   # The name uniquely identifies this model
        ),
)

def ml_pipeline():
    # Data Ignestion step
    raw_data = data_ingestion(
        file_path="C:/Users/aktkr/Flight_Price_Prediction/data/flight-price-prediction/Clean_Dataset.csv"
    )

    # Basic Data Cleaning
    transformed_data = basic_data_cleaning(raw_data)
    
    return transformed_data

if __name__ == "__main__":
    run = ml_pipeline()