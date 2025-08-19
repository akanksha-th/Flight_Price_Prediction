from zenml import Model, pipeline

from steps.data_ingester_step import data_ingestion
from steps.data_cleaning_step import basic_data_cleaning, final_data_cleaning
from steps.feature_transformation_step import feature_transformation
from steps.data_splitter_step import split_data
from steps.model_building_step import model_building
from steps.model_evaluator_step import model_evaluator

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