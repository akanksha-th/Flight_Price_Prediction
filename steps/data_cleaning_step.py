import pandas as pd
from src.data_cleaning import PreEDALightCleaner, PostEDAFullCleaner
from typing import Dict, Any
from zenml import step


@step
def basic_data_cleaning(df: pd.DataFrame)-> Dict[str, Any]:
    """ZenML step to clean data before EDA."""
    cleaner = PreEDALightCleaner(df)
    cleaned_df = cleaner.run()
    report = cleaner.info_summary(cleaned_df)
    return {"cleaned_data": cleaned_df, "summary": report}

@step
def final_data_cleaning():
    pass