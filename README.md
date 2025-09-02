# Flight Price Prediction - End-to-End ML Pipeline

This project implements a complete end-to-end machine learning workflow for predicting flight prices. It has been trained on a smaller kaggle dataset, and showcases how to structure reproducible ML pipelines locally using ZenML and track experiments with MLflow.

---

## Project Structure
    .
    │   .gitignore
    │   README.md
    │   requirements.txt
    │   run_deployment.py
    │   run_pipeline.py
    │   setup.py
    │
    ├───.github
    │   └───workflows
    │           .gitkeep
    │
    ├───analysis
    │   │   eda_and_fe.ipynb
    │   │
    │   └───analyze_src
    │           bivariate_analysis.py
    │           missing_value_analysis.py
    │           multivariate_analysis.py
    │           univariate_analysis.py
    │           model_diagnostics.py
    │           __init__.py
    │
    ├───data
    │   │   .gitkeep
    │   │
    │   └───flight-price-prediction
    │           business.csv
    │           Clean_Dataset.csv
    │           economy.csv
    │
    ├───pipelines
    │       deployment_pipeline.py
    │       training_pipeline.py
    │       __init__.py
    │
    ├───project_utils
    │       init_project.py
    │
    ├───src
    │       data_cleaning.py
    │       data_ingester.py
    │       data_splitter.py
    │       feature_transformation.py
    │       model_building.py
    │       __init__.py
    │
    └───steps
            data_cleaning_step.py
            data_ingester_step.py
            data_splitter_step.py
            feature_transformation_step.py
            model_building_step.py
            model_evaluator_step.py
            __init__.py

---

## Project Highlights

- Local Development: lightweight, easy to run on personal machine.
- ZenML: reproducible ML pipeline (data ingestion → preprocessing → model training → evaluation).
- MLflow: experiment tracking, metrics logging, and model versioning.
- Use Case: regression problem — predict flight ticket prices based on input features (e.g., airline, source, destination, duration, stops).

---

## Tech Stack

    Python
    ZenML (pipeline orchestration)
    MLflow (tracking & registry)
    Scikit-learn / XGBoost (models)
    Pandas, Numpy, Matplotlib (data + viz)

---

## How to Run

# Clone repo
git clone <repo-url>
cd flight-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python run_pipeline.py

# Start MLflow UI
mlflow ui
