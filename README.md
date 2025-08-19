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