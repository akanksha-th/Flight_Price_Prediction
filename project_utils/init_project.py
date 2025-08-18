import os
from pathlib import Path
import logging
import subprocess

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: %(message)s')

logging.info("Creating Project Scaffold.")

list_of_files = [
    ".github/workflows/.gitkeep",
    "data/.gitkeep",              # creates the empty data/ folder
    "src/__init__.py",
    "src/data_ingester.py",
    "src/data_cleaning.py",
    "src/univariate_analysis.py",
    "src/bivariate_analysis.py",
    "src/multivariate_analysis.py",
    "src/feature_engineering.py",
    "src/data_splitter.py",
    "src/model_building.py",
    "src/model_diagnostics.py",
    "steps/__init__.py",
    "steps/data_ingester_step.py",
    "steps/data_cleaning_step.py",
    "steps/feature_engineering_step.py",
    "steps/data_splitter_step.py",
    "steps/model_building_step.py",
    "steps/model_evaluator.py",
    "pipeline/__init__.py",
    "pipeline/training_pipeline.py",
    "pipeline/deployment_pipeline.py",
    ".gitignore",
    "requirements.txt",
    "setup.py",
    "README.md"
]

# Step 1: Create basic scaffold
for filepath in list_of_files:
    # Convert string to Path object
    filepath = Path(filepath) 
    filedir, filename = os.path.split(filepath) 
    
    if filedir:
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir}")
        
    if (not os.path.exists(filepath) or os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass # Create the file with a placeholder content
        logging.info(f"Created File: {filename}")
    else:
        logging.info(f"File {filepath} already exists and is not empty.")
        
# Step 2: Initialize Git repo if not already
if not Path(".git").exists():
    subprocess.run(["git", "init"], check=True)
    logging.info("Initialized a new Git repository.")
else:
    logging.info("Git repository already initialized.")
