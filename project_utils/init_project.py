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

    "analysis/analyze_src/__init__.py",
    "analysis/analyze_src/univariate_analysis.py",
    "analysis/analyze_src/bivariate_analysis.py",
    "analysis/analyze_src/multivariate_analysis.py",
    "analysis/analyze_src/missing_value_analysis.py",
    "analysis/analyze_src/model_diagnostics.py",
    "analysis/eda_and_fe.ipynb",

    "src/__init__.py",
    "src/data_ingester.py",
    "src/data_cleaning.py",
    "src/feature_transformation.py",
    "src/feature_selection.py",
    "src/data_splitter.py",
    "src/model_building.py",

    "steps/__init__.py",
    "steps/data_ingester_step.py",
    "steps/data_cleaning_step.py",
    "steps/feature_transformation_step.py",
    "steps/feature_selection_step.py",
    "steps/data_splitter_step.py",
    "steps/model_building_step.py",
    "steps/model_evaluator_step.py",

    "pipelines/__init__.py",
    "pipelines/training_pipeline.py",
    "pipelines/deployment_pipeline.py",
    
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
        

gitignore_content = """# Python
__pycache__/
*.py[cod]
*.egg-info/
.env
.venv/
.ipynb_checkpoints/

# Data
data/
!data/.gitkeep

# Logs
*.log

# OS
.DS_Store
Thumbs.db
"""
# Step 2: Initialize Git repo if not already
if not Path(".git").exists():
    subprocess.run(["git", "init"], check=True)
    logging.info("Initialized a new Git repository.")
else:
    logging.info("Git repository already initialized.")

# Step 3: Populate .gitignore
gitignore_path = Path(".gitignore")
if not gitignore_path.exists() or gitignore_path.stat().st_size == 0:
    with open(gitignore_path, "w") as f:
        f.write(gitignore_content.strip() + "\n")
    logging.info("Populated .gitignore with Python template.")
