import os
from pathlib import Path
import logging
import subprocess

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: %(message)s')

init_script_path = Path("project_utils/init_project.py")
init_script_path.parent.mkdir(parents=True, exist_ok=True)
logging.info("Created project_utils/init_project.py")

list_of_files = [
    ".github/workflows/.gitkeep",
    "project_utils/init_project.py",
    "project_utils/fetching_data.py",
    "data/.gitkeep",              # creates the empty data/ folder
    ".gitignore",
    ".dvcignore",
    "Dockerfile",
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

# Step 3: Initialize DVC if not already
if not Path(".dvc").exists():
    subprocess.run(["dvc", "init"], check=True)
    logging.info("Initialized DVC in the project.")
else:
    logging.info("DVC already initialized.")

