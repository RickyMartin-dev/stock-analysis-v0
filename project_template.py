# Imports
import os
from pathlib import Path
import logging

# Define Logging Format
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# specify files to create
list_of_files = [
    ".env", # purely for local development
    ".github/workflows/.gitkeep",
    "experiments/testing.py", # for testing files if needed
    "experiments/artifacts/__init__.py", # for testing files if needed
    "requirements.txt", # for package information 
    "Dockerfile.inference",
    "Dockerfile.training",
    "tests/__init__.py",
    "tests/__init__.py",
    "src/__init__.py", # to import files
    "src/artifacts/__init__.py", # to import files
    "src/config.py", # to import files
    "src/utils.py", # utilies aka. repeated function
    "src/mlflow_utils.py", # load in ml flow functionality
    "src/schemas.py", # schema for access
    "src/data_extractor.py", # to extract data using yfinance
    "src/features.py", # to create and manage feature creation/engineering
    "src/model.py", # define model
    "src/train.py", # to train
    "src/drift.py", # define drift checks
    "src/inference_lambda.py", # lambda handler for online inference
    "src/train_lambda.py", # lambda handler for scheduled training
]

# Go through list and create folders/files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # logic for directory
    if filedir != "":
        # create directory
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    # Create the files
    # check if file exists I do not want to replace
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else: # if file already exists
        logging.info(f"{filename} already exists")