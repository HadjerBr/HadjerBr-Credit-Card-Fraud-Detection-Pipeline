import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

list_of_files = [
    "src/model_utils.py",
    "src/preprocess.py",
    "src/logistic_regression_scratch.py",
    "app/app.py",
    "notebook/trials.ipynb", 
    "requirements.txt",
    "src/__init__.py"

]


for filepath in list_of_files:
    filepath = Path(filepath) 
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0): # if the file does not exist, create an empty file
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} already exists") 
