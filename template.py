import os 
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,
                     format="[%(asctime)s]: %(message)s")

list_of_file = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "research/trails.ipynb",
    "app.py",
    "store_index.py",
    "static",
    "templates/chat.html"
]

for file_path in list_of_file:
    file_path = Path(file_path)
    filedir, filename = os.path.split(file_path)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file {filename}")

    if (not os.path.exists(file_path)) or (os.path.getsize(filename) == 0):
        with open(file_path, "w") as f:
            pass
            logging.info(f"Creating empty file: {file_path}")
    else:
        logging.info(f"{filename} is already created")