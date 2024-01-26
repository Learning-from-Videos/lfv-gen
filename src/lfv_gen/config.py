import pathlib
import os

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
DATASETS_DIR = pathlib.Path(os.getenv("DATASETS_DIR", "datasets"))
JAM_MODEL_DIR = pathlib.Path(os.getenv("JAM_MODEL_DIR", "data/models"))
