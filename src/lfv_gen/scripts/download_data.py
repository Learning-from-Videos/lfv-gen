""" Script to download data 

Example usage:
python src/lfv_gen/scripts/download_data.py
"""

import gdown
import pathlib
from lfv_gen.data.dataset_zoo import R3M_DATASETS

DATASETS_DIR = pathlib.Path("datasets")

if __name__ == "__main__":
    for dataset in R3M_DATASETS:
        output_path = DATASETS_DIR / dataset.path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=dataset.gdrive_id, output=str(output_path), quiet=False)