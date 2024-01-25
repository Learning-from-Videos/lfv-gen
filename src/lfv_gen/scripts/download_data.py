""" Script to download data

Example usage:
python src/lfv_gen/scripts/download_data.py
"""

import gdown
import pathlib
from lfv_gen.data.dataset_zoo import R3M_DATASETS

DATASETS_DIR = pathlib.Path("datasets")


def try_download(id: str, output: str):
    try:
        gdown.download(id=id, output=output, quiet=False)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    for dataset in R3M_DATASETS:
        output_path = DATASETS_DIR / dataset.path
        if output_path.exists():
            print(f"Skipping {dataset.path} because it already exists")
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try_download(id=dataset.gdrive_id, output=str(output_path))
