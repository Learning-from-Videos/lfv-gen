""" Script to print the infos of R3M demonstrations 

Example usage:
 
python src/lfv_gen/scripts/print_infos.py > output.txt

Then you can copy-paste the output to src/lfv_gen/data/dataset_zoo.py
"""

import itertools

suites = (
    'metaworld',
)

viewpoints = (
    'left',
    'right',
    'top',
)

template = """
    R3MDemonstrationDataset(
        suite='{suite}',
        viewpoint='{viewpoint}',
        name='{name}',
        gdrive_id='{gdrive_id}',
    ),
""".strip('\n')

if __name__ == "__main__":
    for suite, viewpoint in itertools.product(suites, viewpoints):
        filepath = f"src/lfv_gen/data/infos/{suite}_{viewpoint}_cap2.txt"
        with open(filepath, 'r') as f:
            lines = f.read().splitlines()
            for line in lines[1:]: # Remove the header
                elements = list(filter(None, line.split(' '))) # Remove empty strings
                id, name = elements[:2]
                print(template.format(
                    suite=suite,
                    viewpoint=viewpoint,
                    name=name,
                    gdrive_id=id,
                ))