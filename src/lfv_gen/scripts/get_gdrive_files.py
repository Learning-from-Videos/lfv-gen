""" Script to get file infos from Google Drive 

NOTE: This depends on the `gdrive` command-line utility. 
See details here: https://github.com/prasmussen/gdrive

Example usage: 

python src/lfv_gen/scripts/get_gdrive_files.py
"""

import subprocess

command_template = "gdrive files list --full-name --drive {id} > src/lfv_gen/data/infos/{name}.txt"
with open("src/lfv_gen/data/infos/gdrive_folders.txt", 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        name, id = line.split(' ')
        command = command_template.format(id=id, name=name)
        subprocess.run(command, shell=True)