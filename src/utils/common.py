import yaml
import os
import logging

"""Used to perform common tasks
"""

def read_yaml(yaml_path):
    '''
    Read yaml file and return content of it
    '''
    with open(yaml_path, 'r') as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info('yaml file loaded')
    return content

def create_directories(path_to_dir: list):
    """Creates directories if they don't exist
    """
    full_path = ""
    for path in path_to_dir:
        full_path = os.path.join(full_path, path)
    os.makedirs(full_path, exist_ok=True)
    logging.info(f"Directories created: {full_path}")