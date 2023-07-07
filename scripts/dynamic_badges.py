#%% Imports -------------------------------------------------------------------

import urllib.parse
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
WORKFLOWS_PATH = ROOT_PATH / '.github' / 'workflows'

#%% 

def extract_python_versions():
    
    # Read pytest.yml
    with open(WORKFLOWS_PATH / 'pytest.yml', "r") as file:
        content = file.read()
    
    # Extract tested Python version(s)
    idx0 = content.find("python-version: [") + len("python-version: [")
    idx1 = content.find("]", idx0)
    version_str = content[idx0:idx1]
    version_list = version_str.split(', ')
    version_list = [item.replace("'", "") for item in version_list]
    
    return version_list

def generate_python_badge(version_list):
    version_list = [version.replace('-', '_') for version in version_list]
    version_str = ' | '.join(version_list)
    url_encoded_versions = urllib.parse.quote(version_str)
    badge_md = f'![Python Badge](https://img.shields.io/badge/Python-{url_encoded_versions}-blue?logo=python&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))'
    
    return badge_md

def update_python_badge(readme_path, badge_md):
    with open(readme_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if '![Python Badge]' in line:
            lines[i] = badge_md + '\n'

    with open(readme_path, 'w') as file:
        file.writelines(lines)

#%%

version_list = extract_python_versions()
badge_md = generate_python_badge(version_list)
update_python_badge(ROOT_PATH / 'README.md', badge_md)