#%% Imports -------------------------------------------------------------------

import urllib.parse
from pathlib import Path

ROOT_PATH = Path('.').resolve()
# ROOT_PATH = Path('.').resolve().parents[0] # local execution
SCRIPTS_PATH = ROOT_PATH / 'scripts'
WORKFLOWS_PATH = ROOT_PATH / '.github' / 'workflows'

#%% 

def extract_os_versions():
    
    # Read pytest.yml
    with open(WORKFLOWS_PATH / 'pytest.yml', "r") as file:
        content = file.read()
    
    # Extract os and version list
    idx0 = content.find("os: [") + len("os: [")
    idx1 = content.find("]", idx0)
    version_str = content[idx0:idx1]
    tmp_list = version_str.split(', ')
    tmp_list = [item.replace("'", "") for item in tmp_list]
    
    os_list = []
    os_version_list = []    
    for item in tmp_list:
        parts = item.split('-')
        os_list.append(parts[0])
        os_version_list.append(parts[1])
    
    return os_list, os_version_list

def generate_os_badges(os_list, os_version_list):
    
    badges_md = []
    if 'ubuntu' in os_list:
        idx = os_list.index('ubuntu')
        badges_md.append((
            f'![Ubuntu Badge]'
            f'(https://img.shields.io/badge/Ubuntu-{os_version_list[idx]}-blue?'
            f'logo=ubuntu&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))'
        ))
    if 'windows' in os_list:
        idx = os_list.index('windows')
        badges_md.append((
            f'![Windows Badge]'
            f'(https://img.shields.io/badge/Windows-{os_version_list[idx]}-blue?'
            f'logo=windows11&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))'
        ))
    if 'macos' in os_list:
        idx = os_list.index('macos')
        badges_md.append((
            f'![MacOS Badge]'
            f'(https://img.shields.io/badge/MacOS-{os_version_list[idx]}-blue?'
            f'logo=apple&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))'
        ))
    
    return badges_md

def update_os_badges(badges_md):
    
    # Read BADGES.md
    with open(SCRIPTS_PATH / 'BADGES.md', 'r') as file:
        lines = file.readlines()
    
    for i, badge_md in enumerate(badges_md):
        lines.insert(i+2, badge_md  + '\n')
        
    with open(SCRIPTS_PATH / 'BADGES.md', 'w') as file:
        file.writelines(lines)

os_list, os_version_list = extract_os_versions()
badges_md = generate_os_badges(os_list, os_version_list)
update_os_badges(badges_md)

#%%

# def update_os_badge(readme_path, badges_md):
#     with open(readme_path, 'r') as file:
#         lines = file.readlines()

#     for i, line in enumerate(lines):
#         if '![OS Badge]' in line:
#             lines[i] = badges_md + '\n'

#     with open(readme_path, 'w') as file:
#         file.writelines(lines)

# def extract_python_versions():
    
#     # Read pytest.yml
#     with open(WORKFLOWS_PATH / 'pytest.yml', "r") as file:
#         content = file.read()
    
#     # Extract tested Python version(s)
#     idx0 = content.find("python-version: [") + len("python-version: [")
#     idx1 = content.find("]", idx0)
#     version_str = content[idx0:idx1]
#     os_version_list = version_str.split(', ')
#     os_version_list = [item.replace("'", "") for item in os_version_list]
    
#     return os_version_list

# def generate_python_badge(os_version_list):
#     os_version_list = [version.replace('-', '_') for version in os_version_list]
#     version_str = ' | '.join(os_version_list)
#     url_encoded_versions = urllib.parse.quote(version_str)
#     badges_md = f'![Python Badge](https://img.shields.io/badge/Python-{url_encoded_versions}-blue?logo=python&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))'
    
#     return badges_md

# def update_python_badge(readme_path, badges_md):
#     with open(readme_path, 'r') as file:
#         lines = file.readlines()

#     for i, line in enumerate(lines):
#         if '![Python Badge]' in line:
#             lines[i] = badges_md + '\n'

#     with open(readme_path, 'w') as file:
#         file.writelines(lines)

#%%

# os_list, os_version_list = extract_os_versions()
# badges_md = generate_os_badges(os_list, os_version_list)
# update_os_badge(ROOT_PATH / 'README.md', badges_md)

# os_version_list = extract_python_versions()
# badges_md = generate_python_badge(os_version_list)
# update_python_badge(ROOT_PATH / 'README.md', badges_md)