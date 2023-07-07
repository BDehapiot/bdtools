#%% Imports -------------------------------------------------------------------

from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
WORKFLOWS_PATH = ROOT_PATH / '.github' / 'workflows'

#%% 

def extract_list():
    
    # Read pytest.yml
    with open(WORKFLOWS_PATH / 'pytest.yml', "r") as file:
        content = file.read()
    
    # Extract tested OS 
    idx0 = content.find("os: [") + len("os: [")
    idx1 = content.find("]", idx0)
    os_str = content[idx0:idx1]
    os_list = os_str.split(', ')
    
    # # Extract tested Python version(s)
    idx0 = content.find("python-version: [") + len("python-version: [")
    idx1 = content.find("]", idx0)
    python_str = content[idx0:idx1]
    python_list = python_str.split(', ')
    python_list = [item.replace("'", "") for item in python_list]
    
    return os_list, python_list

def generate_badge_urls(os_list, python_list):
    
    merged_python_str = 
    
    python_badge_urls = [
        f"https://img.shields.io/badge/Python-{item}-blue" 
        for item in python_list
        ]
    os_badge_urls = [
        f"https://img.shields.io/badge/OS-{item}-blue" 
        for item in os_list
        ]
    
    return os_badge_urls, python_badge_urls

# def update_readme_badges(python_badge_urls):
    
#     with open(ROOT_PATH / 'README.md', "a") as file:
#         file.write("\n\n")
#         for url in python_badge_urls:
#             file.write(f"[![Python Version]({url})]({url}) ")
#         file.write("\n")
        
# def update_readme_badges(badge_urls):
#     formatted_badges = " | ".join(f"[![Badge]({url})]({url})" for url in badge_urls)
    
#     with open(ROOT_PATH / 'README.md', "a") as file:
#         file.write("\n\n")
#         file.write(formatted_badges)
#         file.write("\n")

# def update_readme_badges(badge_urls):
#     python_versions_merged = " | ".join(python_versions)
#     badge_url = f"https://img.shields.io/badge/Python-{python_versions_merged}-blue"

#     with open(ROOT_PATH / 'README.md', "a") as file:
#         file.write("\n\n")
#         file.write(f"[![Python Versions]({badge_url})]({badge_url})")
#         file.write("\n")

#%%

os_list, python_list = extract_list()
os_badge_urls, python_badge_urls = generate_badge_urls(os_list, python_list)
# update_readme_badges(python_badge_urls)