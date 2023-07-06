#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io 
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_PATH / 'tests' / 'data' / 'skel'

from bdtools.skel import pixconn, labconn

#%% Initialize ----------------------------------------------------------------

skeleton = io.imread(DATA_PATH / 'skeleton.tif')
skeleton_noborder = io.imread(DATA_PATH / 'skeleton_noborder.tif')

#%% Test cases ----------------------------------------------------------------

test_cases = [
    
    {# Test case 00 - border & conn 1
        'img': skeleton, 'conn': 1,
    },
    {# Test case 01 - border & conn 2
        'img': skeleton, 'conn': 2,
    },    
    {# Test case 02 - noborder & conn 1
        'img': skeleton_noborder, 'conn': 1,
    },
    {# Test case 03 - noborder & conn 2
        'img': skeleton_noborder, 'conn': 2,
    },

]

#%% Save expected outputs -----------------------------------------------------

for i, test_case in enumerate(test_cases):
    io.imsave(
        DATA_PATH / f'pixconn_expected_output_{i:02}.tif',
        pixconn(**test_case).astype('uint8'),
        check_contrast=False,
        )
    
for i, test_case in enumerate(test_cases):
    io.imsave(
        DATA_PATH / f'labconn_expected_output_{i:02}.tif',
        labconn(**test_case).astype('uint8'),
        check_contrast=False,
        )