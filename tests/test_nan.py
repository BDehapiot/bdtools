#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io 
from pathlib import Path
from bdtools import nanfilt

#%% Initialize ----------------------------------------------------------------

mask = io.imread(Path('resources') / 'nan' / 'mask.tif')
mask_3d = io.imread(Path('resources') / 'nan' / 'mask_3d.tif')
noise_nan = io.imread(Path('resources') / 'nan' / 'noise_nan.tif')
noise_nan_3d = io.imread(Path('resources') / 'nan' / 'noise_nan_3d.tif')

#%% Test: nanfilt -------------------------------------------------------------

test_cases = [
    
    {# Test case 00 - 2d image & no mask
        'img': noise_nan, 'mask': None,
        'kernel_size': 3, 'kernel_shape': 'cuboid',
        'filt_method': 'mean', 'iterations': 1,
        'parallel': False,
    },
    {# Test case 01 - 2d image & 2d mask
        'img': noise_nan, 'mask': mask,
        'kernel_size': 3, 'kernel_shape': 'cuboid',
        'filt_method': 'mean', 'iterations': 1,
        'parallel': False,
    },
    {# Test case 02 - 3d image & no mask
        'img': noise_nan_3d, 'mask': None,
        'kernel_size': 3, 'kernel_shape': 'cuboid',
        'filt_method': 'mean', 'iterations': 1,
        'parallel': False,
    },
    {# Test case 03 - 3d image & 2d mask
        'img': noise_nan_3d, 'mask': mask,
        'kernel_size': 3, 'kernel_shape': 'cuboid',
        'filt_method': 'mean', 'iterations': 1,
        'parallel': False,
    },
    {# Test case 04 - 3d image & 3d mask   
        'img': noise_nan_3d, 'mask': mask_3d,
        'kernel_size': 3, 'kernel_shape': 'cuboid',
        'filt_method': 'mean', 'iterations': 1,
        'parallel': False,
    },
    
]

# Load expected outputs
expected_outputs = []
for path in Path('resources', 'nan').iterdir():
    if 'expected' in path.name:
        expected_outputs.append(io.imread(path))

# Load expected outputs
test_outputs = [] 
for i, case in enumerate(test_cases):
    try:
        test_outputs.append(nanfilt(**case))
    except Exception as e:
        print(f'nanfilt test case {i:02} failed')
        print(f'Error: {e}')

# Compare test & expected outputs
for i in range(len(test_cases)):
    comparison = np.array_equal(
        test_outputs[i].astype('float32'), expected_outputs[i], equal_nan=True)
    if comparison:
        print(f'nanfilt test case {i:02}: passed')
    else:
        print(f'nanfilt test case {i:02}: failed')  
    
## Save expected outputs      
# for i, output in enumerate(test_outputs):
#     io.imsave(
#         Path('resources') / 'nan' / f'expected_output_{i:02}.tif',
#         output.astype('float32'),
#         check_contrast=False,
#         )