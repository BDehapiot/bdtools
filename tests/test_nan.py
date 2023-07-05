#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io 
from pathlib import Path
from bdtools import nanfilt, nanreplace

#%% Initialize ----------------------------------------------------------------

mask = io.imread(Path('data') / 'nan' / 'mask.tif')
mask_3d = io.imread(Path('data') / 'nan' / 'mask_3d.tif')
mask_zoom = io.imread(Path('data') / 'nan' / 'mask_zoom.tif')
mask_3d_zoom = io.imread(Path('data') / 'nan' / 'mask_3d_zoom.tif')
noise_nan = io.imread(Path('data') / 'nan' / 'noise_nan.tif')
noise_nan_3d = io.imread(Path('data') / 'nan' / 'noise_nan_3d.tif')
noise_nan_zoom = io.imread(Path('data') / 'nan' / 'noise_nan_zoom.tif')
noise_nan_3d_zoom = io.imread(Path('data') / 'nan' / 'noise_nan_3d_zoom.tif')
noise_nan_hole = io.imread(Path('data') / 'nan' / 'noise_nan_hole.tif')

#%% Test cases ----------------------------------------------------------------

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
    {# Test case 05 - 2d image & asymmetric kernel   
        'img': noise_nan, 'mask': None,
        'kernel_size': (1, 5), 'kernel_shape': 'cuboid',
        'filt_method': 'mean', 'iterations': 1,
        'parallel': False,
    },
    {# Test case 06 - 3d image & asymmetric kernel   
        'img': noise_nan_3d, 'mask': None,
        'kernel_size': (3, 1, 5), 'kernel_shape': 'cuboid',
        'filt_method': 'mean', 'iterations': 1,
        'parallel': False,
    },
    {# Test case 07 - 2d image & ellipsoid kernel   
        'img': noise_nan, 'mask': None,
        'kernel_size': 3, 'kernel_shape': 'ellipsoid',
        'filt_method': 'mean', 'iterations': 1,
        'parallel': False,
    },
    {# Test case 08 - 3d image & ellipsoid kernel   
        'img': noise_nan_3d, 'mask': None,
        'kernel_size': 3, 'kernel_shape': 'ellipsoid',
        'filt_method': 'mean', 'iterations': 1,
        'parallel': False,
    },
    {# Test case 09 - 2d image & median filter
        'img': noise_nan, 'mask': None,
        'kernel_size': 3, 'kernel_shape': 'cuboid',
        'filt_method': 'median', 'iterations': 1,
        'parallel': False,
    },
    {# Test case 10 - 2d image & std filter
        'img': noise_nan, 'mask': None,
        'kernel_size': 3, 'kernel_shape': 'cuboid',
        'filt_method': 'std', 'iterations': 1,
        'parallel': False,
    },
    {# Test case 11 - 2d image & multiple iterations
        'img': noise_nan, 'mask': None,
        'kernel_size': 3, 'kernel_shape': 'cuboid',
        'filt_method': 'mean', 'iterations': 3,
        'parallel': False,
    },
    {# Test case 12 - 3d image & parallel processing
        'img': noise_nan_3d, 'mask': None,
        'kernel_size': 3, 'kernel_shape': 'cuboid',
        'filt_method': 'mean', 'iterations': 1,
        'parallel': True,
    },
    {# Test case 13 - 2d zoom image & no mask
        'img': noise_nan_zoom, 'mask': None,
        'kernel_size': 3, 'kernel_shape': 'cuboid',
        'filt_method': 'mean', 'iterations': 1,
        'parallel': True,
    },
    {# Test case 14 - 2d zoom image & mask
        'img': noise_nan_zoom, 'mask': mask_zoom,
        'kernel_size': 3, 'kernel_shape': 'cuboid',
        'filt_method': 'mean', 'iterations': 1,
        'parallel': True,
    },
    {# Test case 15 - 3d zoom image & no mask
        'img': noise_nan_3d_zoom, 'mask': None,
        'kernel_size': 3, 'kernel_shape': 'cuboid',
        'filt_method': 'mean', 'iterations': 1,
        'parallel': True,
    },
    {# Test case 16 - 3d zoom image & mask
        'img': noise_nan_3d_zoom, 'mask': mask_3d_zoom,
        'kernel_size': 3, 'kernel_shape': 'cuboid',
        'filt_method': 'mean', 'iterations': 1,
        'parallel': True,
    },    
    {# Test case 17 (nanreplace only) - 2d hole image & no mask
        'img': noise_nan_hole, 'mask': None,
        'kernel_size': 3, 'kernel_shape': 'cuboid',
        'filt_method': 'mean', 'iterations': 'inf',
        'parallel': True,
    },
    {# Test case 18 (nanreplace only) - 2d hole image & mask
        'img': noise_nan_hole, 'mask': mask,
        'kernel_size': 3, 'kernel_shape': 'cuboid',
        'filt_method': 'mean', 'iterations': 'inf',
        'parallel': True,
    },
    
]

#%% Test: nanfilt -------------------------------------------------------------

# Load expected outputs
expected_outputs = []
for path in Path('data', 'nan').iterdir():
    if 'nanfilt_expected' in path.name:
        expected_outputs.append(io.imread(path))

# Test cases
test_outputs = [] 
for i, case in enumerate(test_cases):
    if i < 17:
        try:
            test_outputs.append(nanfilt(**case))
        except Exception as e:
            print(f'nanfilt test case {i:02}: failed')
            print(f'Error: {e}')

# Compare test & expected outputs
nanfilt_compare = []
for i in range(len(test_cases)):
    if i < 17:
        compare = np.array_equal(
            test_outputs[i].astype('float32'), 
            expected_outputs[i], 
            equal_nan=True
            )
        nanfilt_compare.append(compare)
        if not compare:
            print(f'nanfilt test case {i:02}: failed')
if all(nanfilt_compare):
    print('nanfilt tests: all passed')
              
# -----------------------------------------------------------------------------
    
# # Save expected outputs      
# for i, output in enumerate(test_outputs):
#     io.imsave(
#         Path('data') / 'nan' / f'nanfilt_expected_output_{i:02}.tif',
#         output.astype('float32'),
#         check_contrast=False,
#         )

#%% Test: nanreplace ----------------------------------------------------------

# Load expected outputs
expected_outputs = []
for path in Path('data', 'nan').iterdir():
    if 'nanreplace_expected' in path.name:
        expected_outputs.append(io.imread(path))

# Test cases
test_outputs = [] 
for i, case in enumerate(test_cases):
    try:
        test_outputs.append(nanreplace(**case))
    except Exception as e:
        print(f'nanreplace test case {i:02}: failed')
        print(f'Error: {e}')

# Compare test & expected outputs
nanreplace_compare = []
for i in range(len(test_cases)):
    compare = np.array_equal(
        test_outputs[i].astype('float32'), 
        expected_outputs[i], 
        equal_nan=True
        )    
    nanreplace_compare.append(compare)
    if not compare:
        print(f'nanreplace test case {i:02}: failed')
if all(nanreplace_compare):
    print('nanreplace tests: all passed')
    
# -----------------------------------------------------------------------------
    
# # Save expected outputs      
# for i, output in enumerate(test_outputs):
#     io.imsave(
#         Path('data') / 'nan' / f'nanreplace_expected_output_{i:02}.tif',
#         output.astype('float32'),
#         check_contrast=False,
#         )