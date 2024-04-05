#%% Imports -------------------------------------------------------------------

import sys
import pytest
import numpy as np
from skimage import io 
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_PATH / 'tests' / 'data' / 'nan'
sys.path.insert(0, str(ROOT_PATH))

from bdtools.nan import nanfilt, nanreplace

#%% Initialize ----------------------------------------------------------------

mask = io.imread(DATA_PATH / 'mask.tif')
mask_3d = io.imread(DATA_PATH / 'mask_3d.tif')
mask_zoom = io.imread(DATA_PATH / 'mask_zoom.tif')
mask_3d_zoom = io.imread(DATA_PATH / 'mask_3d_zoom.tif')
noise_nan = io.imread(DATA_PATH / 'noise_nan.tif')
noise_nan_3d = io.imread(DATA_PATH / 'noise_nan_3d.tif')
noise_nan_zoom = io.imread(DATA_PATH / 'noise_nan_zoom.tif')
noise_nan_3d_zoom = io.imread(DATA_PATH / 'noise_nan_3d_zoom.tif')
noise_nan_hole = io.imread(DATA_PATH / 'noise_nan_hole.tif')

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

#%% Tests ---------------------------------------------------------------------

def compare_outputs(test_output, expected_output, function_name):
    compare = np.array_equal(
        test_output.astype('float32'), 
        expected_output, 
        equal_nan=True
    )
    assert compare, f"{function_name} failed"

# nanfilt ---------------------------------------------------------------------

@pytest.fixture(params=[
    (i, test_case) for i, test_case
    in enumerate(test_cases) if i not in [17, 18]])

def nanfilt_params(request):
    return request.param

def test_nanfilt(nanfilt_params):
    # Unpack idx and test_cases
    i, test_cases = nanfilt_params

    # Load expected output
    expected_output_path = DATA_PATH / f'nanfilt_expected_output_{i:02d}.tif'
    expected_output = io.imread(expected_output_path)

    # Test case
    try:
        test_output = nanfilt(**test_cases)
    except Exception as e:
        pytest.fail(f'nanfilt failed with error: {e}')
        raise e

    compare_outputs(test_output, expected_output, 'nanfilt')

# nanreplace ------------------------------------------------------------------

@pytest.fixture(params=[
    (i, test_case) for i, test_case
    in enumerate(test_cases)])

def nanreplace_params(request):
    return request.param

def test_nanreplace(nanreplace_params):
    # Unpack idx and test_cases
    i, test_cases = nanreplace_params

    # Load expected output
    expected_output_path = DATA_PATH / f'nanreplace_expected_output_{i:02d}.tif'
    expected_output = io.imread(expected_output_path)

    # Test case
    try:
        test_output = nanreplace(**test_cases)
    except Exception as e:
        pytest.fail(f'nanreplace failed with error: {e}')
        raise e

    compare_outputs(test_output, expected_output, 'nanreplace')

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__])