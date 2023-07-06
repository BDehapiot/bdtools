#%% Imports -------------------------------------------------------------------

import pytest
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

#%% Tests ---------------------------------------------------------------------

def compare_outputs(test_output, expected_output, function_name):
    compare = np.array_equal(
        test_output.astype('float32'), 
        expected_output, 
        equal_nan=True
    )
    assert compare, f"{function_name} failed"

@pytest.fixture(params=[
    (i, test_case) for i, test_case
    in enumerate(test_cases)])

def params(request):
    return request.param

# pixconn ---------------------------------------------------------------------

def test_pixconn(params):
    # Unpack idx and test_cases
    i, test_cases = params

    # Load expected output
    expected_output_path = DATA_PATH / f'pixconn_expected_output_{i:02d}.tif'
    expected_output = io.imread(expected_output_path)

    # Test case
    try:
        test_output = pixconn(**test_cases)
    except Exception as e:
        pytest.fail(f'pixconn failed with error: {e}')
        raise e

    compare_outputs(test_output, expected_output, 'pixconn')

# labconn ---------------------------------------------------------------------

def test_labconn(params):
    # Unpack idx and test_cases
    i, test_cases = params

    # Load expected output
    expected_output_path = DATA_PATH / f'labconn_expected_output_{i:02d}.tif'
    expected_output = io.imread(expected_output_path)

    # Test case
    try:
        test_output = labconn(**test_cases)
    except Exception as e:
        pytest.fail(f'labconn failed with error: {e}')
        raise e

    compare_outputs(test_output, expected_output, 'labconn')

#%% Execute -------------------------------------------------------------------

# if __name__ == "__main__":
#     pytest.main([__file__])