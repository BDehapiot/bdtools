#%% Imports -------------------------------------------------------------------

import sys
import time
import pytest
import numpy as np
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_PATH / 'tests' / 'data' / 'patch'
sys.path.insert(0, str(ROOT_PATH))

from bdtools.patch import extract_patches, merge_patches

#%% Test cases ----------------------------------------------------------------

params = []
for i in range(50):
    
    nZ      = np.random.choice([2, 5, 10])
    nY      = np.random.randint(16, 256)
    nX      = np.random.randint(16, 256)
    size    = np.random.randint(16, 256)
    overlap = np.random.choice([0, size // 8, size // 4, size // 2])
    
    if np.random.choice([True, False]):
        shape = (nZ, nY, nX)
    else:
        shape = (nY, nX)
    params.append((shape, size, overlap))    
    
    # # Test execution time    
    # t0 = time.time()
    # arr = np.random.rand(*shape)
    # patches = extract_patches(arr, size, overlap)
    # merged_arr = merge_patches(patches, shape, overlap)
    # t1 = time.time()
    # print(f"{i:02d} - {(t1-t0):<5.2f}s") 

#%% Tests ---------------------------------------------------------------------

@pytest.mark.parametrize("shape, size, overlap", params)
def test_patches(shape, size, overlap):
    arr = np.random.rand(*shape)
    patches = extract_patches(arr, size, overlap)
    merged_arr = merge_patches(patches, shape, overlap)
    assert np.array_equal(arr, merged_arr), "Merged array differs from original"

#%% Execute -------------------------------------------------------------------

# if __name__ == "__main__":
#     pytest.main([__file__])