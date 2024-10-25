#%% Imports -------------------------------------------------------------------

import sys
import pytest
import numpy as np
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_PATH / 'tests' / 'data' / 'patch'
sys.path.insert(0, str(ROOT_PATH))

from skimage.transform import rescale 
from bdtools.norm import norm_gcn, norm_pct

#%% Test cases ----------------------------------------------------------------

params_gcn = []
for i in range(50):

    dtype = str(np.random.choice(["uint8", "uint16", "float32"]))
    shape = str(np.random.choice(["2D", "3D", "4D"]))
    sample_fraction = round(np.random.uniform(0.001, 1.0), 3)
    addMask = np.random.choice([True, False])
    addNaNs = np.random.choice([True, False])
    loc = round(np.random.uniform(0.1, 0.9), 3)
    scale = round(np.random.uniform(0.05, 0.2), 3)
    isList = np.random.choice([True, False])

    params_gcn.append((
        dtype, shape, sample_fraction, addNaNs, addMask, loc, scale, isList)) 
    
params_pct = []
for i in range(50):

    dtype = str(np.random.choice(["uint8", "uint16", "float32"]))
    shape = str(np.random.choice(["2D", "3D", "4D"]))
    sample_fraction = round(np.random.uniform(0.001, 1.0), 3)
    addMask = np.random.choice([True, False])
    addNaNs = np.random.choice([True, False])
    loc = round(np.random.uniform(0.1, 0.9), 3)
    scale = round(np.random.uniform(0.05, 0.2), 3)
    pct_low = round(np.random.uniform(0, 50), 3)
    pct_high = round(np.random.uniform(51, 100), 3)
    isList = np.random.choice([True, False])

    params_pct.append((
        dtype, shape, sample_fraction, addNaNs, addMask, loc, scale, pct_low, pct_high, isList)) 
    
#%% Tests ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "dtype, shape, sample_fraction, addNaNs, addMask, loc, scale, isList", params_gcn)
def test_norm_gcn(dtype, shape, sample_fraction, addNaNs, addMask, loc, scale, isList):
    
    # Get shape
    if shape == "2D": size = (256, 256)
    if shape == "3D": size = (5, 256, 256)
    if shape == "4D": size = (2, 5, 256, 256)
        
    # Get dtype
    if dtype == "float32": maxInt = 1
    if dtype == "uint8"  : maxInt = 255
    if dtype == "uint16" : maxInt = 65535
    
    # Generate random array
    arr = np.random.normal(loc=loc * maxInt, scale=scale * maxInt, size=size)
    arr = np.clip(arr, 0, maxInt)
    
    # Generate random mask
    if addMask: mask = np.random.choice([True, False], size=size)
    else: mask = None
        
    # Generate random NaNs
    if addNaNs:
        nan_idx = np.random.choice(
            arr.size, int(arr.size * 0.1), replace=False)
        nan_idx = np.unravel_index(nan_idx, size)
        arr[nan_idx] = np.nan
        
    # Convert to list (if isList)
    if isList:
        n = int(np.prod(arr.shape[:-2]))
        arr = [arr.reshape(n, *arr.shape[-2:])[i] for i in range(n)]
        if addMask:
            mask = [mask.reshape(n, *mask.shape[-2:])[i] for i in range(n)]
            
        # Rescale
        for i in range(len(arr)):
            rf = round(np.random.uniform(0.5, 2), 3)
            arr[i] = rescale(arr[i], rf, preserve_range=True)
            if addMask:
                mask[i] = rescale(mask[i], rf, order=0, preserve_range=True)
    
    # Perform tests
    try:
        arr_norm = norm_gcn(arr, sample_fraction=sample_fraction, mask=mask)
        if isinstance(arr_norm, np.ndarray):
            val = arr_norm.ravel()
        elif isinstance(arr_norm, list):
            val = np.concatenate([ar.ravel() for ar in arr_norm])
        atol, mean, std = 0.2, np.nanmean(val), np.nanstd(val)
        assert np.isclose(mean, 0, atol=atol), f"mean ({mean:.3f}) out of tolerance (0 +- {atol})"
        assert np.isclose(std, 1, atol=atol), f"std ({std:.3f}) out of tolerance (1 +- {atol})"
                
    except Exception as e:
        pytest.fail(f"An error occurred: {str(e)}")
   
@pytest.mark.parametrize(
    "dtype, shape, sample_fraction, addNaNs, addMask, loc, scale, pct_low, pct_high, isList", params_pct)
def test_norm_pct(dtype, shape, sample_fraction, addNaNs, addMask, loc, scale, pct_low, pct_high, isList):
    
    # Get shape
    if shape == "2D": size = (256, 256)
    if shape == "3D": size = (5, 256, 256)
    if shape == "4D": size = (2, 5, 256, 256)
        
    # Get dtype
    if dtype == "float32": maxInt = 1
    if dtype == "uint8"  : maxInt = 255
    if dtype == "uint16" : maxInt = 65535
    
    # Generate random array
    arr = np.random.normal(loc=loc * maxInt, scale=scale * maxInt, size=size)
    arr = np.clip(arr, 0, maxInt)
    
    # Generate random mask
    if addMask: mask = np.random.choice([True, False], size=size)
    else: mask = None
        
    # Generate random NaNs
    if addNaNs:
        nan_idx = np.random.choice(
            arr.size, int(arr.size * 0.1), replace=False)
        nan_idx = np.unravel_index(nan_idx, size)
        arr[nan_idx] = np.nan
        
    # Convert to list (if isList)
    if isList:
        n = int(np.prod(arr.shape[:-2]))
        arr = [arr.reshape(n, *arr.shape[-2:])[i] for i in range(n)]
        if addMask:
            mask = [mask.reshape(n, *mask.shape[-2:])[i] for i in range(n)]
            
        # Random rescaling (if isList)
        for i in range(len(arr)):
            rf = round(np.random.uniform(0.5, 2), 3)
            arr[i] = rescale(arr[i], rf, preserve_range=True)
            if addMask:
                mask[i] = rescale(mask[i], rf, order=0, preserve_range=True)
    
    # Perform tests
    try:
        arr_norm = norm_pct(arr, pct_low=pct_low, pct_high=pct_high, sample_fraction=sample_fraction, mask=mask) 
        if isinstance(arr_norm, np.ndarray):
            val = arr_norm.ravel()
        elif isinstance(arr_norm, list):
            val = np.concatenate([ar.ravel() for ar in arr_norm])
        atol, min_val, max_val = 0.01, np.nanmin(val), np.nanmax(val)
        assert np.isclose(min_val, 0, atol=atol), f"min ({min_val:.3f}) is != 0"
        assert np.isclose(max_val, 1, atol=atol), f"max ({max_val:.3f}) is != 1"
        
    except Exception as e:
        pytest.fail(f"An error occurred: {str(e)}")
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__])