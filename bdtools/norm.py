#%% Imports -------------------------------------------------------------------

import numpy as np

#%% Function: norm_gcn --------------------------------------------------------

def norm_gcn(arr, sample_fraction=1, mask=None):
    
    # Check inputs
    if arr.dtype != "float32":
        arr = arr.astype("float32")
    if sample_fraction < 0 or sample_fraction > 1:
        raise ValueError("sample_fraction should be float between 0 and 1")
    if mask is not None and mask.shape != arr.shape:
        raise ValueError("array and mask should have the same shape")
    
    # Extract values
    val = arr.ravel()
    if mask is not None:
        val = val[mask.ravel()]
    if sample_fraction < 1:
        val = np.random.choice(val, size=int(arr.size * sample_fraction))
    val = val[~np.isnan(val)]
        
    # Normalize
    arr -= np.mean(val)
    arr /= np.std(val) 
    
    return arr

#%% Test cases ----------------------------------------------------------------

params = []
for i in range(50):

    dtype = str(np.random.choice(["uint8", "uint16", "float32"]))
    shape = str(np.random.choice(["2D", "3D", "4D"]))
    sample_fraction = np.random.rand()
    addMask = np.random.choice([True, False])
    addNaNs = np.random.choice([True, False])
    loc = np.random.rand()
    scale = np.random.rand() * 0.2

    params.append((
        dtype, shape, sample_fraction, addNaNs, addMask, loc, scale))  

#%% Tests ---------------------------------------------------------------------

import pytest

@pytest.mark.parametrize(
    "dtype, shape, sample_fraction, addNaNs, addMask, loc, scale", params)
def test_norm_gcn(dtype, shape, sample_fraction, addNaNs, addMask, loc, scale):
    
    # Get shape
    if shape == "2D": 
        size = (256, 256)
    if shape == "3D": 
        size = (5, 256, 256)
    if shape == "4D": 
        size = (2, 5, 256, 256)
        
    # Generate random array
    if dtype == "float32":
        maxInt = 1
    if dtype == "uint8":
        maxInt = 255
    if dtype == "uint16":    
        maxInt = 65535
    arr = np.random.normal(
        loc=loc * maxInt, scale=scale * maxInt, size=size)
    arr = np.clip(arr, 0, maxInt)
    
    # Generate random mask
    if addMask:
        mask = np.random.choice([True, False], size=size)
    else:
        mask = None
        
    # Generate random NaNs
    if addNaNs:
        nan_idx = np.random.choice(
            arr.size, int(arr.size * 0.1), replace=False)
        nan_idx = np.unravel_index(nan_idx, size)
        arr[nan_idx] = np.nan
    
    # Tests
    try:
        arr_norm = norm_gcn(arr, sample_fraction=sample_fraction, mask=mask)
        
        # Check mean and standard deviation
        atol = 0.01
        mean, std = np.nanmean(arr_norm), np.nanstd(arr_norm)
        assert np.isclose(mean, 0, atol=atol), f"mean = {mean:.3f} is not close enough to 0 (tolerance = {atol})"
        assert np.isclose(std, 1, atol=atol), f"std = {std:.3f} is not close enough to 1 (tolerance = {atol})"
    except Exception as e:
        pytest.fail(f"An error occurred: {str(e)}")
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__])

#%% -----------------------------------------------------------------------------

import time
import napari
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

# Create random array
dtype = "uint8" # "uint16"
loc = np.random.rand()
scale = np.random.rand() * 0.2
if dtype == "float32":
    maxInt = 1
if dtype == "uint8":
    maxInt = 255
if dtype == "uint16":    
    maxInt = 65535
arr = np.random.normal(
    loc=loc * maxInt, scale=scale * maxInt, size=(256, 256))
arr = np.clip(arr, 0, maxInt)
    
# # Add NaNs
# nan_idx = np.random.choice(arr.size, int(arr.size * 0.1), replace=False)
# nan_idx = np.unravel_index(nan_idx, shape)
# arr[nan_idx] = np.nan

# Normalize
arr_norm = norm_gcn(arr)

# Test
print(np.mean(arr_norm))
print(np.std(arr_norm))

# -----------------------------------------------------------------------------

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 12))

# Plot histogram for raw array
axes[0].hist(arr.ravel(), bins=100)
axes[0].set_title('Histogram raw array')
# axes[0].set_xlim([0, 255])
axes[0].set_xlabel('Pixel Intensity')
axes[0].set_ylabel('Frequency')

# Plot histogram for normalized array
axes[1].hist(arr_norm.ravel(), bins=100)
axes[1].set_title('Histogram normalized array')
# axes[1].set_xlim([0, 255])
axes[1].set_xlabel('Pixel Intensity')
axes[1].set_ylabel('Frequency')

plt.show()

#%%
