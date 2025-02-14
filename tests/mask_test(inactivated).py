#%% Imports -------------------------------------------------------------------

import sys
import pytest
import numpy as np
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_PATH / 'tests' / 'data' / 'patch'
sys.path.insert(0, str(ROOT_PATH))

from bdtools.mask import get_edt

#%% Function(s) ---------------------------------------------------------------

from skimage.morphology import disk, ball

def generate_random_array(nZ, nY, nX, nObj, min_radius, max_radius):
        
    # Define random variables
    zIdx = np.random.randint(0, nZ, nObj)
    yIdx = np.random.randint(0, nY, nObj)
    xIdx = np.random.randint(0, nX, nObj)
    if min_radius >= max_radius:
        min_radius -= 1
    radius = np.random.randint(min_radius, max_radius, nObj)
    labels = np.random.choice(
        np.arange(1, nObj * 2), size=nObj, replace=False)
    
    # Create array
    arr = []
    for i in range(nObj):
        tmp = np.zeros((nZ, nY, nX), dtype="int32").squeeze()
        
        if nZ > 1:
            obj = ball(radius[i])
            z0 = zIdx[i] - obj.shape[0] // 2
            y0 = yIdx[i] - obj.shape[1] // 2
            x0 = xIdx[i] - obj.shape[2] // 2
            z1 = z0 + obj.shape[0]
            y1 = y0 + obj.shape[1]
            x1 = x0 + obj.shape[2]
            if z0 < 0:
                obj = obj[-z0:, :, :]; z0 = 0
            if z1 > nZ:
                obj = obj[:nZ - z0, :, :]; z1 = nZ
            if y0 < 0:  
                obj = obj[:, -y0:, :]; y0 = 0
            if y1 > nY: 
                obj = obj[:, :nY - y0, :]; y1 = nY
            if x0 < 0:  
                obj = obj[:, :, -x0:]; x0 = 0
            if x1 > nX: 
                obj = obj[:, :, :nX - x0]; x1 = nX
            tmp[z0:z1, y0:y1, x0:x1] = obj
        
        else:
            obj = disk(radius[i])
            y0 = yIdx[i] - obj.shape[0] // 2
            x0 = xIdx[i] - obj.shape[1] // 2
            y1 = y0 + obj.shape[0]
            x1 = x0 + obj.shape[1]
            if y0 < 0:  
                obj = obj[-y0:, :]; y0 = 0
            if y1 > nY: 
                obj = obj[:nY - y0, :]; y1 = nY
            if x0 < 0:  
                obj = obj[:, -x0:]; x0 = 0
            if x1 > nX: 
                obj = obj[:, :nX - x0]; x1 = nX
            tmp[y0:y1, x0:x1] = obj
        
        tmp *= labels[i]
        arr.append(tmp)
    if arr:
        arr = arr[0] if nObj == 1 else np.max(np.stack(arr), axis=0)
    else:
        arr = np.zeros((nZ, nY, nX), dtype="int32").squeeze()
    
    return arr

#%% Test cases ----------------------------------------------------------------

params_edt = []
for i in range(50):
    
    # generate_random_array() parameters
    dtype = str(np.random.choice(["uint8", "uint16", "int32", "bool"]))
    nZ = np.random.choice([1, np.random.randint(2, 32)])
    nY = np.random.randint(64, 512)
    nX = np.random.randint(64, 512)
    nObj = np.random.randint(0, 16)
    min_radius = np.random.randint(8, 16)
    max_radius = round(min_radius * np.random.uniform(1.1, 3))
    
    # get_edt() parameters
    target = str(np.random.choice(["foreground", "background"]))
    sampling_tuple = (
        round(np.random.uniform(0.001, 2), 3), 
        round(np.random.uniform(0.001, 2), 3), 
        round(np.random.uniform(0.001, 2), 3),
        )
    sampling_tuple = sampling_tuple[:2] if nZ == 1 else sampling_tuple
    sampling = 1 if np.random.choice([True, False]) else sampling_tuple
    normalize = str(np.random.choice(["none", "global", "object"]))
    rescale_factor = np.random.choice([1, round(np.random.uniform(0.2, 1), 3)])
    parallel = np.random.choice([True, False]) if nZ == 1 else True

    params_edt.append((
        dtype, nZ, nY, nX, nObj, min_radius, max_radius,
        target, sampling, normalize, rescale_factor, parallel, 
        )) 

#%% Tests ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "dtype, nZ, nY, nX, nObj, min_radius, max_radius, "
    "target, sampling, normalize, rescale_factor, parallel",
    params_edt
    )

def test_get_edt(
        dtype, nZ, nY, nX, nObj, min_radius, max_radius,
        target, sampling, normalize, rescale_factor, parallel,
        ):
       
    # Generate random array
    arr = generate_random_array(nZ, nY, nX, nObj, min_radius, max_radius)
    arr = arr.astype(f"{dtype}")
        
    # Perform tests
    try:
        edt = get_edt(
            arr, 
            target=target, 
            sampling=sampling, 
            normalize=normalize,
            rescale_factor=rescale_factor,
            parallel=parallel,
            )
        
        assert edt.shape == arr.shape, (
            "edt.shape != arr.shape"
            )
        
        max_edt = np.max(edt)
        if normalize == "global":
            assert max_edt <= 1, (
                f"np.max(edt) = {max_edt:.3f}" 
                f" with target='{target}'"
                f" and normalize='{normalize}'"
                )    
        
        if (normalize == "global" or
            normalize == "object" and
            target == "foreground"):
            assert max_edt <= 1, (
                f"np.max(edt) = {max_edt:.3f}" 
                f" with target='{target}'"
                f" and normalize='{normalize}'"
                )        
        
    except Exception as e:
        pytest.fail(f"An error occurred: {str(e)}")

#%% Execute -------------------------------------------------------------------

# if __name__ == "__main__":
#     pytest.main([__file__])