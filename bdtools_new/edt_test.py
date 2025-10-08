#%% Imports -------------------------------------------------------------------

import sys
import pytest
import numpy as np
from pathlib import Path
from skimage.morphology import disk, ball

# bdtools
from bdtools.mask import get_edt

ROOT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_PATH / 'tests' / 'data' / 'patch'
sys.path.insert(0, str(ROOT_PATH))

#%% Function(s) ---------------------------------------------------------------

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