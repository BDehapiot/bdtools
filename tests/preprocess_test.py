#%% Imports -------------------------------------------------------------------

import sys
import pytest
import numpy as np
from pathlib import Path

# bdtools
from bdtools.norm import norm_pct
from bdtools.models import preprocess

# Skimage
from skimage.morphology import disk

ROOT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_PATH / 'tests' / 'data' / 'patch'
sys.path.insert(0, str(ROOT_PATH))

#%% Function(s) ---------------------------------------------------------------

def random_data(
        nY, nX, 
        nObj, min_radius, max_radius,
        img_noise, img_dtype, msk_dtype
        ):
        
    # Define random variables
    yIdx = np.random.randint(0, nY, nObj)
    xIdx = np.random.randint(0, nX, nObj)
    if min_radius >= max_radius:
        min_radius -= 1
    radius = np.random.randint(min_radius, max_radius, nObj)
    labels = np.random.choice(
        np.arange(1, nObj * 2), size=nObj, replace=False)
    
    # Create mask
    msk = []
    for i in range(nObj):
        tmp = np.zeros((nY, nX), dtype="int32").squeeze()
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
        msk.append(tmp)
        
    if msk:
        msk = msk[0] if nObj == 1 else np.max(np.stack(msk), axis=0)
    else:
        msk = np.zeros((nY, nX), dtype="int32").squeeze()
    if msk_dtype == "uint8":
        msk = msk.astype("uint8")
    if msk_dtype == "uint16":
        msk = msk.astype("uint16")
    if msk_dtype == "bool":
        msk = msk > 0
    
    # Create associated image
    img = (msk > 0).astype(float)
    img += np.random.normal(0, img_noise, img.shape)
    img = norm_pct(img)
    if img_dtype == "uint8" : 
        img = (img * 255).astype("uint8")
    if img_dtype == "uint16" :
        img = (img * 65535).astype("uint16")
    
    return img, msk

#%% Test cases ----------------------------------------------------------------

params_preprocess = []

for i in range(50):
    
    # random_data() parameters
    nData = np.random.choice([1, 5])
    nY = np.random.randint(64, 256, nData)
    nX = np.random.randint(64, 256, nData)
    nObj = np.random.randint(0, 16, nData)
    min_radius = np.random.randint(8, 16, nData)
    max_radius = [radius * np.random.uniform(1.1, 3) for radius in min_radius]
    max_radius = np.stack(max_radius).astype("int32")
    img_noise = round(np.random.uniform(0.5, 1.0), 3)
    img_dtype = str(np.random.choice(["uint8", "uint16", "float32"]))
    msk_dtype = str(np.random.choice(["uint8", "uint16", "bool"]))
      
    # preprocess() parameters
    addMsk = np.random.choice([True, False])
    msk_type = str(np.random.choice(["normal", "edt", "bounds"]))
    img_norm = str(np.random.choice(["global", "image"]))
    patch_size = 2 * np.random.randint(8, 32)
    patch_overlap = np.random.randint(0, patch_size - 1)

    params_preprocess.append((
        nData, nY, nX,
        nObj, min_radius, max_radius,
        img_noise, img_dtype, msk_dtype,
        addMsk, msk_type, img_norm,
        patch_size, patch_overlap,
        )) 
       
#%% Tests ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "nData, nY, nX, nObj, min_radius, max_radius, img_noise, img_dtype, msk_dtype, " 
    "addMsk, msk_type, img_norm, patch_size, patch_overlap", 
    params_preprocess
    )
def test_preprocess(
        nData, nY, nX, nObj, min_radius, max_radius, img_noise, img_dtype, msk_dtype,
        addMsk, msk_type, img_norm, patch_size, patch_overlap
        ):
       
    # Generate random data
    imgs, msks = [], []
    for d in range(nData):
        img, msk = random_data(
            nY[d], nX[d], 
            nObj[d], min_radius[d], max_radius[d],
            img_noise, img_dtype, msk_dtype
            )
        imgs.append(img)
        msks.append(msk)
        
    # Preprocess
    if addMsk:
        imgs, msks = preprocess(
            imgs, msks=msks,
            img_norm=img_norm,
            msk_type=msk_type, 
            patch_size=patch_size, 
            patch_overlap=patch_overlap,
            )
    else:
        imgs = preprocess(
            imgs, msks=None,
            img_norm=img_norm,
            msk_type=msk_type, 
            patch_size=patch_size, 
            patch_overlap=patch_overlap,
            )   
        
    # Asserts
    try:
        
        assert isinstance(imgs, np.ndarray), "output image(s) is(are) not np.ndarray"
        assert imgs.dtype == "float32", "output image(s) is(are) not 'float32'"
        assert np.min(imgs) >= 0, "image(s) normalization issues, values < 0 detected"
        assert np.max(imgs) <= 1.001, "image(s) normalization issues, values > 1 detected"
        
        if addMsk:
            
            assert isinstance(msks, np.ndarray), "output mask(s) is(are) not np.ndarray"
            assert msks.dtype == "float32", "output mask(s) is(are) not 'float32'"
            assert np.min(msks) >= 0, "mask(s) normalization issues, values < 0 detected"
            assert np.max(msks) <= 1.001, "mask(s) normalization issues, values > 1 detected"
        
                
    except Exception as e:
        pytest.fail(f"An error occurred: {str(e)}")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__])