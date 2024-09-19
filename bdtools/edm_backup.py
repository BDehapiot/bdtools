#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from joblib import Parallel, delayed 

# bdtools
from norm import norm_pct

# Scipy
from scipy.ndimage import rotate, distance_transform_edt

# Skimage
from skimage.measure import label
from skimage.morphology import ellipse, disk, ball

#%% Function: -----------------------------------------------------------------

def get_edm(arr, direction="in", normalize="none", parallel=True):
    
    global labels, edm
    
    # Nested function(s) ------------------------------------------------------
    
    def _get_edm(lab, normalize=normalize):
        edm = arr == lab
        edm = distance_transform_edt(edm)
        if normalize == "object":
            edm = norm_pct(edm, pct_low=0, pct_high=99.9, mask=edm > 0)
        return edm
        
    # Execute -----------------------------------------------------------------
    
    if not (np.issubdtype(arr.dtype, np.integer) or
            np.issubdtype(arr.dtype, np.bool_)):
        raise TypeError("Provided mask must be bool or int labels")
    
    if direction == "in":
        if np.issubdtype(arr.dtype, np.bool_):
            arr = label(arr)
        labels = np.unique(arr)[1:]
        if parallel:
            edm = Parallel(n_jobs=-1)(
                delayed(_get_edm)(lab) for lab in labels)
        else:
            edm = [_get_edm(lab) for lab in labels]        
        edm = np.nanmax(np.stack(edm), axis=0).astype("float32")
    
    elif direction == "out":
        edm = distance_transform_edt(np.invert(arr > 0))
        
    if normalize == "global":
        edm = norm_pct(edm, pct_low=0, pct_high=99.9, mask=edm > 0)
            
                    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    # Parameters    
    nObjects = 50
    nY, nX = 512, 512
    dMax = nY * 0.05
    dMin = dMax * 0.25

    t0 = time.time(); 
    print("Random mask 2D: ", end='')

    # Define random variables
    oIdx = np.arange(nObjects)
    yIdx = np.random.randint(0, nY, nObjects)
    xIdx = np.random.randint(0, nX, nObjects)
    width = np.random.randint(dMin, dMax, nObjects)
    height = np.random.randint(dMin, dMax, nObjects)
    angle = np.random.randint(0, 180, nObjects)
    labels = np.random.choice(
        np.arange(1, nObjects * 2), size=nObjects, replace=False)
    
    # Create array
    arr = []
    for i in range(nObjects):
        tmp = np.zeros((nY, nX), dtype="uint16")
        obj = ellipse(width[i], height[i])
        obj = rotate(obj, angle[i], reshape=True)
        y0 = yIdx[i] - obj.shape[0] // 2
        x0 = xIdx[i] - obj.shape[1] // 2
        y1 = y0 + obj.shape[0]
        x1 = x0 + obj.shape[1]
        if y0 < 0:  
            obj = obj[-y0:, ...]; y0 = 0
        if y1 > nY: 
            obj = obj[:nY - y0, ...]; y1 = nY
        if x0 < 0:  
            obj = obj[:, -x0:]; x0 = 0
        if x1 > nX: 
            obj = obj[:, :nX - x0]; x1 = nX
        tmp[y0:y1, x0:x1] = obj
        tmp *= labels[i]
        arr.append(tmp)
    arr = np.max(np.stack(arr), axis=0)

    t1 = time.time()
    print(f"{(t1-t0):<5.5f}s")
    
    t0 = time.time(); 
    print("Random mask 3D: ", end='')
    
    t1 = time.time()
    print(f"{(t1-t0):<5.5f}s")
    
    # t0 = time.time(); 
    # print("get_edm() : ", end='')
    
    # get_edm(arr, direction="in", normalize="object", parallel=True)
    
    # t1 = time.time()
    # print(f"{(t1-t0):<5.5f}s")
       
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_labels(arr)
    # viewer.add_image(edm)

