#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np

# Scipy
from scipy.ndimage import rotate, distance_transform_edt

# Skimage
from skimage.filters import gaussian
from skimage.morphology import ellipse, binary_dilation

#%% Function: -----------------------------------------------------------------

def get_edm(msk, direction="in", normalize="none"):
    
    global labels, edm
    
    if not (np.issubdtype(msk.dtype, np.integer) or
            np.issubdtype(msk.dtype, np.bool_)):
        raise TypeError("Provided mask must be bool or int labels")
    
    if np.issubdtype(msk.dtype, np.bool_):
        pass
    elif np.issubdtype(msk.dtype, np.integer):
        labels = np.unique(arr)[1:]
        edm = np.zeros((labels.shape[0], arr.shape[0], arr.shape[1]))
        for l, lab in enumerate(labels):
            tmp = msk == lab
            tmp = distance_transform_edt(tmp)
            pMax = np.percentile(tmp[tmp > 0], 99.9)
            tmp[tmp > pMax] = pMax
            tmp = (tmp / pMax)
            edm[l,...] = tmp
        edm = np.max(edm, axis=0).astype("float32")  
                    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    # Parameters    
    nObjects = 100
    nY, nX = 1024, 1024
    dMax = nY * 0.05
    dMin = dMax * 0.25

    t0 = time.time(); 
    print("Random mask : ", end='')

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
    print("get_edm() : ", end='')
    
    get_edm(arr)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.5f}s")
       
    # Display
    viewer = napari.Viewer()
    viewer.add_labels(arr)
    viewer.add_image(edm)

