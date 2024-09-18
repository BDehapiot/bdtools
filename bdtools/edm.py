#%% Imports -------------------------------------------------------------------

import napari
import numpy as np

# Scipy
from scipy.ndimage import rotate

# Skimage
from skimage.filters import gaussian
from skimage.morphology import disk, ellipse, ball, binary_dilation, remove_small_objects

#%% Function: -----------------------------------------------------------------

# def get_edm(msk, direction="in", normalize="none"):
    
#     global edm
    
#     if not (np.issubdtype(msk.dtype, np.integer) or
#             np.issubdtype(msk.dtype, np.bool_)):
#         raise TypeError("Provided mask must be bool or int labels")
    
#     if np.issubdtype(msk.dtype, np.bool_):
#         if direction == "in":
#             edm = distance_transform_edt(msk)
#         elif direction == "out":
#             edm = distance_transform_edt(np.invert(msk))
#         if normalize == "global":
            
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Parameters
    nObjects = 20
    nY, nX = 256, 256
    nProb = 0.25
    dMin = nY * 0.02
    dMax = dMin * 3 

    # Define random variables
    oIdx = np.arange(nObjects)
    yIdx = np.random.randint(0, nY, nObjects)
    xIdx = np.random.randint(0, nX, nObjects)
    width = np.random.randint(dMin, dMax, nObjects)
    height = np.random.randint(dMin, dMax, nObjects)
    angle = np.random.randint(0, 180, nObjects)

    # Create array
    arr = np.zeros((nObjects, nY, nX), dtype=float)
    arr[(oIdx, yIdx, xIdx)] = 1
    for i in range(nObjects):
        footprint = ellipse(width[i], height[i])
        footprint = rotate(footprint, angle[i], reshape=True)
        arr[i, ...] = binary_dilation(arr[i, ...], footprint=footprint)
        
    # Add noise & threshold
    noise = np.random.rand(nObjects, nY, nX)
    arr = np.maximum(arr, noise)
    for i in range(nObjects):
        arr[i, ...] = gaussian(arr[i, ...], sigma=2)
    arr = arr > 0.8
    
    
    # # Add boolean noise
    # noise = np.random.choice(
    #     [0, 1], size=(nObjects, nY, nX), p=[1 - nProb, nProb])
    # arr = np.maximum(arr, noise)
 
    # #
    # for i in range(nObjects):
    #     arr[i, ...] = gaussian(arr[i, ...], sigma=2)
    # arr = arr > 0.5 
    # arr = remove_small_objects(arr, min_size=(4/3 * np.pi * dMin ** 2) / 2)
    
    # # Generate random arr
    # arr = np.zeros((nObjects, nY, nX), dtype=bool)

    # arr[idx] = 1
    # for i in range(nObjects):
    #     arr[i, ...] = binary_dilation(
    #         arr[i, ...], footprint=disk(np.random.randint(dMin, dMax)))
    
    # Display
    viewer = napari.Viewer()
    viewer.add_image(arr)    

#%%

# from skimage.morphology import ellipse
# footprint = ellipse(10, 20)
# footprint = rotate(footprint, 25, reshape=True)

# # Display
# viewer = napari.Viewer()
# viewer.add_image(footprint )    
