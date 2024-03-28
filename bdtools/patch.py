#%% Imports -------------------------------------------------------------------

import numpy as np
from joblib import Parallel, delayed 

#%% 

def get_patches(arr, size, overlap):
    
    # Get dimensions
    if arr.ndim == 2: nT = 1; nY, nX = arr.shape 
    if arr.ndim == 3: nT, nY, nX = arr.shape
    
    # Get variables
    y0s = np.arange(0, nY, size - overlap)
    x0s = np.arange(0, nX, size - overlap)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1, yPad2 = yPad // 2, (yPad + 1) // 2
    xPad1, xPad2 = xPad // 2, (xPad + 1) // 2
    
    # Pad array
    if arr.ndim == 2:
        arr_pad = np.pad(
            arr, ((yPad1, yPad2), (xPad1, xPad2)), mode='reflect') 
    if arr.ndim == 3:
        arr_pad = np.pad(
            arr, ((0, 0), (yPad1, yPad2), (xPad1, xPad2)), mode='reflect')         
    
    # Extract patches
    patches = []
    if arr.ndim == 2:
        for y0 in y0s:
            for x0 in x0s:
                patches.append(arr_pad[y0:y0 + size, x0:x0 + size])
    if arr.ndim == 3:
        for t in range(nT):
            for y0 in y0s:
                for x0 in x0s:
                    patches.append(arr_pad[t, y0:y0 + size, x0:x0 + size])
            
    return patches

def merge_patches(patches, shape, size, overlap):
    
    # Get dimensions 
    if len(shape) == 2: nT = 1; nY, nX = shape
    if len(shape) == 3: nT, nY, nX = shape
    nPatch = len(patches) // nT

    # Get variables
    y0s = np.arange(0, nY, size - overlap)
    x0s = np.arange(0, nX, size - overlap)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1 = yPad // 2
    xPad1 = xPad // 2

    # Merge patches
    def _merge_patches(patches):
        count = 0
        arr = np.full((2, nY + yPad, nX + xPad), np.nan)
        for i, y0 in enumerate(y0s):
            for j, x0 in enumerate(x0s):
                if i % 2 == j % 2:
                    arr[0, y0:y0 + size, x0:x0 + size] = patches[count]
                else:
                    arr[1, y0:y0 + size, x0:x0 + size] = patches[count]
                count += 1 
        arr = np.nanmean(arr, axis=0)
        arr = arr[yPad1:yPad1 + nY, xPad1:xPad1 + nX]
        return arr
        
    if len(shape) == 2:
        arr = _merge_patches(patches)

    if len(shape) == 3:
        patches = np.stack(patches).reshape(nT, nPatch, size, size)
        arr = Parallel(n_jobs=-1)(
            delayed(_merge_patches)(patches[t,...])
            for t in range(nT)
            )
        arr = np.stack(arr)
        
    return arr

#%% 

import time
import napari

# -----------------------------------------------------------------------------

size = 64
overlap = 16
nZ, nY, nX = 60, 600, 1000

# -----------------------------------------------------------------------------

# Generate random arrays
arr = np.random.randn(nZ, nY, nX)
arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
arr[:, :,  0] = 1 
arr[:, :, -1] = 1
arr[:,  0, :] = 1 
arr[:, -1, :] = 1

# # Display
# viewer = napari.Viewer()
# viewer.add_image(arr)

# -----------------------------------------------------------------------------

t0 = time.time()
print("Get patches : ", end='')

patches = get_patches(arr, size, overlap)
patches = np.stack(patches)

t1 = time.time()
print(f"{(t1-t0):<5.2f}s") 

# # Display
# viewer = napari.Viewer()
# viewer.add_image(patches)

# -----------------------------------------------------------------------------

t0 = time.time()
print("Merge patches : ", end='')

arr_m = merge_patches(patches, arr.shape, size, overlap)

t1 = time.time()
print(f"{(t1-t0):<5.2f}s") 

if np.array_equal(arr, arr_m):
    print("arrays are equal")

# # Display
# viewer = napari.Viewer()
# viewer.add_image(arr_m)

#%%


'''
1) Setup two scenarios: 
    - pad + reflect (already implemented) 
    - no pad and get patches from the center of image (not from [0, 0])
2) The Class should be made thinking that one could add processed patches to the 
instance and be able to merge them from an embeded method (merge_patches())
3) All metadata (patch coords, original array shape, if patch is from padded area) 
4) Investigate instance duplication (to store processed patches?)
'''

class Patch:
    
    def __init__(self, arr, size, overlap, pad=True):
        self.arr = arr
        self.size = size
        self.overlap = overlap
        self.pad = pad
        self._extract_patches()
    
    def _extract_patches(self):
        
        # Get dimensions
        if self.arr.ndim == 2: 
            nT = 1
            nY, nX = self.arr.shape 
        if self.arr.ndim == 3: 
            nT, nY, nX = self.arr.shape
            
        if self.pad:
        
            # Get variables
            y0s = np.arange(0, nY, self.size - self.overlap)
            x0s = np.arange(0, nX, self.size - self.overlap)
            yMax = y0s[-1] + self.size
            xMax = x0s[-1] + self.size
            yPad = yMax - nY
            xPad = xMax - nX
            yPad1, yPad2 = yPad // 2, (yPad + 1) // 2
            xPad1, xPad2 = xPad // 2, (xPad + 1) // 2
            
            # Pad array
            if self.arr.ndim == 2:
                arr_pad = np.pad(
                    self.arr, ((yPad1, yPad2), (xPad1, xPad2)), mode='reflect') 
            if self.arr.ndim == 3:
                arr_pad = np.pad(
                    self.arr, ((0, 0), (yPad1, yPad2), (xPad1, xPad2)), mode='reflect')    
                
        else:
            
            ''' Define new way for variables '''
            
            pass
               
        # Extract patches
        ''' Ideally this part should be common with pad=True and pad=False '''
        self.patches = []
        if self.arr.ndim == 2:
            for y0 in y0s:
                for x0 in x0s:
                    self.patches.append(arr_pad[y0:y0 + self.size, x0:x0 + self.size])
        if self.arr.ndim == 3:
            for t in range(nT):
                for y0 in y0s:
                    for x0 in x0s:
                        self.patches.append(arr_pad[t, y0:y0 + self.size, x0:x0 + self.size])
        self.patches = np.stack(self.patches)
        
        # Create metadata
        self.y0s = y0s
        self.x0s = x0s
        
        
    # def get_patches(self):
    #     return self.patches
    
    # def get_metadata(self):
    #     return self.metadata
    
patches = Patch(arr, size, overlap)
patches.newPatches = patches.patches + 1
print(np.mean(patches.newPatches))
y0s = patches.y0s

