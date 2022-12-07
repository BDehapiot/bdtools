#%% Imports

import time
import napari
import warnings
import numpy as np
from skimage import io 
from pathlib import Path
from skimage.morphology import disk

#%% Initialize

img = io.imread(Path(Path.cwd(), 'data', 'noisynan.tif'))
mask = io.imread(Path(Path.cwd(), 'data', 'mask.tif'))
# mask = None
raw = img.copy()

#%% Parameters

kernel_size = 21
method = 'mean'
iterations = 3

#%% Function

start = time.time()
print('nan.replace')

filt = {
    'mean': np.nanmean, 
    'median': np.nanmedian, 
    'std': np.nanstd, 
    'max': np.nanmax, 
    'min': np.nanmin,
    }

# Warnings
warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
warnings.filterwarnings(action="ignore", message="Mean of empty slice")

# Round kernel_size to next odd integer
if kernel_size % 2 == 0:
    kernel_size += 1 
    
# Create mask (if None) and convert to bool
if mask is None:
    mask = np.full(img.shape, True)
else:
    mask = mask.astype('bool')

# Pad img and mask border with NaNs or 0
pad = kernel_size//2
img = np.pad(img, pad_width=pad, constant_values=np.nan)
mask = np.pad(mask, pad_width=pad, constant_values=0)

# Define nan_disk
nan_disk = disk(pad, dtype=float)
nan_disk[nan_disk == 0] = np.nan

if iterations == 'inf':

    while np.isnan(img[mask==True]).any():

        # Find NaNs coordinates (outside mask)
        idx = np.where(np.isnan(img) & (mask==True) == True)
        idx_y = idx[0]; idx_x = idx[1]
    
        # Define all kernels
        mesh_range = np.arange((-kernel_size//2)+1, (kernel_size//2)+1)
        mesh_y, mesh_x = np.meshgrid(mesh_range, mesh_range)
        kernel_y = (idx_y[:, None, None] + mesh_y)
        kernel_x = (idx_x[:, None, None] + mesh_x)
        
        # Filter img
        all_kernels = img[kernel_y,kernel_x]*nan_disk
        all_kernels = filt[method](all_kernels, axis=(1, 2))
        img[idx] = all_kernels
        
else:
    
    for _ in range(iterations):
    
        # Find NaNs coordinates (outside mask)
        idx = np.where(np.isnan(img) & (mask==True) == True)
        idx_y = idx[0]; idx_x = idx[1]
        
        # Define all kernels
        mesh_range = np.arange((-kernel_size//2)+1, (kernel_size//2)+1)
        mesh_y, mesh_x = np.meshgrid(mesh_range, mesh_range)
        kernel_y = (idx_y[:, None, None] + mesh_y)
        kernel_x = (idx_x[:, None, None] + mesh_x)
        
        # Filter img
        all_kernels = img[kernel_y,kernel_x]*nan_disk
        all_kernels = filt[method](all_kernels, axis=(1, 2))
        img[idx] = all_kernels

# Unpad img
img = img[pad:-pad,pad:-pad]

end = time.time()
print(f'  {(end-start):5.3f} s') 

#%%

viewer = napari.Viewer()
viewer.add_image(raw)
viewer.add_image(img)


