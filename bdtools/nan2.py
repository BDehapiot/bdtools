#%% Imports

import time
import warnings
import numpy as np
from skimage import io 
from pathlib import Path
from skimage.draw import ellipsoid

# -----------------------------------------------------------------------------

stack_name = 'noise_3d_256.tif'
mask_name = 'mask_256.tif'
stackfilt_name = 'filt.tif'

# -----------------------------------------------------------------------------

kernel_size = (3,5,5)
img = io.imread(Path('../data', stack_name))
mask = io.imread(Path('../data', mask_name))
mask = None
method = 'mean'
iterations = 3

#%%

start = time.time()
print('Get time')

# Convert img to float
img = img.astype(float)
    
# Add z dimension (if ndim == 2) 
if img.ndim == 2: 
    img = np.expand_dims(img, 0) 

if mask is not None:
          
    # Add z dimension (if ndim == 2) 
    if mask.ndim == 2: 
        mask = np.expand_dims(mask, 0) 

    # Match mask and img shape
    if mask.shape[0] < img.shape[0]: 
        mask = np.repeat(mask, img.shape[0], axis=0)
    
    # Set out of mask pixels as NaNs
    if mask is not None:
        img[~mask.astype(bool)] = np.nan

# Warnings
warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
warnings.filterwarnings(action="ignore", message="Mean of empty slice")

# Extract kernel_size variables
if isinstance(kernel_size, int):
    if img.ndim == 2:
        kernel_size_z = 1  
        kernel_size_y = kernel_size
        kernel_size_x = kernel_size          
    elif img.ndim == 3:
        kernel_size_z = kernel_size
        kernel_size_y = kernel_size
        kernel_size_x = kernel_size         
elif len(kernel_size) == 2:
    kernel_size_z = 1  
    kernel_size_y = kernel_size[0]
    kernel_size_x = kernel_size[1]  
elif len(kernel_size) == 3:
    kernel_size_z = kernel_size[0]
    kernel_size_y = kernel_size[1]
    kernel_size_x = kernel_size[2]
    
# Round kernel_size variables to next odd integer    
if kernel_size_z % 2 == 0:
    print(f'z kernel size adjusted from {kernel_size_z} to {kernel_size_z + 1}')
    kernel_size_z += 1     
if kernel_size_y % 2 == 0:
    print(f'y kernel size adjusted from {kernel_size_y} to {kernel_size_y + 1}')
    kernel_size_y += 1 
if kernel_size_x % 2 == 0:
    print(f'x kernel size adjusted from {kernel_size_x} to {kernel_size_x + 1}')
    kernel_size_x += 1 
    
# # Define structuring element
# if kernel_size_z == 1:
#     strel = ellipsoid(1, pad_y, pad_x, spacing=(2, 1, 1))
# else:    
#     strel = ellipsoid(pad_z, pad_y, pad_x, spacing=(1, 1, 1)) 
# strel = strel[1:-1,1:-1,1:-1].astype('float').squeeze()
# strel[strel == 0] = np.nan
    
end = time.time()
print(f'  {(end-start):5.3f} s') 