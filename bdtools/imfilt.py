#%% Imports

import time
import warnings
import numpy as np
from skimage import io 
from numba import njit, prange
from pathlib import Path
from skimage.draw import ellipsoid

# -----------------------------------------------------------------------------

stack_name = 'noise_3d_512(crop).tif'
mask_name = 'mask_3d_512(crop).tif'
stackfilt_name = 'filt.tif'

# -----------------------------------------------------------------------------


img = io.imread(Path('../data', stack_name))
mask = io.imread(Path('../data', mask_name))
# mask = None

# -----------------------------------------------------------------------------

kernel_size = (3, 5, 5)
iterations = 3
filt_method = 'mean'
pad_method = 'reflect'
pad_value = None
parallel = True

#%%

@njit(nogil=True, parallel=parallel)
def imfilt_mean(
        img, 
        img_pad, 
        kernel_size_z,
        kernel_size_y,
        kernel_size_x,
        strel,
        ):
                
    pad_z = kernel_size_z//2
    pad_y = kernel_size_y//2
    pad_x = kernel_size_x//2
    
    img_filt = img_pad.copy()      

    for z in prange(img.shape[0]):
        for y in range(img.shape[1]):
            for x in range(img.shape[2]):
                
                if ~np.isnan(img[z,y,x]):
  
                    img_filt[z+pad_z,y+pad_y,x+pad_x] = np.nanmean(
                        img_pad[
                            z:z+kernel_size_z,
                            y:y+kernel_size_y,
                            x:x+kernel_size_x,
                            ] * strel
                        )
        
    return img_filt
                    
#%%

start = time.time()
print('Get time')

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
if kernel_size_z == 1: parallel = False # deactivate parallel

# Convert img to float
img = img.astype(float)
    
# Add z dimension (if ndim == 2) 
if img.ndim == 2: 
    img = np.expand_dims(img, 0) 
    
# Mask operations
if mask is not None:
          
    # Add z dimension (if ndim == 2) 
    if mask.ndim == 2: 
        mask = np.expand_dims(mask, 0) 

    # Match mask and img shape
    if mask.shape[0] == img.shape[0]:
        pass
    elif mask.shape[0] == 1 and mask.shape[0] < img.shape[0]: 
        mask = np.repeat(mask, img.shape[0], axis=0)
    else:
        raise Exception('mask shape is not compatible with img shape')
        
    # Set "out of mask" img pixels as NaNs
    img[~mask.astype(bool)] = np.nan

# Pad img
pad_z = kernel_size_z//2
pad_y = kernel_size_y//2
pad_x = kernel_size_x//2
pad_all = ((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x))

if pad_method == 'constant':
    img_pad = np.pad(img, 
        pad_width=pad_all, 
        mode=pad_method, 
        constant_values=pad_value,
        )
else:
    img_pad = np.pad(img, 
        pad_width=pad_all, 
        mode=pad_method, 
        )   
    
# Define structuring element
if kernel_size_z == 1:
    strel = ellipsoid(1, pad_y, pad_x, spacing=(2, 1, 1))
else:    
    strel = ellipsoid(pad_z, pad_y, pad_x, spacing=(1, 1, 1)) 
strel = strel[1:-1,1:-1,1:-1].astype('float').squeeze()
strel[strel == 0] = np.nan
    
for _ in range(iterations):

    img_filt = imfilt_mean(
        img, img_pad, 
        kernel_size_z,
        kernel_size_y,
        kernel_size_x,
        strel,
        )
    
    img_pad = img_filt.copy()  

# Unpad img_filt
if kernel_size_z == 1:
    img_filt = img_filt[:,pad_y:-pad_y,pad_x:-pad_x].squeeze()
else: 
    img_filt = img_filt[pad_z:-pad_z,pad_y:-pad_y,pad_x:-pad_x].squeeze()
    
io.imsave(
    Path('../data', stack_name.replace('.tif', '_filt.tif')),
    img_filt.astype('float32'),
    check_contrast=False,
    )

end = time.time()
print(f'  {(end-start):5.3f} s')     