#%%

import time
import warnings
import numpy as np
from skimage import io 
from pathlib import Path
from skimage.draw import ellipsoid

# -----------------------------------------------------------------------------

# img_name = 'noise+nan_3d_256.tif'
# mask_name = 'mask_3d_256.tif'

img_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8_vecU.tif'
mask_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8_vecMask.tif'

# -----------------------------------------------------------------------------

img = io.imread(Path('../data/nan', img_name))
mask = io.imread(Path('../data/nan', mask_name))

# -----------------------------------------------------------------------------

kernel_size = 7
method = 'mean'
iterations = 3

#%%

start = time.time()
print('nanfilt')

# Warnings
warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
warnings.filterwarnings(action="ignore", message="Mean of empty slice")

# Extract kernel_size variables
if isinstance(kernel_size, int):
    if img.ndim == 2:
        kernel_size_z = 1  
        kernel_size_y = kernel_size
        kernel_size_x = kernel_size          
    elif img.ndim ==3:
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
     
# Set out of mask pixels as NaNs
if mask is not None:
    img[~mask.astype(bool)] = np.nan
    
# Add one dimension (if ndim == 2)
ndim = (img.ndim)        
if ndim == 2:
    img = img.reshape((1, img.shape[0], img.shape[1]))      
    
# Pad img with NaNs
pad_z = kernel_size_z//2
pad_y = kernel_size_y//2
pad_x = kernel_size_x//2
img = np.pad(img, 
    pad_width=((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x)), 
    constant_values=np.nan
    )

# Define structuring element
if kernel_size_z == 1:
    strel = ellipsoid(1, pad_y, pad_x, spacing=(2, 1, 1))
else:    
    strel = ellipsoid(pad_z, pad_y, pad_x, spacing=(1, 1, 1)) 
strel = strel[1:-1,1:-1,1:-1].astype('float').squeeze()
strel[strel == 0] = np.nan

# Find non-NaNs coordinates
idx = np.where(~np.isnan(img) == True)
idx_z = idx[0]; idx_y = idx[1]; idx_x = idx[2]  

# Define all kernels
mesh_range_z = np.arange(-pad_z, pad_z+1)
mesh_range_y = np.arange(-pad_y, pad_y+1)
mesh_range_x = np.arange(-pad_x, pad_x+1)
mesh_z, mesh_y, mesh_x = np.meshgrid(
    mesh_range_z,
    mesh_range_y,
    mesh_range_x,
    indexing='ij'
    )
kernel_z = idx_z[:, None, None, None] + mesh_z
kernel_y = idx_y[:, None, None, None] + mesh_y
kernel_x = idx_x[:, None, None, None] + mesh_x

# Filter img

filt = {
    'mean': np.nanmean, 
    'median': np.nanmedian, 
    'std': np.nanstd, 
    }

if kernel_size_z == 1:
    for _ in range(iterations):
        all_kernels = img[kernel_z,kernel_y,kernel_x].squeeze()*strel
        all_kernels = filt[method](all_kernels, axis=(1, 2))
        img[idx] = all_kernels
else:
    for _ in range(iterations):
        all_kernels = img[kernel_z,kernel_y,kernel_x].squeeze()*strel
        all_kernels = filt[method](all_kernels, axis=(1, 2, 3))
        img[idx] = all_kernels
                
# Unpad img
if kernel_size_z == 1:
    img = img[:,pad_y:-pad_y,pad_x:-pad_x].squeeze()
else: 
    img = img[pad_z:-pad_z,pad_y:-pad_y,pad_x:-pad_x].squeeze()
    
end = time.time()
print(f'  {(end-start):5.3f} s')

#%%

# Save img_filt    
io.imsave(
    Path('../data/nan', img_name.replace('.tif', '_filt.tif')),
    img.astype('float32'),
    check_contrast=False,
    ) 