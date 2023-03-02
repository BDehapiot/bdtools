#%% Imports

import warnings
import numpy as np
from skimage.morphology import disk, square, binary_erosion

#%% Functions

def nanfilt(img, mask=None, kernel_size=(), method='mean', iterations=1):
    
    """ 
    Filter image ignoring NaNs.
    
    Parameters
    ----------
    img : ndarray (float)
        Image to be filtered.
        
    kernel_size : int
        Size of filter disk kernel.
        Should be odd.
        
    method : str
        Type of applied filter.
        'mean', 'median', 'std', 'max', 'min'.
        
    iterations : int
        Iterations of filtering process.
    
    Returns
    -------  
    img : ndarray
        Processed image.
    
    """

    
    return img

# -----------------------------------------------------------------------------

def nanreplace(img, kernel_size=3, method='mean', mask=None):
    
    """ 
    Replace NaNs using filtering kernel.    
    
    The function iterates to replace NaNs connected to real numbers 
    until no more NaNs are found. A mask can be provided to select
    NaNs to be replaced.
    
    Parameters
    ----------
    img : ndarray (float)
        Image to be processed.
        
    kernel_size : int
        Size of filter disk kernel.
        Should be odd.
        
    method : str
        Type of applied filter.
        'mean', 'median', 'std', 'max', 'min'.
        
    mask : ndarray (bool)
        Only the NaNs inside the mask will be replaced.
    
    Returns
    -------  
    img : ndarray
        Processed image.
    
    """
    
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
        
    # Create mask if None and convert to bool
    if mask is None:
        mask = np.full(img.shape, True)
    else:
        mask = mask.astype('bool')
    
    # Pad img and mask with NaNs and False
    pad = kernel_size//2
    img = np.pad(img, pad_width=pad, constant_values=np.nan)
    mask = np.pad(mask, pad_width=pad, constant_values=False)
    
    # Define nan_disk
    if kernel_size == 3:
        nan_disk = square(kernel_size, dtype=float)
    else:
        nan_disk = disk(pad, dtype=float)
        nan_disk[nan_disk == 0] = np.nan
    
    while np.isnan(img[mask==True]).any():
    
        # Counting the NaNs to be replaced
        nan_count = np.count_nonzero(np.isnan(img[mask==True]))    
    
        # Find NaNs coordinates (outside mask)   
        nan_mask = np.isnan(img) & mask == True
        nan_mask = nan_mask ^ binary_erosion(nan_mask, square(kernel_size))
        idx = np.where(nan_mask == True)
        idx_y = idx[0]; idx_x = idx[1]
    
        # Define all kernels
        mesh_range = np.arange((-kernel_size//2)+1, (kernel_size//2)+1)
        mesh_y, mesh_x = np.meshgrid(mesh_range, mesh_range)
        kernel_y = idx_y[:, None, None] + mesh_y
        kernel_x = idx_x[:, None, None] + mesh_x
        
        # Filter img
        all_kernels = img[kernel_y,kernel_x]*nan_disk
        all_kernels = filt[method](all_kernels, axis=(1, 2))
        img[idx] = all_kernels
        
        # Break while loop if NaNs can't be replaced
        if nan_count == np.count_nonzero(np.isnan(img[mask==True])):
            warnings.warn('Cannot replace all NaNs, please increase kernel size')
            break
            
    # Unpad img
    img = img[pad:-pad,pad:-pad]
    
    return img

#%%

import time
import numpy as np
from skimage import io 
from pathlib import Path
from skimage.draw import ellipsoid

# -----------------------------------------------------------------------------

stack_name = 'noise_3d_256.tif'
mask_name = 'mask_3d_256.tif'
stackfilt_name = 'filt.tif'

# -----------------------------------------------------------------------------

kernel_size = 3
img = io.imread(Path('../data', stack_name))
mask = io.imread(Path('../data', mask_name))
method = 'mean'
iterations = 3

#%%

start = time.time()
print('Get time')

# Warnings
warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
warnings.filterwarnings(action="ignore", message="Mean of empty slice")

# Get filtering method
filt = {
    'mean': np.nanmean, 
    'median': np.nanmedian, 
    'std': np.nanstd, 
    'max': np.nanmax, 
    'min': np.nanmin,
    }

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
    kernel_size_z += 1 
if kernel_size_y % 2 == 0:
    kernel_size_y += 1 
if kernel_size_x % 2 == 0:
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
    
io.imsave(
    Path('../data', stackfilt_name),
    img.astype('float32'),
    check_contrast=False,
    )

end = time.time()
print(f'  {(end-start):5.3f} s') 