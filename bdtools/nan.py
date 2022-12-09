#%% Imports

import warnings
import numpy as np
from skimage.morphology import disk, square, binary_erosion

#%% Functions

def nanfilt(img, kernel_size=3, method='mean', iterations=1):
    
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

    # Pad img with NaNs
    pad = kernel_size//2
    img = np.pad(img, pad_width=pad, constant_values=np.nan)

    # Define nan_disk
    if kernel_size == 3:
        nan_disk = square(kernel_size, dtype=float)
    else:
        nan_disk = disk(pad, dtype=float)
        nan_disk[nan_disk == 0] = np.nan

    # Find non-NaNs coordinates
    idx = np.where(~np.isnan(img) == True)
    idx_y = idx[0]; idx_x = idx[1]
        
    # Define all kernels
    mesh_range = np.arange((-kernel_size//2)+1, (kernel_size//2)+1)
    mesh_y, mesh_x = np.meshgrid(mesh_range, mesh_range)
    kernel_y = idx_y[:, None, None] + mesh_y
    kernel_x = idx_x[:, None, None] + mesh_x

    # Filter img
    for _ in range(iterations):
        all_kernels = img[kernel_y,kernel_x]*nan_disk
        all_kernels = filt[method](all_kernels, axis=(1, 2))
        all_kernels[np.isnan(all_kernels)] = filt[method](img) # to end while loop
        img[idx] = all_kernels

    # Unpad img
    img = img[pad:-pad,pad:-pad]
    
    return img

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
    nan_disk = disk(pad, dtype=float)
    nan_disk[nan_disk == 0] = np.nan
    
    while np.isnan(img[mask==True]).any():
    
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
            
    # Unpad img
    img = img[pad:-pad,pad:-pad]
    
    return img
