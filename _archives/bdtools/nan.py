#%% Imports -------------------------------------------------------------------

import warnings
import numpy as np
from numba import njit, prange

# Skimage
from skimage.draw import ellipsoid

#%% Function: nan_filt() ------------------------------------------------------

def nan_filt(
        arr,
        mask=None,
        kernel_size=3,
        kernel_shape='cuboid',
        filt_method='mean',
        iterations=1,
        parallel=True
        ):
    
    """ 
    Filter image ignoring NaNs.
    
    Parameters
    ----------
    arr : 2D or 3D ndarray (float)
        Image to be filtered.
        
    mask : 2D or 3D ndarray (bool)
        Only pixels within mask will be filtered.
        
    kernel_size : int or tuple of int
        Filtering kernel size, should be odd.
        Use tuple of int to specify kernel sizes for each dimension.
                
    kernel_shape : int
        Filtering kernel shape, 'cuboid' or 'ellipsoid'.
        
    filt_method : str
        Filtering method, 'mean', 'median' or 'std'.
        
    iterations : int
        Iterations of filtering process.
        
    parallel : bool
        Use or not parallel processing.
    
    Returns
    -------  
    arr_filt : 2D or 3D ndarray
        Processed image.
    
    """
    
    # Nested functions --------------------------------------------------------

    @njit(nogil=True, parallel=parallel)
    def imfilt_mean(arr_pad):

        arr_filt = arr_pad.copy()      
    
        for z in prange(arr.shape[0]):
            for y in range(arr.shape[1]):
                for x in range(arr.shape[2]):
                    
                    if ~np.isnan(arr[z,y,x]):
      
                        arr_filt[z+pad_z,y+pad_y,x+pad_x] = np.nanmean(
                            arr_pad[
                                z:z+kernel_size_z,
                                y:y+kernel_size_y,
                                x:x+kernel_size_x,
                                ] * strel
                            )            
       
        return arr_filt
    
    @njit(nogil=True, parallel=parallel)
    def imfilt_median(arr_pad):

        arr_filt = arr_pad.copy()      
    
        for z in prange(arr.shape[0]):
            for y in range(arr.shape[1]):
                for x in range(arr.shape[2]):
                    
                    if ~np.isnan(arr[z,y,x]):
      
                        arr_filt[z+pad_z,y+pad_y,x+pad_x] = np.nanmedian(
                            arr_pad[
                                z:z+kernel_size_z,
                                y:y+kernel_size_y,
                                x:x+kernel_size_x,
                                ] * strel
                            )            
       
        return arr_filt
    
    @njit(nogil=True, parallel=parallel)
    def imfilt_std(arr_pad):

        arr_filt = arr_pad.copy()      
    
        for z in prange(arr.shape[0]):
            for y in range(arr.shape[1]):
                for x in range(arr.shape[2]):
                    
                    if ~np.isnan(arr[z,y,x]):
      
                        arr_filt[z+pad_z,y+pad_y,x+pad_x] = np.nanstd(
                            arr_pad[
                                z:z+kernel_size_z,
                                y:y+kernel_size_y,
                                x:x+kernel_size_x,
                                ] * strel
                            )            
       
        return arr_filt

    # Execute -----------------------------------------------------------------

    # Warnings
    warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
    warnings.filterwarnings(action="ignore", message="Mean of empty slice")
    
    # Convert arr to float
    arr = arr.astype(float)
        
    # Add z dimension (if ndim == 2) 
    if arr.ndim == 2: 
        arr = np.expand_dims(arr, 0) 
    
    # Mask operations
    if mask is not None:
              
        # Convert mask to bool
        mask = mask.astype(bool)
        
        # Add z dimension (if ndim == 2) 
        if mask.ndim == 2: 
            mask = np.expand_dims(mask, 0)
            
        # Check mask and arr shape and match
        if mask[0,...].shape != arr[0,...].shape:
            raise Exception('mask shape is not compatible with arr shape')
        elif mask.shape[0] != arr.shape[0] and mask.shape[0] != 1:
            raise Exception('mask shape is not compatible with arr shape')
        elif mask.shape[0] == 1:
            mask = np.repeat(mask, arr.shape[0], axis=0)       
    
        # Set "out of mask" arr pixels as NaNs
        arr[~mask] = np.nan
        
    # Extract kernel_size variables
    if isinstance(kernel_size, int):
        if arr.ndim == 2:
            kernel_size_z = 1  
            kernel_size_y = kernel_size
            kernel_size_x = kernel_size       
        elif arr.ndim == 3:
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
        
    # Pad arr
    pad_z = kernel_size_z//2
    pad_y = kernel_size_y//2
    pad_x = kernel_size_x//2
    arr_pad = np.pad(arr, 
        pad_width=((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x)), 
        constant_values=np.nan
        )
        
    # Define structuring element
    if kernel_shape == 'cuboid':
        strel = np.ones(kernel_size)    
    if kernel_shape == 'ellipsoid':
        if kernel_size_z == 1:
            strel = ellipsoid(1, pad_y, pad_x, spacing=(2, 1, 1))
        else:    
            strel = ellipsoid(pad_z, pad_y, pad_x, spacing=(1, 1, 1)) 
        strel = strel[1:-1,1:-1,1:-1].astype('float')
        strel[strel == 0] = np.nan
    
    # Filter arr
    
    filt = {
        'mean': imfilt_mean, 
        'median': imfilt_median, 
        'std': imfilt_std, 
        }
    
    for _ in range(iterations):
        arr_filt = filt[filt_method](arr_pad)
        arr_pad = arr_filt.copy()    
        
    # Unpad arr_filt
    z_slice = slice(None) if kernel_size_z == 1 else slice(pad_z, -pad_z)
    y_slice = slice(None) if kernel_size_y == 1 else slice(pad_y, -pad_y)
    x_slice = slice(None) if kernel_size_x == 1 else slice(pad_x, -pad_x)
    arr_filt = arr_filt[z_slice, y_slice, x_slice].squeeze()

    return arr_filt

#%% Function: nan_replace() ----------------------------------------------------

def nan_replace(
        arr,
        mask=None,
        kernel_size=3,
        kernel_shape='cuboid',
        filt_method='mean',
        iterations='inf',
        parallel=True
        ):
    
    """ 
    Replace NaNs using filtering kernel.    
    
    The function iterates to replace NaNs connected to real numbers 
    until no more NaNs are found. A mask can be provided to select
    NaNs to be replaced.
    
    Parameters
    ----------
    arr : 2D or 3D ndarray (float)
        Image to be filtered.
        
    mask : 2D or 3D ndarray (bool)
        Only NaNs within mask will be replaced.
        
    kernel_size : int or tuple of int
        Filtering kernel size, should be odd.
        Use tuple of int to specify kernel sizes for each dimension.
                
    kernel_shape : int
        Filtering kernel shape, 'cuboid' or 'ellipsoid'.
        
    filt_method : str
        Filtering method, 'mean', 'median' or 'std'.
        
    iterations : int
        Iterations of replacing process.
        Use 'inf' to replace all NaNs.
        
    parallel : bool
        Use or not parallel processing.
    
    Returns
    -------  
    arr_filt : 2D or 3D ndarray
        Processed image.
    
    """
    
    # Nested functions --------------------------------------------------------

    @njit(nogil=True, parallel=parallel)
    def nan_replace_mean(arr_pad):

        arr_filt = arr_pad.copy()        
    
        for z in prange(arr.shape[0]):
            for y in range(arr.shape[1]):
                for x in range(arr.shape[2]):
                    
                    if np.isnan(arr[z,y,x]) and mask[z,y,x] is True:
      
                        arr_filt[z+pad_z,y+pad_y,x+pad_x] = np.nanmean(
                            arr_pad[
                                z:z+kernel_size_z,
                                y:y+kernel_size_y,
                                x:x+kernel_size_x,
                                ] * strel
                            )
            
        return arr_filt
    
    @njit(nogil=True, parallel=parallel)
    def nan_replace_median(arr_pad):

        arr_filt = arr_pad.copy()        
    
        for z in prange(arr.shape[0]):
            for y in range(arr.shape[1]):
                for x in range(arr.shape[2]):
                    
                    if np.isnan(arr[z,y,x]) and mask[z,y,x] is True:
      
                        arr_filt[z+pad_z,y+pad_y,x+pad_x] = np.nanmedian(
                            arr_pad[
                                z:z+kernel_size_z,
                                y:y+kernel_size_y,
                                x:x+kernel_size_x,
                                ] * strel
                            )
            
        return arr_filt
    
    @njit(nogil=True, parallel=parallel)
    def nan_replace_std(arr_pad):

        arr_filt = arr_pad.copy()        
    
        for z in prange(arr.shape[0]):
            for y in range(arr.shape[1]):
                for x in range(arr.shape[2]):
                    
                    if np.isnan(arr[z,y,x]) and mask[z,y,x] is True:
      
                        arr_filt[z+pad_z,y+pad_y,x+pad_x] = np.nanstd(
                            arr_pad[
                                z:z+kernel_size_z,
                                y:y+kernel_size_y,
                                x:x+kernel_size_x,
                                ] * strel
                            )
            
        return arr_filt

    # Execute -----------------------------------------------------------------

    # Warnings
    warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
    warnings.filterwarnings(action="ignore", message="Mean of empty slice")
    
    # Convert arr to float
    arr = arr.astype(float)
        
    # Add z dimension (if ndim == 2) 
    if arr.ndim == 2: 
        arr = np.expand_dims(arr, 0) 
        
    # Mask operations
    if mask is not None:
              
        # Convert mask to bool
        mask = mask.astype(bool)
        
        # Add z dimension (if ndim == 2) 
        if mask.ndim == 2: 
            mask = np.expand_dims(mask, 0) 
    
        # Check mask and arr shape and match
        if mask[0,...].shape != arr[0,...].shape:
            raise Exception('mask shape is not compatible with arr shape')
        elif mask.shape[0] != arr.shape[0] and mask.shape[0] != 1:
            raise Exception('mask shape is not compatible with arr shape')
        elif mask.shape[0] == 1:
            mask = np.repeat(mask, arr.shape[0], axis=0)
            
    else:
        
        # Create a True mask (to run nested functions)
        mask = np.full_like(arr, True, dtype=bool)
            
    # Extract kernel_size variables
    if isinstance(kernel_size, int):
        if arr.ndim == 2:
            kernel_size_z = 1  
            kernel_size_y = kernel_size
            kernel_size_x = kernel_size       
        elif arr.ndim == 3:
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
        
    # Pad arr and mask
    pad_z = kernel_size_z//2
    pad_y = kernel_size_y//2
    pad_x = kernel_size_x//2
    pad_all = ((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x))
    arr_pad = np.pad(arr, pad_all, constant_values=np.nan) 
    mask_pad = np.pad(mask, pad_all, constant_values=False) 
    
    # Define structuring element
    if kernel_shape == 'cuboid':
        strel = np.ones(kernel_size)    
    if kernel_shape == 'ellipsoid':
        if kernel_size_z == 1:
            strel = ellipsoid(1, pad_y, pad_x, spacing=(2, 1, 1))
        else:    
            strel = ellipsoid(pad_z, pad_y, pad_x, spacing=(1, 1, 1)) 
        strel = strel[1:-1,1:-1,1:-1].astype('float')
        strel[strel == 0] = np.nan
    
    # Filter arr
    
    filt = {
        'mean': nan_replace_mean, 
        'median': nan_replace_median, 
        'std': nan_replace_std, 
        }
    
    if isinstance(iterations, int):
    
        for _ in range(iterations):                 
            arr_filt = filt[filt_method](arr_pad)
            arr_pad = arr_filt.copy()  
            
    elif iterations == 'inf':    
        
        nan_count = np.count_nonzero(np.isnan(arr_pad[mask_pad==True]))         
        
        while nan_count > 0:                
            arr_filt = filt[filt_method](arr_pad)
            arr_pad = arr_filt.copy()           
            nan_count = np.count_nonzero(np.isnan(arr_pad[mask_pad==True]))  
                
    # Unpad arr_filt
    z_slice = slice(None) if kernel_size_z == 1 else slice(pad_z, -pad_z)
    y_slice = slice(None) if kernel_size_y == 1 else slice(pad_y, -pad_y)
    x_slice = slice(None) if kernel_size_x == 1 else slice(pad_x, -pad_x)
    arr_filt = arr_filt[z_slice, y_slice, x_slice].squeeze()

    return arr_filt