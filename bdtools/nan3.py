#%% Imports

import warnings
import numpy as np
from numba import njit, prange
from skimage.draw import ellipsoid

#%% Function: nanfilt

def nanfilt(
        img,
        mask=None,
        kernel_size=3,
        kernel_type='cuboid',
        filt_method='mean',
        iterations=1,
        parallel=True
        ):
    
    # Nested functions --------------------------------------------------------

    @njit(nogil=True, parallel=parallel)
    def imfilt_mean(img_pad):

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
    
    @njit(nogil=True, parallel=parallel)
    def imfilt_median(img_pad):

        img_filt = img_pad.copy()      
    
        for z in prange(img.shape[0]):
            for y in range(img.shape[1]):
                for x in range(img.shape[2]):
                    
                    if ~np.isnan(img[z,y,x]):
      
                        img_filt[z+pad_z,y+pad_y,x+pad_x] = np.nanmedian(
                            img_pad[
                                z:z+kernel_size_z,
                                y:y+kernel_size_y,
                                x:x+kernel_size_x,
                                ] * strel
                            )            
       
        return img_filt
    
    @njit(nogil=True, parallel=parallel)
    def imfilt_std(img_pad):

        img_filt = img_pad.copy()      
    
        for z in prange(img.shape[0]):
            for y in range(img.shape[1]):
                for x in range(img.shape[2]):
                    
                    if ~np.isnan(img[z,y,x]):
      
                        img_filt[z+pad_z,y+pad_y,x+pad_x] = np.nanstd(
                            img_pad[
                                z:z+kernel_size_z,
                                y:y+kernel_size_y,
                                x:x+kernel_size_x,
                                ] * strel
                            )            
       
        return img_filt

    # Execute -----------------------------------------------------------------

    # Warnings
    warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
    warnings.filterwarnings(action="ignore", message="Mean of empty slice")
    
    # Convert img to float
    img = img.astype(float)
        
    # Add z dimension (if ndim == 2) 
    if img.ndim == 2: 
        img = np.expand_dims(img, 0) 
    
    # Mask operations
    if mask is not None:
              
        # Convert mask to bool
        mask = mask.astype(bool)
        
        # Add z dimension (if ndim == 2) 
        if mask.ndim == 2: 
            mask = np.expand_dims(mask, 0)
            
        # Check mask and img shape and match
        if mask[0,...].shape != img[0,...].shape:
            raise Exception('mask shape is not compatible with img shape')
        elif mask.shape[0] != img.shape[0] and mask.shape[0] != 1:
            raise Exception('mask shape is not compatible with img shape')
        elif mask.shape[0] == 1:
            mask = np.repeat(mask, img.shape[0], axis=0)       
    
        # Set "out of mask" img pixels as NaNs
        img[~mask] = np.nan
        
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
        
    # Pad img
    pad_z = kernel_size_z//2
    pad_y = kernel_size_y//2
    pad_x = kernel_size_x//2
    img_pad = np.pad(img, 
        pad_width=((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x)), 
        constant_values=np.nan
        )
        
    # Define structuring element
    if kernel_type == 'cuboid':
        strel = np.ones(kernel_size)    
    if kernel_type == 'ellipsoid':
        if kernel_size_z == 1:
            strel = ellipsoid(1, pad_y, pad_x, spacing=(2, 1, 1))
        else:    
            strel = ellipsoid(pad_z, pad_y, pad_x, spacing=(1, 1, 1)) 
        strel = strel[1:-1,1:-1,1:-1].astype('float')
        strel[strel == 0] = np.nan
    
    # Filter img
    
    filt = {
        'mean': imfilt_mean, 
        'median': imfilt_median, 
        'std': imfilt_std, 
        }
    
    for _ in range(iterations):
        img_filt = filt[filt_method](img_pad)
        img_pad = img_filt.copy()    
        
    # Unpad img_filt
    if kernel_size_z == 1:
        img_filt = img_filt[:,pad_y:-pad_y,pad_x:-pad_x].squeeze()
    else: 
        img_filt = img_filt[pad_z:-pad_z,pad_y:-pad_y,pad_x:-pad_x].squeeze()

    return img_filt

#%% Function: nanreplace

def nanreplace(
        img,
        mask=None,
        kernel_size=3,
        kernel_type='cuboid',
        filt_method='mean',
        iterations='inf',
        parallel=True
        ):
    
    # Nested functions --------------------------------------------------------

    @njit(nogil=True, parallel=parallel)
    def nanreplace_mean(img_pad):

        img_filt = img_pad.copy()        
    
        for z in prange(img.shape[0]):
            for y in range(img.shape[1]):
                for x in range(img.shape[2]):
                    
                    if np.isnan(img[z,y,x]) and mask[z,y,x] is True:
      
                        img_filt[z+pad_z,y+pad_y,x+pad_x] = np.nanmean(
                            img_pad[
                                z:z+kernel_size_z,
                                y:y+kernel_size_y,
                                x:x+kernel_size_x,
                                ] * strel
                            )
            
        return img_filt
    
    @njit(nogil=True, parallel=parallel)
    def nanreplace_median(img_pad):

        img_filt = img_pad.copy()        
    
        for z in prange(img.shape[0]):
            for y in range(img.shape[1]):
                for x in range(img.shape[2]):
                    
                    if np.isnan(img[z,y,x]) and mask[z,y,x] is True:
      
                        img_filt[z+pad_z,y+pad_y,x+pad_x] = np.nanmedian(
                            img_pad[
                                z:z+kernel_size_z,
                                y:y+kernel_size_y,
                                x:x+kernel_size_x,
                                ] * strel
                            )
            
        return img_filt
    
    @njit(nogil=True, parallel=parallel)
    def nanreplace_std(img_pad):

        img_filt = img_pad.copy()        
    
        for z in prange(img.shape[0]):
            for y in range(img.shape[1]):
                for x in range(img.shape[2]):
                    
                    if np.isnan(img[z,y,x]) and mask[z,y,x] is True:
      
                        img_filt[z+pad_z,y+pad_y,x+pad_x] = np.nanstd(
                            img_pad[
                                z:z+kernel_size_z,
                                y:y+kernel_size_y,
                                x:x+kernel_size_x,
                                ] * strel
                            )
            
        return img_filt

    # Execute -----------------------------------------------------------------

    # Warnings
    warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
    warnings.filterwarnings(action="ignore", message="Mean of empty slice")
    
    # Convert img to float
    img = img.astype(float)
        
    # Add z dimension (if ndim == 2) 
    if img.ndim == 2: 
        img = np.expand_dims(img, 0) 
        
    # Mask operations
    if mask is not None:
              
        # Convert mask to bool
        mask = mask.astype(bool)
        
        # Add z dimension (if ndim == 2) 
        if mask.ndim == 2: 
            mask = np.expand_dims(mask, 0) 
    
        # Check mask and img shape and match
        if mask[0,...].shape != img[0,...].shape:
            raise Exception('mask shape is not compatible with img shape')
        elif mask.shape[0] != img.shape[0] and mask.shape[0] != 1:
            raise Exception('mask shape is not compatible with img shape')
        elif mask.shape[0] == 1:
            mask = np.repeat(mask, img.shape[0], axis=0)
            
    else:
        
        # Create a True mask (to run nested functions)
        mask = np.full_like(img, True, dtype=bool)
            
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
        
    # Pad img and mask
    pad_z = kernel_size_z//2
    pad_y = kernel_size_y//2
    pad_x = kernel_size_x//2
    pad_all = ((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x))
    img_pad = np.pad(img, pad_all, constant_values=np.nan) 
    mask_pad = np.pad(mask, pad_all, constant_values=False) 
    
    # Define structuring element
    if kernel_type == 'cuboid':
        strel = np.ones(kernel_size)    
    if kernel_type == 'ellipsoid':
        if kernel_size_z == 1:
            strel = ellipsoid(1, pad_y, pad_x, spacing=(2, 1, 1))
        else:    
            strel = ellipsoid(pad_z, pad_y, pad_x, spacing=(1, 1, 1)) 
        strel = strel[1:-1,1:-1,1:-1].astype('float')
        strel[strel == 0] = np.nan
    
    # Filter img
    
    filt = {
        'mean': nanreplace_mean, 
        'median': nanreplace_median, 
        'std': nanreplace_std, 
        }
    
    if isinstance(iterations, int):
    
        for _ in range(iterations):                 
            img_filt = filt[filt_method](img_pad)
            img_pad = img_filt.copy()  
            
    elif iterations == 'inf':    
        
        nan_count = np.count_nonzero(np.isnan(img_pad[mask_pad==True]))         
        
        while nan_count > 0:                
            img_filt = filt[filt_method](img_pad)
            img_pad = img_filt.copy()           
            nan_count = np.count_nonzero(np.isnan(img_pad[mask_pad==True]))  
                
    # Unpad img_filt
    if kernel_size_z == 1:
        img_filt = img_filt[:,pad_y:-pad_y,pad_x:-pad_x].squeeze()
    else: 
        img_filt = img_filt[pad_z:-pad_z,pad_y:-pad_y,pad_x:-pad_x].squeeze()

    return img_filt

#%% Tests

import time
from skimage import io 
from pathlib import Path

# -----------------------------------------------------------------------------

stack_name = 'noise+nan(holes)_3d_1024.tif'
mask_name = 'mask_3d_1024.tif'
stackfilt_name = 'filt.tif'

# -----------------------------------------------------------------------------

img = io.imread(Path('../data', stack_name))
mask = io.imread(Path('../data', mask_name))

# -----------------------------------------------------------------------------

# start = time.time()
# print('nanfilt')

# img_filt = nanfilt(
#     img,
#     mask=None,
#     kernel_size=3,
#     kernel_type='cuboid',
#     filt_method='mean',
#     iterations=1,
#     parallel=True    
#     )

# end = time.time()
# print(f'  {(end-start):5.3f} s') 

# # Save img_filt    
# io.imsave(
#     Path('../data', stack_name.replace('.tif', '_filt.tif')),
#     img_filt.astype('float32'),
#     check_contrast=False,
#     )

# -----------------------------------------------------------------------------

start = time.time()
print('nanfilt')

img_filt = nanreplace(
    img,
    mask=None,
    kernel_size=3,
    kernel_type='cuboid',
    filt_method='mean',
    iterations=20,
    parallel=True    
    )

end = time.time()
print(f'  {(end-start):5.3f} s') 

# Save img_filt    
io.imsave(
    Path('../data', stack_name.replace('.tif', '_filt.tif')),
    img_filt.astype('float32'),
    check_contrast=False,
    )