#%% Imports -------------------------------------------------------------------

import numpy as np
from numba import njit

# Scipy
from scipy.ndimage import distance_transform_edt

#%% Function: extract_patches() -----------------------------------------------

def extract_patches(arr, size, overlap):
    
    """ 
    Extract patches from 2D or 3D ndarray.    
    
    For 3D array, patches are extracted from each 2D slice along the first 
    dimension. If necessary, the input array is padded using 'reflect' 
    padding mode.
    
    Parameters
    ----------
    arr : 2D or 3D ndarray
        Array to be patched.
        
    size : int
        Size of extracted patches.
        
    overlap : int
        Overlap between patches (Must be between 0 and size - 1).
                
    Returns
    -------  
    patches : list of ndarrays
        List containing extracted patches
    
    """
    
    # Get dimensions
    if arr.ndim == 2: 
        nT = 1
        nY, nX = arr.shape 
    if arr.ndim == 3: 
        nT, nY, nX = arr.shape
    
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

#%% 

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def extract_patches(arr, size, overlap):
    """
    Extract patches from a 2D or 3D ndarray.

    For 3D arrays, patches are extracted from each 2D slice along the first 
    dimension. If necessary, the input array is padded using 'reflect' mode.

    Parameters
    ----------
    arr : ndarray (2D or 3D)
        Array to be patched.
    size : int
        Size of extracted patches.
    overlap : int
        Overlap between patches (must be between 0 and size - 1).

    Returns
    -------
    patches : ndarray
        For 2D: an array of shape (n_patches, size, size).
        For 3D: an array of shape (n_patches_total, size, size) where patches
        are extracted from each 2D slice along the first dimension.
    """
    # Determine dimensions
    if arr.ndim == 2:
        nT = 1
        nY, nX = arr.shape
    elif arr.ndim == 3:
        nT, nY, nX = arr.shape
    else:
        raise ValueError("Input array must be 2D or 3D.")

    # The step between patches is (size - overlap)
    step = size - overlap

    # Compute patch starting indices and required padding
    y0s = np.arange(0, nY, step)
    x0s = np.arange(0, nX, step)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1, yPad2 = yPad // 2, (yPad + 1) // 2
    xPad1, xPad2 = xPad // 2, (xPad + 1) // 2

    if arr.ndim == 2:
        # Pad the array
        arr_pad = np.pad(arr, ((yPad1, yPad2), (xPad1, xPad2)), mode='reflect')
        # Create a sliding window view: shape is (new_y, new_x, size, size)
        windows = sliding_window_view(arr_pad, (size, size))
        # Downsample the view to pick only the windows we want
        patches = windows[::step, ::step].reshape(-1, size, size)
    else:  # arr.ndim == 3
        # Pad only spatial dimensions; do not pad along the time axis.
        arr_pad = np.pad(arr, ((0, 0), (yPad1, yPad2), (xPad1, xPad2)), mode='reflect')
        patch_list = []
        # Process each 2D slice independently
        for t in range(nT):
            windows = sliding_window_view(arr_pad[t], (size, size))
            patches_t = windows[::step, ::step].reshape(-1, size, size)
            patch_list.append(patches_t)
        patches = np.concatenate(patch_list, axis=0)
        
    return patches


#%% Function: merge_patches() -------------------------------------------------

@njit
def merge_patches_2d_numba(patches, patch_edt, arr, edt, y0s, x0s, size):
    count = 0
    ny0 = y0s.shape[0]
    nx0 = x0s.shape[0]
    for i0 in range(ny0):
        y0 = y0s[i0]
        for j0 in range(nx0):
            x0 = x0s[j0]
            for i in range(size):
                for j in range(size):
                    y_idx = y0 + i
                    x_idx = x0 + j
                    if patch_edt[i, j] > edt[y_idx, x_idx]:
                        edt[y_idx, x_idx] = patch_edt[i, j]
                        arr[y_idx, x_idx] = patches[count, i, j]
            count += 1

def merge_patches(patches, shape, overlap):
    
    """ 
    Reassemble a 2D or 3D ndarray from extract_patches().
    
    The shape of the original array and the overlap between patches used with
    extract_patches() must be provided to instruct the reassembly process. 
    When merging patches with overlap, priority is given to the central regions
    of the overlapping patches.
    
    Parameters
    ----------
    patches : list of ndarrays
        List containing extracted patches.
        
    shape : tuple of int
        Shape of the original ndarray.
        
    overlap : int
        Overlap between patches (Must be between 0 and size - 1).
                
    Returns
    -------
    arr : 2D or 3D ndarray
        Reassembled array.
    
    """
    
    def get_patch_edt(patch_shape):
        edt_temp = np.ones(patch_shape, dtype=float)
        edt_temp[:, 0] = 0
        edt_temp[:, -1] = 0
        edt_temp[0, :] = 0
        edt_temp[-1, :] = 0
        return distance_transform_edt(edt_temp) + 1

    # Get size & dimensions 
    size = patches[0].shape[0]
    if len(shape) == 2:
        nT = 1
        nY, nX = shape
    elif len(shape) == 3:
        nT, nY, nX = shape
    else:
        raise ValueError("shape must be 2D or 3D")
    nPatch = len(patches) // nT

    # Get patch edt
    patch_edt = get_patch_edt(patches[0].shape)
    
    # Get variables
    y0s = np.arange(0, nY, size - overlap)
    x0s = np.arange(0, nX, size - overlap)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1 = yPad // 2
    xPad1 = xPad // 2

    # Initialize arrays
    y0s_arr = np.array(y0s, dtype=np.int64)
    x0s_arr = np.array(x0s, dtype=np.int64)

    # Merge patches (2D)
    if len(shape) == 2:
        out_shape = (nY + yPad, nX + xPad)
        arr_out = np.zeros(out_shape, dtype=patches[0].dtype)
        edt_out = np.zeros(out_shape, dtype=patch_edt.dtype)
        patches_array = np.stack(patches)
        merge_patches_2d_numba(patches_array, patch_edt, arr_out, edt_out,
                               y0s_arr, x0s_arr, size)
        
        return arr_out[yPad1:yPad1 + nY, xPad1:xPad1 + nX]

    # Merge patches (3D)
    elif len(shape) == 3:
        patches_array = np.stack(patches).reshape(nT, nPatch, size, size)
        merged_slices = []
        for t in range(nT):
            out_shape = (nY + yPad, nX + xPad)
            arr_out = np.zeros(out_shape, dtype=patches_array.dtype)
            edt_out = np.zeros(out_shape, dtype=patch_edt.dtype)
            merge_patches_2d_numba(patches_array[t], patch_edt, arr_out, edt_out,
                                   y0s_arr, x0s_arr, size)
            merged_slice = arr_out[yPad1:yPad1 + nY, xPad1:xPad1 + nX]
            merged_slices.append(merged_slice)
        
        return np.stack(merged_slices)