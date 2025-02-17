#%% Imports -------------------------------------------------------------------

import numpy as np

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

#%% Function: merge_patches() -------------------------------------------------

# def merge_patches(patches, shape, overlap):
    
#     """ 
#     Reassemble a 2D or 3D ndarray from extract_patches().
    
#     The shape of the original array and the overlap between patches used with
#     extract_patches() must be provided to instruct the reassembly process. 
#     Padded areas are discarded.
    
#     Parameters
#     ----------
#     patches : list of ndarrays
#         List containing extracted patches.
        
#     shape : tuple of int
#         Shape of the original ndarray.
        
#     overlap : int
#         Overlap between patches (Must be between 0 and size - 1).
                
#     Returns
#     -------
#     arr : 2D or 3D ndarray
#         Reassembled array.
    
#     """
    
#     # Get size & dimensions 
#     size = patches[0].shape[0]
#     if len(shape) == 2: 
#         nT = 1; 
#         nY, nX = shape
#     if len(shape) == 3: 
#         nT, nY, nX = shape
#     nPatch = len(patches) // nT

#     # Get variables
#     y0s = np.arange(0, nY, size - overlap)
#     x0s = np.arange(0, nX, size - overlap)
#     yMax = y0s[-1] + size
#     xMax = x0s[-1] + size
#     yPad = yMax - nY
#     xPad = xMax - nX
#     yPad1 = yPad // 2
#     xPad1 = xPad // 2

#     # Merge patches
#     def _merge_patches(patches):
#         count = 0
#         arr = np.full((2, nY + yPad, nX + xPad), np.nan)
#         for i, y0 in enumerate(y0s):
#             for j, x0 in enumerate(x0s):
#                 if i % 2 == j % 2:
#                     arr[0, y0:y0 + size, x0:x0 + size] = patches[count]
#                 else:
#                     arr[1, y0:y0 + size, x0:x0 + size] = patches[count]
#                 count += 1 
#         arr = np.nanmean(arr, axis=0)
#         arr = arr[yPad1:yPad1 + nY, xPad1:xPad1 + nX]
#         return arr
    
#     if len(shape) == 2:
#         arr = _merge_patches(patches)

#     if len(shape) == 3:
#         patches = np.stack(patches).reshape(nT, nPatch, size, size)
#         arr = Parallel(n_jobs=-1)(
#             delayed(_merge_patches)(patches[t,...])
#             for t in range(nT)
#             )
#         arr = np.stack(arr)

#     return arr

# def merge_patches(patches, shape, overlap):
    
#     """ 
#     Reassemble a 2D or 3D ndarray from extract_patches().
    
#     The shape of the original array and the overlap between patches used with
#     extract_patches() must be provided to instruct the reassembly process. 
#     When merging patches with overlap, priority is given to the central regions
#     of the overlapping patches.
    
#     Parameters
#     ----------
#     patches : list of ndarrays
#         List containing extracted patches.
        
#     shape : tuple of int
#         Shape of the original ndarray.
        
#     overlap : int
#         Overlap between patches (Must be between 0 and size - 1).
                
#     Returns
#     -------
#     arr : 2D or 3D ndarray
#         Reassembled array.
    
#     """
    
#     # Nested function(s) ------------------------------------------------------
    
#     def get_patch_edt(patches):
#         edt = np.full_like(patches[0], 1)
#         edt[:, 0] = 0; edt[:, -1] = 0
#         edt[0, :] = 0; edt[-1, :] = 0
#         return distance_transform_edt(edt) + 1
    
#     # Execute -----------------------------------------------------------------
    
#     # Get size & dimensions 
#     size = patches[0].shape[0]
#     if len(shape) == 2: 
#         nT = 1; 
#         nY, nX = shape
#     if len(shape) == 3: 
#         nT, nY, nX = shape
#     nPatch = len(patches) // nT

#     # Get variables
#     patch_edt = get_patch_edt(patches)
#     y0s = np.arange(0, nY, size - overlap)
#     x0s = np.arange(0, nX, size - overlap)
#     yMax = y0s[-1] + size
#     xMax = x0s[-1] + size
#     yPad = yMax - nY
#     xPad = xMax - nX
#     yPad1 = yPad // 2
#     xPad1 = xPad // 2

#     # Merge patches
#     def _merge_patches(patches):
#         count = 0
#         arr = np.full((nY + yPad, nX + xPad), 0, dtype=patches[0].dtype)
#         edt = np.full((nY + yPad, nX + xPad), 0)
#         for i, y0 in enumerate(y0s):
#             for j, x0 in enumerate(x0s):
#                 tmp_arr = np.full((nY + yPad, nX + xPad), np.nan)
#                 tmp_edt = np.full((nY + yPad, nX + xPad), np.nan)
#                 tmp_arr[y0:y0 + size, x0:x0 + size] = patches[count]
#                 tmp_edt[y0:y0 + size, x0:x0 + size] = patch_edt
#                 idx = np.where(tmp_edt > edt)
#                 edt[idx] = tmp_edt[idx]
#                 arr[idx] = tmp_arr[idx]
#                 count += 1 
#         arr = arr[yPad1:yPad1 + nY, xPad1:xPad1 + nX]
#         return arr
    
#     if len(shape) == 2:
#         arr = _merge_patches(patches)

#     if len(shape) == 3:
#         patches = np.stack(patches).reshape(nT, nPatch, size, size)
#         arr = Parallel(n_jobs=-1)(
#             delayed(_merge_patches)(patches[t,...])
#             for t in range(nT)
#             )
#         arr = np.stack(arr)

#     return arr

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

    # Nested function(s) ------------------------------------------------------
    
    def get_patch_edt(patch_shape):
        edt = np.full(patch_shape, 1)
        edt[:, 0] = 0; edt[:, -1] = 0
        edt[0, :] = 0; edt[-1, :] = 0
        return distance_transform_edt(edt) + 1
    
    def _merge_patches(patches):
        arr_sum = np.zeros((nY + yPad, nX + xPad), dtype=np.float64)
        weight_sum = np.zeros((nY + yPad, nX + xPad), dtype=np.float64)
        count = 0
        for i, y0 in enumerate(y0s):
            for j, x0 in enumerate(x0s):
                patch = patches[count].astype(np.float64)
                arr_sum[y0:y0+size, x0:x0+size] += patch * patch_edt
                weight_sum[y0:y0+size, x0:x0+size] += patch_edt
                count += 1
        arr = np.divide(
            arr_sum, weight_sum, 
            out=np.zeros_like(arr_sum),
            where=weight_sum != 0
            )
        arr = arr[yPad1:yPad1 + nY, xPad1:xPad1 + nX]
        return arr

    # Execute -----------------------------------------------------------------

    # Get size & dimensions
    size = patches[0].shape[0]
    if len(shape) == 2:
        nT = 1
        nY, nX = shape
    if len(shape) == 3:
        nT, nY, nX = shape
    nPatch = len(patches) // nT

    # Get variables
    patch_shape = patches[0].shape
    patch_edt = get_patch_edt(patch_shape).astype(np.float64)
    y0s = np.arange(0, nY, size - overlap)
    x0s = np.arange(0, nX, size - overlap)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1 = yPad // 2
    xPad1 = xPad // 2

    if len(shape) == 2:
        arr = _merge_patches(patches)

    if len(shape) == 3:
        patches = np.stack(patches).reshape(nT, nPatch, size, size)
        arr_list = []
        for t in range(nT):
            arr_t = _merge_patches(patches[t])
            arr_list.append(arr_t)
            arr = np.stack(arr_list)

    arr = arr.astype(patches[0].dtype)

    return arr