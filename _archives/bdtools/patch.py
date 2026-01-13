#%% Imports -------------------------------------------------------------------

import numba
import numpy as np
from joblib import Parallel, delayed

# Scipy
from scipy.ndimage import distance_transform_edt

#%% Function: extract_patches() -----------------------------------------------

def extract_patches(arr, size, patch_overlap):
    
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
        
    patch_overlap : int
        patch_overlap between patches (Must be between 0 and size - 1).
                
    Returns
    -------  
    patches : list of ndarrays
        List containing extracted patches
    
    """
    
    # Get dimensions
    if arr.ndim == 2: 
        nS = 1
        nY, nX = arr.shape 
    if arr.ndim == 3: 
        nS, nY, nX = arr.shape
    
    # Get variables
    y0s = np.arange(0, nY, size - patch_overlap)
    x0s = np.arange(0, nX, size - patch_overlap)
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
        for t in range(nS):
            for y0 in y0s:
                for x0 in x0s:
                    patches.append(arr_pad[t, y0:y0 + size, x0:x0 + size])
            
    return patches

#%% Function : merge_patches() ------------------------------------------------

# def merge_patches(patches, shape, patch_overlap):
    
#     """
#     Reassemble a 2D or 3D ndarray from extract_patches().

#     The shape of the original array and the patch_overlap between patches used with
#     extract_patches() must be provided to instruct the reassembly process.
#     When merging patches with patch_overlap, priority is given to the central regions
#     of the overlapping patches.

#     Parameters
#     ----------
#     patches : list of ndarrays
#         List containing extracted patches.

#     shape : tuple of int
#         Shape of the original ndarray.

#     patch_overlap : int
#         patch_overlap between patches (Must be between 0 and size - 1).

#     Returns
#     -------
#     arr : 2D or 3D ndarray
#         Reassembled array.
    
#     """

#     # Nested function(s) ------------------------------------------------------
    
#     def get_patch_edt(patch_shape):
#         edt = np.full(patch_shape, 1)
#         edt[:, 0] = 0; edt[:, -1] = 0
#         edt[0, :] = 0; edt[-1, :] = 0
#         return distance_transform_edt(edt) + 1
    
#     def _merge_patches(patches):
#         arr_sum = np.zeros((nY + yPad, nX + xPad), dtype=np.float64)
#         weight_sum = np.zeros((nY + yPad, nX + xPad), dtype=np.float64)
#         count = 0
#         for i, y0 in enumerate(y0s):
#             for j, x0 in enumerate(x0s):
#                 patch = patches[count].astype(np.float64)
#                 arr_sum[y0:y0+size, x0:x0+size] += patch * patch_edt
#                 weight_sum[y0:y0+size, x0:x0+size] += patch_edt
#                 count += 1
#         arr = np.divide(
#             arr_sum, weight_sum, 
#             out=np.zeros_like(arr_sum),
#             where=weight_sum != 0
#             )
#         arr = arr[yPad1:yPad1 + nY, xPad1:xPad1 + nX]
#         return arr

#     # Execute -----------------------------------------------------------------

#     # Get size & dimensions
#     size = patches[0].shape[0]
#     if len(shape) == 2:
#         nS = 1
#         nY, nX = shape
#     if len(shape) == 3:
#         nS, nY, nX = shape
#     nPatch = len(patches) // nS

#     # Get variables
#     patch_shape = patches[0].shape
#     patch_edt = get_patch_edt(patch_shape).astype(np.float64)
#     y0s = np.arange(0, nY, size - patch_overlap)
#     x0s = np.arange(0, nX, size - patch_overlap)
#     yMax = y0s[-1] + size
#     xMax = x0s[-1] + size
#     yPad = yMax - nY
#     xPad = xMax - nX
#     yPad1 = yPad // 2
#     xPad1 = xPad // 2

#     if len(shape) == 2:
#         arr = _merge_patches(patches)

#     if len(shape) == 3:
#         patches = np.stack(patches).reshape(nS, nPatch, size, size)
#         arr_list = []
#         for t in range(nS):
#             arr_t = _merge_patches(patches[t])
#             arr_list.append(arr_t)
#             arr = np.stack(arr_list)

#     arr = arr.astype(patches[0].dtype)

#     return arr

#%% Function : merge_patches() ------------------------------------------------

@numba.njit(parallel=True)
def merge_2d_numba(
        patches, patch_edt, size, nY, nX, y0s, x0s, yPad, xPad, yPad1, xPad1):
    
    out_h = nY + yPad
    out_w = nX + xPad
    arr_sum = np.zeros((out_h, out_w), dtype=np.float64)
    weight_sum = np.zeros((out_h, out_w), dtype=np.float64)
    count = 0
    
    for i in range(y0s.shape[0]):
        for j in range(x0s.shape[0]):
            patch = patches[count]
            for di in range(size):
                for dj in range(size):
                    r = y0s[i] + di
                    c = x0s[j] + dj
                    arr_sum[r, c] += patch[di, dj] * patch_edt[di, dj]
                    weight_sum[r, c] += patch_edt[di, dj]
            count += 1

    merged = np.empty((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            if weight_sum[i, j] != 0:
                merged[i, j] = arr_sum[i, j] / weight_sum[i, j]
            else:
                merged[i, j] = 0.0
    
    return merged[yPad1:yPad1+nY, xPad1:xPad1+nX]

def merge_patches(patches, shape, patch_overlap):
    
    """
    Reassemble a 2D or 3D ndarray from extract_patches().

    The shape of the original array and the patch_overlap between patches 
    used with extract_patches() must be provided to instruct the reassembly 
    process. When merging patches with patch_overlap, priority is given to the 
    central regions of the overlapping patches.

    Parameters
    ----------
    patches : list of ndarrays
        List containing extracted patches.

    shape : tuple of int
        Shape of the original ndarray.

    patch_overlap : int
        patch_overlap between patches (Must be between 0 and size - 1).

    Returns
    -------
    arr : 2D or 3D ndarray
        Reassembled array.
    
    """
    
    def get_patch_edt(size):
        edt = np.ones((size, size), dtype=np.float64)
        edt[[0, -1], :] = 0
        edt[:, [0, -1]] = 0
        return distance_transform_edt(edt) + 1

    # Get size & dimensions
    size = patches[0].shape[0]
    if len(shape) == 2:
        nS = 1
        nY, nX = shape
    elif len(shape) == 3:
        nS, nY, nX = shape
    else:
        raise ValueError("Shape must be 2D or 3D")

    # Get variables
    size = patches[0].shape[0]
    step = size - patch_overlap
    patch_edt = get_patch_edt(size)
    y0s = np.arange(0, nY, step)
    x0s = np.arange(0, nX, step)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1 = yPad // 2
    xPad1 = xPad // 2

    if nS == 1:
        patches_arr = np.array([p.astype(np.float64) for p in patches])
        merged = merge_2d_numba(
            patches_arr, patch_edt, size, nY, nX, 
            y0s, x0s, yPad, xPad, yPad1, xPad1
            )
    else:
        patches_arr = np.array([p.astype(np.float64) for p in patches])
        patches_arr = patches_arr.reshape(nS, -1, size, size)
        merged_slices = []
        for t in range(nS):
            merged_slice = merge_2d_numba(
                patches_arr[t], patch_edt, size, nY, nX, 
                y0s, x0s, yPad, xPad, yPad1, xPad1
                )
            merged_slices.append(merged_slice)
        merged = np.stack(merged_slices, axis=0)

    return merged.astype(patches[0].dtype)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports
    import time
    import napari
    from skimage import io
    from pathlib import Path

    # Parameters
    dataset = "em_mito"
    # dataset = "fluo_nuclei"
    size = 256 # patch size
    patch_overlap = 128 # patch patch_overlap 
    
    # Paths
    local_path = Path.cwd().parent / "_local"
    img_path = local_path / f"{dataset}" / f"{dataset}_trn.tif"
    msk_path = local_path / f"{dataset}" / f"{dataset}_msk_trn.tif"
    
    # Load images & masks
    imgs = io.imread(img_path)
    msks = io.imread(msk_path)
        
    # Patch tests
    print("extract patches : ", end=" ", flush=True)
    t0 = time.time()
    patches = extract_patches(imgs, size, patch_overlap)
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")
        
    print("merge patches : ", end=" ", flush=True)
    t0 = time.time()
    imgs_merged = merge_patches(patches, imgs.shape, patch_overlap)
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")
    
    # Display
    viewer = napari.Viewer()
    # viewer.add_image(np.stack(patches))
    viewer.add_image(np.stack(imgs_merged))