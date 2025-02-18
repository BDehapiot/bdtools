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
        edt_temp[ :,  0] = 0
        edt_temp[ :, -1] = 0
        edt_temp[ 0,  :] = 0
        edt_temp[-1,  :] = 0
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
    overlap = 128 # patch overlap 
    
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
    patches = extract_patches(imgs, size, overlap)
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")
        
    print("merge patches : ", end=" ", flush=True)
    t0 = time.time()
    imgs_merged = merge_patches(patches, imgs.shape, overlap)
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")
    
    # Display
    viewer = napari.Viewer()
    viewer.add_image(np.stack(patches))