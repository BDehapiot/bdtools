#%% Imports -------------------------------------------------------------------

import numba
import numpy as np

# Scipy
from scipy.ndimage import distance_transform_edt

#%% Function: get_patches() -----------------------------------------------

def get_patches(arr, size, overlap):
    
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
        overlap between patches (Must be between 0 and size - 1).
                
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
    if arr.ndim == 4:
        nS, nY, nX, nC = arr.shape
    
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
        for t in range(nS):
            for y0 in y0s:
                for x0 in x0s:
                    patches.append(arr_pad[t, y0:y0 + size, x0:x0 + size])
            
    return patches

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

def merge_patches(patches, shape, overlap):
    
    """
    Reassemble a 2D or 3D ndarray from get_patches().

    The shape of the original array and the overlap between patches 
    used with get_patches() must be provided to instruct the reassembly 
    process. When merging patches with overlap, priority is given to the 
    central regions of the overlapping patches.

    Parameters
    ----------
    patches : list of ndarrays
        List containing extracted patches.

    shape : tuple of int
        Shape of the original ndarray.

    overlap : int
        overlap between patches (Must be between 0 and size - 1).

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
    step = size - overlap
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
    import numpy as np
    from skimage import io
    from pathlib import Path
    from skimage.measure import label
    
    # -------------------------------------------------------------------------
    
    def load_data(paths):
        data = []
        for path in paths:
            data.append(io.imread(path))
        if len(data) == 1:
            data = np.stack(data).squeeze()
        return data
    
    def prep_mask(msk):
        msk_1 = label(msk == 1)
        msk_2 = label(msk == 2)
        msk_3 = label(msk == 3)
        msk_2[msk_2 > 0] += np.max(msk_1)
        msk_3[msk_3 > 0] += np.max(msk_2)
        return msk_1 + msk_2 + msk_3
    
    # Load --------------------------------------------------------------------
    
    # Paths
    # dataset = "em_mito"
    # dataset = "fluo_tissue"
    # dataset = "fluo_nuclei_instance"
    # dataset = "fluo_nuclei_semantic"
    dataset = "sat_roads"
    data_path = Path.cwd().parent / "_local" / dataset
    raw_trn_paths = list(data_path.rglob("*raw_trn.tif"))
    msk_trn_paths = list(data_path.rglob("*msk_trn.tif"))
    
    # Load data
    raw_trn = load_data(raw_trn_paths)
    msk_trn = load_data(msk_trn_paths)
    if "nuclei_semantic" in dataset:
        msk_trn = prep_mask(msk_trn)
        
    # -------------------------------------------------------------------------
        
    idx = 0
    data = raw_trn
    
#%% ---------------------------------------------------------------------------

    # Parameters
    size = 256
    overlap = 128 
    
    # Initialize
    if isinstance(data, list):
        arr = data[idx]
    else:
        arr = data

    # -------------------------------------------------------------------------

    # Get dimensions
    if arr.ndim == 2: 
        nS = 1
        nY, nX = arr.shape 
    if arr.ndim == 3: 
        nS, nY, nX = arr.shape
    if arr.ndim == 4:
        nS, nY, nX, nC = arr.shape
        
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
    if arr.ndim == 4:
        arr_pad = np.pad(
            arr, ((0, 0), (yPad1, yPad2), (xPad1, xPad2), (0, 0)), mode='reflect')
        
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
    if arr.ndim == 4:
        for t in range(nS):
            for y0 in y0s:
                for x0 in x0s:
                    patches.append(arr_pad[t, y0:y0 + size, x0:x0 + size, :])
                
    # Display
    viewer = napari.Viewer()
    viewer.add_image(np.stack(patches))
    
#%% get_patches() -------------------------------------------------------------
    
    # # Parameters
    # size = 256
    # overlap = 128 
    
    # # Initialize
    # if isinstance(data, list):
    #     arr = data[idx]
    # else:
    #     arr = data
    
    # t0 = time.time()
    # print("extract patches : ", end="", flush=True)
    
    # patches = get_patches(arr, size, overlap)
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(np.stack(patches))
    
#%% merge_patches() -----------------------------------------------------------
    
    # # Parameters
    # size = 256
    # overlap = 128 
    
    # # Initialize
    # if isinstance(data, list):
    #     arr = data[idx]
    # else:
    #     arr = data

    # t0 = time.time()
    # print("merge patches : ", end="", flush=True)
    
    # arr_merged = merge_patches(patches, arr.shape, overlap)
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")

    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(np.stack(arr_merged))