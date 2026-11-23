#%% Imports -------------------------------------------------------------------

import numba
import numpy as np
from scipy.ndimage import distance_transform_edt

#%% Function(s) shared --------------------------------------------------------

def get_dims(shape, multichannel=False):

    """ Return (nS, nY, nX, nC) from 2D/3D/4D shape """

    if len(shape) == 2:
        return 1, shape[0], shape[1], 1
    if len(shape) == 3:
        if multichannel:
            return 1, shape[0], shape[1], shape[2]
        return shape[0], shape[1], shape[2], 1
    if len(shape) == 4:
        return tuple(shape)
    raise ValueError(f"Unsupported number of dimensions ({len(shape)})")

def get_grid(n, size, overlap):

    """
    Return patch origins along one axis + padding (before, after).

    Use the minimum number of patches required to cover n pixels
    (np.arange(0, n, step) can add a redundant last patch).

    """

    step = size - overlap
    nP = max(1, int(np.ceil((n - size) / step)) + 1)
    p0s = np.arange(nP) * step
    pad = p0s[-1] + size - n
    return p0s, pad // 2, (pad + 1) // 2

def get_weights(size):

    """ Return (size, size) weights (EDT to patch border + 1) """

    edt = np.ones((size, size), dtype=np.float32)
    edt[[0, -1], :] = 0
    edt[:, [0, -1]] = 0
    return (distance_transform_edt(edt) + 1).astype(np.float32)

#%% Function : extract_patches() ----------------------------------------------

def extract_patches(arr, size, overlap, multichannel=False):

    """
    Extract patches from single/multichannel 2D/3D ndarray.

    For 3D array, patches are extracted from individual 2D slices defined along
    the first dimension. For multichannel array, channels must be defined along
    the last dimension. If necessary, the input array is padded using 'reflect'
    padding mode.

    Parameters
    ----------
    arr : 2D, 3D or 4D ndarray
        Array to be patched.

    size : int
        Size of extracted patches.

    overlap : int
        Overlap between patches (Must be between 0 and size - 1).

    multichannel : bool
        Define single or multichannel input array (4D is always multichannel).

    Returns
    -------
    patches : list of ndarrays
        List containing extracted patches (views of the padded array).

    """

    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size - 1], got {overlap}")

    # Get dimensions
    multichannel = multichannel or arr.ndim == 4
    nS, nY, nX, nC = get_dims(arr.shape, multichannel)
    y0s, yPad1, yPad2 = get_grid(nY, size, overlap)
    x0s, xPad1, xPad2 = get_grid(nX, size, overlap)

    # Pad array (as 4D)
    arr = arr.reshape(nS, nY, nX, nC)
    arr_pad = np.pad(
        arr, ((0, 0), (yPad1, yPad2), (xPad1, xPad2), (0, 0)), mode="reflect")
    if not multichannel:
        arr_pad = arr_pad[..., 0]

    # Extract patches
    patches = [
        arr_pad[t, y0:y0 + size, x0:x0 + size]
        for t in range(nS) for y0 in y0s for x0 in x0s
        ]

    return patches

#%% Function : merge_patches() ------------------------------------------------

@numba.njit(parallel=True, cache=True)
def merge_patches_numba(
        patches, weights, weights_sum, y0s, x0s, yPad1, xPad1, nY, nX):

    # patches : (nS, nPy, nPx, size, size, nC)
    # Each (slice, output row) is computed independently by gathering
    # contributions from all patches covering it (no race condition, so
    # parallel even for single 2D images). Crop & normalization are done
    # on the fly, only the final array is allocated.

    nS, nPy, nPx, size, _, nC = patches.shape
    merged = np.zeros((nS, nY, nX, nC), dtype=np.float32)

    for idx in numba.prange(nS * nY):
        t, y = idx // nY, idx % nY
        r = y + yPad1 # row in padded frame
        for i in range(nPy):
            di = r - y0s[i]
            if di < 0 or di >= size:
                continue
            for j in range(nPx):
                c0 = x0s[j] - xPad1 # patch origin in output frame
                dj0 = max(0, -c0)
                dj1 = min(size, nX - c0)
                for dj in range(dj0, dj1):
                    w = weights[di, dj]
                    for k in range(nC):
                        merged[t, y, c0 + dj, k] += patches[t, i, j, di, dj, k] * w
        for x in range(nX):
            w = weights_sum[r, x + xPad1]
            for k in range(nC):
                merged[t, y, x, k] /= w

    return merged

def merge_patches(patches, shape, overlap, multichannel=False):

    """
    Merge patches (from extract_patches()) back into original shape.

    Overlapping regions are blended using a weighted average, weights being
    the distance to the patch border (center pixels prevail).

    Parameters
    ----------
    patches : list of ndarrays
        List of patches, as returned by extract_patches().

    shape : tuple
        Shape of the original array.

    overlap : int
        Overlap between patches (same as in extract_patches()).

    multichannel : bool
        Define single or multichannel original array.

    Returns
    -------
    merged : ndarray
        Merged array (same shape as original, same dtype as patches).

    """

    # Get dimensions
    size = patches[0].shape[0]
    multichannel = multichannel or len(shape) == 4
    nS, nY, nX, nC = get_dims(shape, multichannel)
    y0s, yPad1, yPad2 = get_grid(nY, size, overlap)
    x0s, xPad1, xPad2 = get_grid(nX, size, overlap)

    # Prepare patches
    patches_arr = np.stack(patches).astype(np.float32, copy=False)
    patches_arr = patches_arr.reshape(
        nS, len(y0s), len(x0s), size, size, nC)

    # Get weights (shared by all slices & channels)
    weights = get_weights(size)
    weights_sum = np.zeros(
        (nY + yPad1 + yPad2, nX + xPad1 + xPad2), dtype=np.float32)
    for y0 in y0s:
        for x0 in x0s:
            weights_sum[y0:y0 + size, x0:x0 + size] += weights

    # Merge patches
    merged = merge_patches_numba(
        patches_arr, weights, weights_sum, y0s, x0s, yPad1, xPad1, nY, nX)

    # Final formatting
    merged = merged.reshape(shape)

    return merged.astype(patches[0].dtype, copy=False)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports -----------------------------------------------------------------
    
    import time
    import napari
    from _test import load_data
        
    # Initialize --------------------------------------------------------------
    
    # Paths
    # dataset = "em_mito"
    dataset = "fluo_tissue"
    # dataset = "fluo_nuclei_instance"
    # dataset = "fluo_nuclei_semantic"
    # dataset = "sat_roads"
    X, y = load_data(dataset=dataset, display=False)
           
    # Parameters
    idx = 0
    size = 128
    overlap = 64
    
    # extract_patches() -------------------------------------------------------

    # Initialize
    X = X[idx] if isinstance(X, list) else X

    t0 = time.time()
    print("extract_patches() : ", end="", flush=True)
    
    patches = extract_patches(X, size, overlap)
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display
    viewer = napari.Viewer()
    viewer.add_image(np.stack(patches))
    
    # merge_patches() ---------------------------------------------------------
        
    # Initialize
    X = X[idx] if isinstance(X, list) else X
    
    t0 = time.time()
    print("merge_patches() : ", end="", flush=True)
    
    merged_patches = merge_patches(patches, X.shape, overlap)
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display
    viewer = napari.Viewer()
    viewer.add_image(np.stack(merged_patches))