#%% Imports -------------------------------------------------------------------

import warnings
import numpy as np
from numba import njit, prange

# Skimage
from skimage.draw import ellipsoid

#%% Numba kernels -------------------------------------------------------------

# Method codes passed to the kernels
_METHODS = {"mean": 0, "median": 1, "std": 2}

@njit(cache=True, nogil=True)
def _median_inplace(buf, n):

    """ Median of buf[:n] using quickselect (buf is reordered in place) """

    k = n // 2
    lo, hi = 0, n - 1
    while lo < hi:
        pivot = buf[(lo + hi) // 2]
        i, j = lo, hi
        while i <= j:
            while buf[i] < pivot: i += 1
            while buf[j] > pivot: j -= 1
            if i <= j:
                tmp = buf[i]; buf[i] = buf[j]; buf[j] = tmp
                i += 1; j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            break
    med = buf[k]
    if n % 2 == 1:
        return med
    # Even count: buf[:k] <= buf[k], so the lower middle is max(buf[:k])
    low = buf[0]
    for i in range(1, k):
        if buf[i] > low:
            low = buf[i]
    return (low + med) / 2

@njit(cache=True, nogil=True)
def _reduce(src, center, offsets, buf, method):

    """ NaN-ignoring mean / median / std of src[center + offsets] """

    n = 0
    for k in range(offsets.size):
        v = src[center + offsets[k]]
        if v == v: # not NaN
            buf[n] = v
            n += 1
    if n == 0:
        return np.nan
    if method == 1:
        return _median_inplace(buf, n)
    s = 0.0
    for i in range(n):
        s += buf[i]
    mean = s / n
    if method == 0:
        return mean
    ss = 0.0
    for i in range(n):
        d = buf[i] - mean
        ss += d * d
    return np.sqrt(ss / n)

def _filt_impl(src, dst, shape, pads, offsets, method):

    """ Filter all non-NaN pixels of src (flat, padded) into dst """

    nZ, nY, nX = shape
    pz, py, px = pads
    PY = nY + 2 * py
    PX = nX + 2 * px
    for r in prange(nZ * nY):
        z = r // nY
        y = r - z * nY
        buf = np.empty(offsets.size)
        base = ((z + pz) * PY + (y + py)) * PX + px
        for x in range(nX):
            c = base + x
            if src[c] == src[c]: # not NaN
                dst[c] = _reduce(src, c, offsets, buf, method)

def _targets_impl(src, targets, offsets, method, vals):

    """ Compute filtered values of src (flat, padded) at target indices """

    chunk = 4096
    n = targets.size
    for ci in prange((n + chunk - 1) // chunk):
        buf = np.empty(offsets.size)
        for i in range(ci * chunk, min(n, (ci + 1) * chunk)):
            vals[i] = _reduce(src, targets[i], offsets, buf, method)

# Parallel and serial versions (compiled once, cached on disk)
_filt_par = njit(cache=True, nogil=True, parallel=True)(_filt_impl)
_filt_ser = njit(cache=True, nogil=True)(_filt_impl)
_targets_par = njit(cache=True, nogil=True, parallel=True)(_targets_impl)
_targets_ser = njit(cache=True, nogil=True)(_targets_impl)

#%% Helpers -------------------------------------------------------------------

def _prepare(arr, mask, kernel_size, kernel_shape, filt_method):

    """ Shared input handling for nan_filt() and nan_replace() """

    if filt_method not in _METHODS:
        raise ValueError(f"filt_method should be one of {list(_METHODS)}")

    # Convert arr to float (copy) and add z dimension (if ndim == 2)
    is2D = arr.ndim == 2
    arr = arr.astype(float)
    if is2D:
        arr = arr[np.newaxis]

    # Mask operations
    if mask is not None:
        mask = mask.astype(bool)
        if mask.ndim == 2:
            mask = mask[np.newaxis]
        if mask.shape[1:] != arr.shape[1:] or \
            mask.shape[0] not in (1, arr.shape[0]):
            raise Exception('mask shape is not compatible with arr shape')
        mask = np.broadcast_to(mask, arr.shape)

    # Extract kernel_size variables
    if isinstance(kernel_size, (int, np.integer)):
        ks = [1 if is2D else kernel_size, kernel_size, kernel_size]
    elif len(kernel_size) == 2:
        ks = [1, kernel_size[0], kernel_size[1]]
    elif len(kernel_size) == 3:
        ks = list(kernel_size)
    else:
        raise ValueError("kernel_size should be an int or a tuple of 2 or 3 int")

    # Round kernel_size variables to next odd integer
    for i, ax in enumerate("zyx"):
        if ks[i] % 2 == 0:
            print(f'{ax} kernel size adjusted from {ks[i]} to {ks[i] + 1}')
            ks[i] += 1
    kz, ky, kx = ks
    pads = (kz // 2, ky // 2, kx // 2)
    pz, py, px = pads

    # Define structuring element
    if kernel_shape == 'cuboid':
        strel = np.ones((kz, ky, kx), dtype=bool)
    elif kernel_shape == 'ellipsoid':
        if kz == 1:
            strel = ellipsoid(1, py, px, spacing=(2, 1, 1))
        else:
            strel = ellipsoid(pz, py, px, spacing=(1, 1, 1))
        strel = strel[1:-1, 1:-1, 1:-1].astype(bool)
    else:
        raise ValueError("kernel_shape should be 'cuboid' or 'ellipsoid'")

    # Pad arr (NaN borders)
    arr_pad = np.pad(arr, ((pz, pz), (py, py), (px, px)),
                     constant_values=np.nan)

    # Flat offsets (in padded arr) of the strel elements, in C order
    PY, PX = arr_pad.shape[1], arr_pad.shape[2]
    dz, dy, dx = np.nonzero(strel)
    offsets = ((dz - pz) * PY + (dy - py)) * PX + (dx - px)
    offsets = offsets.astype(np.int64)

    return arr, mask, arr_pad, pads, offsets, _METHODS[filt_method]

def _unpad(arr_pad, shape, pads):
    pz, py, px = pads
    nZ, nY, nX = shape
    return arr_pad[pz:pz + nZ, py:py + nY, px:px + nX].squeeze()

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

    kernel_shape : str
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

    arr, mask, arr_pad, pads, offsets, method = _prepare(
        arr, mask, kernel_size, kernel_shape, filt_method)

    # Set "out of mask" pixels as NaNs
    if mask is not None:
        pz, py, px = pads
        nZ, nY, nX = arr.shape
        arr_pad[pz:pz + nZ, py:py + nY, px:px + nX][~mask] = np.nan

    # Filter (ping-pong between two buffers, NaN pixels never change)
    kernel = _filt_par if parallel else _filt_ser
    shape = np.array(arr.shape, dtype=np.int64)
    pads_arr = np.array(pads, dtype=np.int64)
    src = arr_pad.ravel()
    dst = src.copy()
    for _ in range(iterations):
        kernel(src, dst, shape, pads_arr, offsets, method)
        src, dst = dst, src

    return _unpad(src.reshape(arr_pad.shape), arr.shape, pads)

#%% Function: nan_replace() ---------------------------------------------------

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

    kernel_shape : str
        Filtering kernel shape, 'cuboid' or 'ellipsoid'.

    filt_method : str
        Filtering method, 'mean', 'median' or 'std'.

    iterations : int or 'inf'
        Iterations of replacing process.
        Use 'inf' to replace all NaNs.

    parallel : bool
        Use or not parallel processing.

    Returns
    -------
    arr_filt : 2D or 3D ndarray
        Processed image.

    """

    arr, mask, arr_pad, pads, offsets, method = _prepare(
        arr, mask, kernel_size, kernel_shape, filt_method)

    # Flat indices (in padded arr) of the NaNs to be replaced
    pz, py, px = pads
    nZ, nY, nX = arr.shape
    to_replace = np.isnan(arr)
    if mask is not None:
        to_replace &= mask
    z, y, x = np.nonzero(to_replace)
    PY, PX = arr_pad.shape[1], arr_pad.shape[2]
    targets = ((z + pz) * PY + (y + py)) * PX + (x + px)
    targets = targets.astype(np.int64)
    del z, y, x, to_replace

    if iterations == 'inf':
        n_iter = np.inf
    elif isinstance(iterations, (int, np.integer)):
        n_iter = iterations
    else:
        raise ValueError("iterations should be an int or 'inf'")

    # Replace
    # At each iteration, all originally-NaN pixels (within mask) are
    # recomputed from the current values, so already-replaced pixels keep
    # being smoothed by their newly replaced neighbors.
    kernel = _targets_par if parallel else _targets_ser
    flat = arr_pad.ravel()
    vals = np.empty(targets.size)
    n_nan = targets.size
    it = 0
    while it < n_iter and targets.size > 0:
        if iterations == 'inf' and n_nan == 0:
            break
        kernel(flat, targets, offsets, method, vals)
        flat[targets] = vals
        n_nan_new = np.count_nonzero(np.isnan(vals))
        if iterations == 'inf' and n_nan_new > 0 and n_nan_new == n_nan:
            warnings.warn(
                f"{n_nan_new} NaN(s) could not be replaced "
                "(not connected to any real value)"
                )
            break
        n_nan = n_nan_new
        it += 1

    return _unpad(arr_pad, arr.shape, pads)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports -----------------------------------------------------------------
    
    import time
    import napari
    from data.create import random_objects

    # random_objects() --------------------------------------------------------
    
    random_objects_params = {
        "shape"        : (32, 512, 512),
        "is3D"         : True,
        "output_type"  : "nan_objects",
        "n_objects"    : 64,
        "radius"       : (16, 32),
        "elongation"   : (0.1, 0.7),
        "irregularity" : 0.2,
        "brightness"   : (1.5, 5.0),
        "noise"        : 0.5,
        "seed"         : 1,
        }
    
    t0 = time.time()
    print("random_objects() : ", end="", flush=True)
    
    # Create random objects array
    arr = random_objects(**random_objects_params)
        
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # nan_filt() --------------------------------------------------------------

    # nan_filt_params = {
    #     "mask"         : None,
    #     "kernel_size"  : 5,
    #     "kernel_shape" : "ellipsoid",
    #     "filt_method"  : "std",
    #     "iterations"   : 3,
    #     "parallel"     : True,
    #     }

    # t0 = time.time()
    # print("nan_filt() : ", end="", flush=True)
    
    # arr_filt = nan_filt(arr, **nan_filt_params)
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(arr_filt)
    
    # nan_replace() -----------------------------------------------------------

    nan_filt_params = {
        "mask"         : None,
        "kernel_size"  : 9,
        "kernel_shape" : "ellipsoid",
        "filt_method"  : "mean",
        "iterations"   : "inf",
        "parallel"     : True,
        }

    t0 = time.time()
    print("nan_replace() : ", end="", flush=True)
    
    arr_filt = nan_replace(arr, **nan_filt_params)
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display
    viewer = napari.Viewer()
    viewer.add_image(arr)
    viewer.add_image(arr_filt)

