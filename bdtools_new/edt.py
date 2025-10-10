#%% Imports -------------------------------------------------------------------

import numpy as np
from joblib import Parallel, delayed 

# bdtools
from bdtools.norm import norm_pct

# skimage
from skimage.filters import gaussian
from skimage.transform import rescale, resize
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation
from skimage.segmentation import find_boundaries

# scipy
from scipy.ndimage import distance_transform_edt

#%% Function: get_edt ---------------------------------------------------------

def get_edt(
        arr, 
        target="foreground", 
        sampling=1, 
        normalize="none", 
        rescale_factor=1, 
        parallel=True
        ):
        
    """ 
    Euclidean distance tranform.
    Based on scipy.ndimage distance_transform_edt().

    Compute Euclidean distance tranform (edt) for boolean or integer labelled
    mask array. If boolean, edt is applied over the entire array, whereas if 
    labelled, edt is applied individually for each objects.
    
    Parameters
    ----------
    arr : 2D or 3D ndarray (bool or uint8, uint16, int32)
        Boolean : True foreground, False background.
        Labelled : non-zero integers objects, 0 background.
        
    target : str, optional, default="foreground"
        - "foreground" : foreground distance to closest background pixel.
        - "background" : background distance to closest foreground pixel.
        
    sampling : float or tuple of floats, optional, default=1
        From Scipy "Spacing of elements along each dimension. If a sequence, 
        must be of length equal to the input rank; if a single number, this is
        used for all axes".
    
    normalize : str, optional, default="none"
        - "none" : no normalization.
        - "global" : 0 to 1 normalization globally over the entire array.
        - "object" : 0 to 1 normalization for each object individually.
    
    rescale_factor : float, optional, default=1.
        Rescaling factor applied to input array before processing. Rescaling
        introduces precision loss, especially near object boundaries or for 
        small objects.
    
    parallel : bool
        Compute edt in parallel if True and target is "foreground".
                
    Returns
    -------  
    edt : 2D or 3D ndarray (float)
        Euclidean distance tranform of the input array.
        
    """
    
    if not (np.issubdtype(arr.dtype, np.integer) or
            np.issubdtype(arr.dtype, np.bool_)):
        raise TypeError("Provided array must be bool or integers labels")
        
    if np.all(arr == arr.flat[0]):
        return np.zeros_like(arr, dtype="bool")
    
    valid_targets = ["foreground", "background"]
    if target not in valid_targets:
        raise ValueError(
            f"Invalid value for target: '{target}'."
            f" Expected one of {valid_targets}."
            )
    
    valid_normalize = ["none", "global", "object"]
    if normalize not in valid_normalize:
        raise ValueError(
            f"Invalid value for normalize: '{normalize}'."
            f" Expected one of {valid_normalize}."
            )
    
    # Nested function(s) ------------------------------------------------------
    
    def _get_edt(lab, normalize=normalize):
        edt = arr == lab
        edt = binary_dilation(edt)
        edt = distance_transform_edt(edt, sampling=sampling) * 1 / rescale_factor
        if normalize == "object":
            edt = norm_pct(edt, pct_low=0, pct_high=99.9, mask=edt > 0)
        return edt
        
    # Execute -----------------------------------------------------------------
        
    arr = arr.copy()
    
    if rescale_factor < 1:
        arr_copy = arr.copy()
        arr = rescale(arr, rescale_factor, order=0)
            
    if target == "foreground":
        arr[find_boundaries(arr, mode="inner") == 1] = 0
        arr = label(arr > 0)
        labels = np.unique(arr)[1:]
        if parallel:
            edt = Parallel(n_jobs=-1)(
                delayed(_get_edt)(lab) for lab in labels)
        else:
            edt = [_get_edt(lab) for lab in labels]        
        edt = np.nanmax(np.stack(edt), axis=0).astype("float32")
    else:
        edt = distance_transform_edt(np.invert(arr > 0), sampling=sampling)
        
    if normalize == "global":
        edt = norm_pct(edt, pct_low=0, pct_high=99.9, mask=edt > 0)
        
    if rescale_factor < 1:
        edt = resize(edt, arr_copy.shape, order=1)
        edt = gaussian(edt, sigma=1 / rescale_factor / 2)
        if target == "foreground":
            edt[arr_copy == 0] = 0
        else:
            edt[arr_copy != 0] = 0
        
    return edt

#%% Execute (test) ------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports
    import time
    import napari
    from edt_test import generate_random_array
    
    # Inputs
    nZ, nY, nX, = 1, 1024, 1024
    nObj = 32
    min_radius = 8
    max_radius = 32
    
    # Generate random arrays
    arr = generate_random_array(
        nZ, nY, nX, 
        nObj=nObj, 
        min_radius=min_radius, 
        max_radius=max_radius,
        )
        
    # get_edt() ---------------------------------------------------------------
    
    # Inputs 
    reference = "centroids" # "outlines" or "centroids"
    process = "foreground"  # "foreground", "background" or "both"
    normalize = "object"    # "none", "global" or "object"
    rescale_factor = 0.5
    sampling = 1 
    
    if not (np.issubdtype(arr.dtype, np.integer) or
            np.issubdtype(arr.dtype, np.bool_)):
        raise TypeError("Provided array must be bool or integers labels")
        
    if np.all(arr == arr.flat[0]):
        # return np.zeros_like(arr, dtype="bool")
        edt = np.zeros_like(arr, dtype="bool")
    
    valid_reference = ["outlines", "centroids"]
    if reference not in valid_reference:
        raise ValueError(
            f"Invalid value for reference: '{reference}'."
            f" Expected one of {valid_reference}."
            )
    
    valid_process = ["foreground", "background", "both"]
    if process not in valid_process:
        raise ValueError(
            f"Invalid value for process: '{process}'."
            f" Expected one of {valid_process}."
            )
    
    valid_normalize = ["none", "global", "object"]
    if normalize not in valid_normalize:
        raise ValueError(
            f"Invalid value for normalize: '{normalize}'."
            f" Expected one of {valid_normalize}."
            )
    
    # ---
    
    def get_centroids_coords(arr):
        
        # Get flat indices and array
        indices = np.indices(arr.shape)
        indices_flat = [idx.ravel() for idx in indices]
        arr_flat = arr.ravel()
    
        # Get unique labels
        labels, inverse = np.unique(arr_flat, return_inverse=True)
    
        # Get centroids coords
        coords = []
        for lbl_idx in range(len(labels)):
            mask = inverse == lbl_idx
            if not np.any(mask):
                coords.append(tuple([np.nan] * arr.ndim))
                continue
            means = [np.mean(idx[mask]) for idx in indices_flat]
            int_coords = tuple(int(round(m)) for m in means)
            coords.append(int_coords)
    
        return coords
    
    def get_centroids_array(arr):
        ctd = np.zeros_like(arr, dtype=bool)
        coords = get_centroids_coords(arr)
        for coord in coords:
            ctd[coord] = 1
        return ctd
            
    # ---
    
    t0 = time.time()
    print("rescale #1 : ", end="", flush=True)
    
    msk = arr > 0
    
    if rescale_factor < 1:
        if arr.ndim == 3:
            arr = rescale(arr, (1, rescale_factor, rescale_factor), order=0)
        else: 
            arr = rescale(arr, rescale_factor, order=0)            
        
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # ---
    
    t0 = time.time()
    print("edt : ", end="", flush=True)
    
    if reference == "outlines": 
        out = find_boundaries(arr, mode="thick")
        edt = distance_transform_edt(~out)
    if reference == "centroids":
        ctd = get_centroids_array(arr)
        edt = distance_transform_edt(~ctd)
    # if process == "foreground":
    #     edt[arr == 0] = 0
    # if process == "background":
    #     edt[arr != 0] = 0           
            
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
        
    # ---
    
    t0 = time.time()
    print("normalize : ", end="", flush=True)
    
    if normalize == "global":
        edt = norm_pct(edt, pct_low=0, pct_high=100)
    if normalize == "object" and process == "foreground":
        for props in regionprops(arr):
            coords = tuple(props.coords.T)
            edt[coords] /= np.max(edt[coords])
                
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")

    # ---

    t0 = time.time()
    print("rescale #2 : ", end="", flush=True)

    # if rescale_factor < 1:
    #     edt = resize(edt, msk.shape, order=1)
    #     # edt = gaussian(edt, sigma=1 / rescale_factor / 2)
    #     if process == "foreground":
    #         edt[~msk] = 0
    #     elif process == "background":
    #         edt[msk] = 0
            
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")

    # -------------------------------------------------------------------------
    
    # Display
    vwr = napari.Viewer()
    vwr.add_labels(arr, opacity=0.5, visible=0)
    # vwr.add_image(out, blending="additive", visible=1)
    vwr.add_image(ctd, blending="additive", visible=1)
    vwr.add_image(edt, blending="additive", visible=1)