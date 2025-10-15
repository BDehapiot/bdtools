#%% Imports -------------------------------------------------------------------

import warnings
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

    Compute Euclidean distance tranform (edt) for boolean or integer labeled
    mask array. If boolean, edt is applied over the entire array, whereas if 
    labeled, edt is applied individually for each objects.
    
    Parameters
    ----------
    arr : 2D or 3D ndarray (bool or uint8, uint16, int32)
        Boolean : True foreground, False background.
        labeled : non-zero integers objects, 0 background.
        
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

#%% Function: get_edt() -------------------------------------------------------

def get_edt(
    arr,
    reference="outlines",
    process="both",
    normalize="object",
    invert=False,
    sampling=1,
    ):
    
    """ 
    Euclidean distance tranform.
    Based on scipy.ndimage distance_transform_edt().
    Compute Euclidean distance tranform (edt) for boolean or integer labeled
    mask array. If boolean, edt is applied over the entire array, whereas if 
    labeled, edt is applied individually for each objects.
    
    Parameters
    ----------
    arr : 2D or 3D ndarray (bool or uint8, uint16, int32)
        if boolean, True foreground, False background.
        if labeled, non-zero integers foreground, 0 background.
        
    reference : str, optional, default="outlines"
        reference from which edt is computed.
        - "outlines" : boundaries of bool/labeled objects
        - "centroids" : centroids of bool/labeled objects
        
    process : str, optional, default="foreground"
        portion of the input array to be process
        - "both" : consider the all image.
        - "foreground" : consider only foreground pixels.
        - "background" : consider only background pixels.

    normalize : str, optional, default="none"
        - "none" : no normalization.
        - "global" : 0 to 1 normalization globally over the entire array.
        - "object" : 0 to 1 normalization for each object individually.
        - "object" normalization is only compatible with process "foreground".
                
    Returns
    -------  
    edt : 2D or 3D ndarray (float)
        Euclidean distance tranform of the input array.
        
    """
    
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
        
    if not (np.issubdtype(arr.dtype, np.integer) or
            np.issubdtype(arr.dtype, np.bool_)):
        raise TypeError("Provided array must be bool or integers labels")
        
    if not process == "foreground" and normalize == "object":
        warnings.warn(
            "'object' normalization is only compatible with process 'foreground'"
            )
        
    if np.all(arr == arr.flat[0]):
        # return np.zeros_like(arr, dtype="bool")
        edt = np.zeros_like(arr, dtype="bool")    
        
    if len(np.unique(arr)) == 2:
        arr = label(arr)
    
    # Nested function(s) ------------------------------------------------------
    
    def get_centroids_coords(arr):
        
        # Get flat indices and array
        indices = np.indices(arr.shape)
        indices_flat = [idx.ravel() for idx in indices]
        arr_flat = arr.ravel()
    
        # Get unique labels
        labels, inverse = np.unique(arr_flat, return_inverse=True)
    
        # Get centroids coords
        coords = []
        for lbl_idx, lbl in enumerate(labels):
            if lbl == 0:
                continue
            mask = inverse == lbl_idx
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
    
    def get_centroids_edt_object(arr, ctd):

        edt = np.zeros_like(arr, dtype=np.float32)
    
        for props in regionprops(arr):
            if props.label == 0:
                continue

            # Get bbox data
            if arr.ndim == 3:
                min_z, min_y, min_x, max_z, max_y, max_x = props.bbox
                bbox_slc = np.s_[min_z:max_z, min_y:max_y, min_x:max_x]
            else:
                min_y, min_x, max_y, max_x = props.bbox
                bbox_slc = np.s_[min_y:max_y, min_x:max_x]
            bbox_msk = (arr[bbox_slc] == props.label)
            bbox_ctd = np.logical_and(ctd[bbox_slc], bbox_msk)
            bbox_edt = distance_transform_edt(~bbox_ctd)
    
            # Append edt
            edt[bbox_slc][bbox_msk] = bbox_edt[bbox_msk]
    
        return edt
    
    # Execute -----------------------------------------------------------------
    
    if reference == "outlines": 
        out = find_boundaries(arr, mode="thick")
        edt = distance_transform_edt(~out)
    if reference == "centroids":
        ctd = get_centroids_array(arr)  
        if normalize == "object" and process == "foreground":
            edt = get_centroids_edt_object(arr, ctd)
        else:
            edt = distance_transform_edt(~ctd)
            
    if process == "foreground": edt[arr == 0] = 0
    if process == "background": edt[arr != 0] = 0
        
    if normalize == "global":
        edt = norm_pct(edt, pct_low=0, pct_high=100, mask=arr > 0)
        if invert:
            edt = 1 - edt
    if normalize == "object" and process == "foreground":
        for props in regionprops(arr):
            coords = tuple(props.coords.T)
            edt[coords] /= np.max(edt[coords])
        if invert:
            edt = 1 - edt
            edt[arr == 0] = 0
                
    return edt

#%% Execute (test) ------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports
    import time
    import napari
    
    # Generate random arrays() ------------------------------------------------
    
    from edt_test import generate_random_array
    
    # Inputs
    nZ, nY, nX, = 1, 1024, 1024
    nObj = 32
    min_radius = 8
    max_radius = 32
    
    ismask = True
    
    t0 = time.time()
    print("generate_random_array() : ", end="", flush=True)

    arr = generate_random_array(
        nZ, nY, nX, 
        nObj=nObj, 
        min_radius=min_radius, 
        max_radius=max_radius,
        )
        
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    if ismask:
        arr = arr > 0
    
    # get_edt() ---------------------------------------------------------------
    
    # Inputs 
    reference = "outlines" # "outlines" or "centroids"
    process = "background" # "foreground", "background" or "both"
    normalize = "global"   # "none", "global" or "object"
    invert = False
    sampling = 1 
    
    t0 = time.time()
    print("get_edt() : ", end="", flush=True)
    
    edt = get_edt(
        arr, 
        reference=reference,
        process=process,
        normalize=normalize,
        invert=invert,
        sampling=sampling,
        )
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # -------------------------------------------------------------------------
    
    # Display
    vwr = napari.Viewer()
    vwr.add_labels(arr, opacity=0.5, visible=0)
    vwr.add_image(edt, blending="additive", colormap="turbo", visible=1)