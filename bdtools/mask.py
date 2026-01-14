#%% Imports -------------------------------------------------------------------

import warnings
import numpy as np
from joblib import Parallel, delayed 

# bdtools
from bdtools.norm import norm_pct
from bdtools.conn import lbl_conn

# skimage
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries

# scipy
from scipy.ndimage import distance_transform_edt, center_of_mass

#%% Comments ------------------------------------------------------------------

'''
- get_edt() and get_skeletons works on 2D ndarray
- prepare_mask() works either on 2D ndarrays or list of 2D ndarrays.
  3D ndarrays are converted in list of 2D ndarrays over the 1st dimension.
  It's necessary that prepare_mask() works in differently sized images.
  Also, prepare_mask() outputs a list if a list was inputed.
'''

#%% Function: get_edt() -------------------------------------------------------

def get_edt(
    arr,
    reference="outlines",
    process="foreground",
    normalize="object",
    invert=False,
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
                
    invert : bool
        invert edt values if normalized
        
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
        
    if invert and normalize == "none":
        warnings.warn(
            "invert requires 'global' or 'object' normalization"
            )
        
    if np.all(arr == arr.flat[0]):
        return np.zeros_like(arr, dtype="bool") 
        
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
        out = find_boundaries(arr, mode="inner")
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
        for props in regionprops(label(arr, connectivity=1)):
            coords = tuple(props.coords.T)
            edt[coords] /= np.maximum(1, np.max(edt[coords]))
        if invert:
            edt = 1 - edt
            edt[arr == 0] = 0
    
    return edt.astype("float32")

#%% Function: get_skeletons() -------------------------------------------------

def get_skeletons(arr, parallel=True):
    
    """ 
    Skeletonize.
    Based on scipy.ndimage skeletonize().

    Compute skeleton for boolean or integer labelled mask array. If boolean, 
    skeletonize() is applied over the entire array, whereas if labelled, 
    skeletonize() is applied individually for each objects.
    
    Parameters
    ----------
    arr : 2D ndarray (bool or uint8, uint16, int32)
        Boolean : True foreground, False background.
        Labelled : non-zero integers objects, 0 background.
    
    parallel : bool
        Compute skeletonize() in parallel if True.
                
    Returns
    -------  
    skel : 2D ndarray (bool)
        skeleton of the input array.
        
    """
    
    # Checks
    if not (np.issubdtype(arr.dtype, np.integer) or
            np.issubdtype(arr.dtype, np.bool_)):
        raise TypeError("Input array must be bool or integers labels")
    if np.all(arr == arr.flat[0]):
        return np.zeros_like(arr, dtype="bool")
    
    # Nested function(s) ------------------------------------------------------
    
    def _get_skel(lbl):
        skl = np.zeros_like(arr)
        skl[arr == lbl] = True
        skl = skeletonize(skl)
        return skl
    
    # Execute -----------------------------------------------------------------
    
    arr = arr.copy()
    arr[find_boundaries(arr, mode="inner") == 1] = 0
    arr = label(arr > 0)
    lbls = np.unique(arr)[1:]
    if parallel:
        skl = Parallel(n_jobs=-1)(
            delayed(_get_skel)(lbl) for lbl in lbls)
    else:
        skl = [_get_skel(lbl) for lbl in lbls]   

    return np.max(np.stack(skl), axis=0)

#%% Function: prepare_mask() -----------------------------------------------------

def prepare_mask(arr, mask_type="binary"):
    
    # Nested function(s) ------------------------------------------------------
    
    def get_centroids(img):
        if len(np.unique(img)) <= 2:
            img = label(img, connectivity=1)
        coords = center_of_mass(
            np.ones_like(img), img, range(1, img.max() + 1))
        coords = np.round(coords).astype(int)
        centroids = np.zeros_like(img, dtype=bool)
        for y, x in coords:
            centroids[y, x] = True
        return centroids
    
    def _prepare_mask(img):
        
        if mask_type == "binary":
            prp = img > 0
        elif mask_type == "edt":
            prp = get_edt(
                img, 
                reference="outlines",
                process="foreground",
                normalize="object",                
                )
        elif mask_type == "outlines":
            prp = find_boundaries(img)
        elif mask_type == "interfaces":
            prp = lbl_conn(img, conn=2) > 1
        elif mask_type == "centroids":
            prp = get_centroids(img)
        elif mask_type == "skeletons":
            prp = get_skeletons(img)
 
        return prp
    
    # Execute -----------------------------------------------------------------
    
    if arr.ndim == 3:
        
        prp = Parallel(n_jobs=-1)(
            delayed(_prepare_mask)(img)
            for img in arr
            )           
        prp = np.stack(prp)
        
    elif arr.ndim == 2:
        
        prp = _prepare_mask(arr)
        
    return prp

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports
    import time
    import napari
    from skimage import io
    from pathlib import Path
    
    # Paths
    idx = "all"
    # dataset = "em_mito"
    # dataset = "skel_wdisk"
    dataset = "fluo_tissue"
    # dataset = "fluo_nuclei_instance"
    # dataset = "fluo_nuclei_semantic"
    data_path = Path.cwd().parent / "_local" / dataset
    paths = list(data_path.rglob("*msk_trn.tif"))
    
    # Open data
    data = []
    for path in paths:
        data.append(io.imread(path))
    if len(data) == 1:
        data = data[0]
    if "nuclei" in dataset:
        data = list(data)
    if isinstance(idx, int):
        data = data[idx]
    if "nuclei_semantic" in dataset:
        data_1 = label(data == 1)
        data_2 = label(data == 2)
        data_3 = label(data == 3) 
        data_2[data_2 > 0] += np.max(data_1)
        data_3[data_3 > 0] += np.max(data_2)
        data = data_1 + data_2 + data_3
    
    # get_edt() ---------------------------------------------------------------
    
    # Parameters
    reference = "outlines"
    process = "foreground" 
    normalize = "object"   
    invert = False
    
    if isinstance(data, list):
        img = data[0]
    
    t0 = time.time()
    print("get_edt() : ", end="", flush=True)
    
    edt = get_edt(
        img, 
        reference=reference,
        process=process,
        normalize=normalize,
        invert=invert,
        )
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # get_skeletons() ---------------------------------------------------------
        
    # t0 = time.time()
    # print("get_skeletons() : ", end="", flush=True)
    
    # skl = get_skeletons(arr, parallel=False)
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
    # prep_mask() -------------------------------------------------------------
        
    # Parameters
    # mask_type = "binary"
    # mask_type = "edt"
    # mask_type = "outlines"
    # mask_type = "interfaces"
    # mask_type = "centroids"
    # mask_type = "skeletons"
    
    # t0 = time.time()
    # print("prepare_mask() : ", end="", flush=True)
    
    # prp = prepare_mask(arr, mask_type=mask_type)
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")

    # -------------------------------------------------------------------------

    # Display
    # vwr = napari.Viewer()
    # vwr.add_labels(arr, visible=1, opacity=0.5)
    # vwr.add_image(prp, blending="additive", visible=1)
    # vwr.add_image(edt, blending="additive", visible=0)
    # vwr.add_image(skl, blending="additive", visible=0)
        
