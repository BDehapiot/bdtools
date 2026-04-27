#%% Imports -------------------------------------------------------------------

import numpy as np
from joblib import Parallel, delayed 

# bdtools
from bdtools.norm import norm_pct
from bdtools.conn import lbl_conn
from bdtools.check import Check

# skimage
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries

# scipy
from scipy.ndimage import distance_transform_edt, center_of_mass

#%% Function: get_edt() -------------------------------------------------------

def get_edt(
    arr,
    reference="outlines",
    mode="foreground",
    normalize="object",
    invert=False,
    ):
    
    """ 
    Euclidean distance tranform based on scipy.ndimage distance_transform_edt().
    Compute Euclidean distance tranform (edt) for boolean or integer labeled
    mask array. If boolean, edt is applied over the entire array, whereas if 
    labeled, edt is applied individually for each objects.
    
    Parameters
    ----------
    arr : 2D or 3D ndarray (bool or int)
        If boolean, True foreground, False background.
        If labeled, non-zero integers foreground, 0 background.
        
    reference : str, default="outlines"
        Reference from which edt is computed.
        - "outlines" : boundaries of bool/labeled objects
        - "centroids" : centroids of bool/labeled objects
        
    mode : str, default="foreground"
        Portion of the input array to be process
        - "both" : consider the all image.
        - "foreground" : consider only foreground pixels.
        - "background" : consider only background pixels.

    normalize : str, default="none"
        - "none" : no normalization.
        - "global" : 0 to 1 normalization globally over the entire array.
        - "object" : 0 to 1 normalization for each object individually.
        - "object" normalization is only compatible with process "foreground".
                
    invert : bool
        Invert edt values if normalized
        
    Returns
    -------  
    edt : 2D or 3D ndarray (float)
        Euclidean distance tranform of the input array.
        
    """
    
    # Checks ------------------------------------------------------------------
       
    # Parameter : arr
    Check(
        arr, name="arr", ctype=np.ndarray, dtype=(int, bool), ndim=(2, 3))
    if np.all(arr == arr.flat[0]): # Skip if empty 
        return np.zeros_like(arr, dtype="float32") 
    if len(np.unique(arr)) == 2: # Label if binary
        arr = label(arr)
    
    # Parameter : reference, process, normalize
    Check(
        reference, name="reference", ctype=str, 
        vvalue=("outlines", "centroids")
        )
    Check(
        mode, name="mode", ctype=str,
        vvalue=("foreground", "background", "both")
        )
    Check(
        normalize, name="normalize", ctype=str,
        vvalue=("none", "global", "object")
        )
    
    # Parameter : compatibility
    if not mode == "foreground" and normalize == "object":
        raise ValueError(
            "'object' normalization is only compatible with process 'foreground'"
            )
    if invert and normalize == "none":
        raise ValueError(
            "invert requires 'global' or 'object' normalization"
            )
        
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
        if normalize == "object" and mode == "foreground":
            edt = get_centroids_edt_object(arr, ctd)
        else:
            edt = distance_transform_edt(~ctd)
    
    if mode == "foreground": edt[arr == 0] = 0
    if mode == "background": edt[arr != 0] = 0
    
    if normalize == "global":
        edt = norm_pct(edt, pct_low=0, pct_high=100, mask=arr > 0)
        if invert:
            edt = 1 - edt
    
    if normalize == "object" and mode == "foreground":
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
    Skeletonize based on scipy.ndimage skeletonize().
    Compute skeleton for boolean or integer labelled mask array. If boolean, 
    skeletonize() is applied over the entire array, whereas if labelled, 
    skeletonize() is applied individually for each objects.
    
    Parameters
    ----------
    arr : 2D or 3D ndarray (bool or int)
        Boolean : True foreground, False background.
        Labelled : non-zero integers objects, 0 background.
    
    parallel : bool
        Compute skeletonize() in parallel if True.
                
    Returns
    -------  
    skel : 2D or 3D ndarray (bool)
        Skeleton of the input array.
        
    """
    
    # Checks ------------------------------------------------------------------
    
    # Parameter : arr
    Check(
        arr, name="arr", ctype=np.ndarray, dtype=(int, bool), ndim=(2, 3))
    if np.all(arr == arr.flat[0]): # Skip if empty 
        return np.zeros_like(arr, dtype=bool) 
    
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

#%% Function: process_masks() -------------------------------------------------

def process_masks(
        data, method="binary", 
        edt_parameters={
            "reference" : "outlines",
            "mode"      : "foreground",
            "normalize" : "object",   
            }
        ):
    
    """
    Process masks using the selected method.
    
    Parameters
    ----------
    data : single or list of 2D or 3D ndarrays (bool or int)
        Boolean : True foreground, False background.
        Labelled : non-zero integers objects, 0 background.
        
    method : str, default="binary"
        Select mask type output
        - "binary" : simple thresholding (arr > 0).
        - "edt" : object Euclidean Distance Transform, see get_edt().
        - "outlines" : object outlines.
        - "interfaces" : object interfaces (contact between objects).
        - "centroids" : object centroids.
        - "skeletons" : object skeletons, see get_skeletons().
        
    edt_parameters : dict
        See get_edt()
    
    Returns
    -------      
    prc : unique or list of 2D or 3D ndarray (bool, int or float)
        Processed masks output in the format of input data.
    
    """
        
    # Checks ------------------------------------------------------------------
    
    # Parameter : data
    Check(
        data, name="data", ctype=(np.ndarray, list), dtype=(int, bool), ndim=(2, 3))
        
    # Parameter : mask_type
    Check(
        method, name="method", ctype=str,
        vvalue=("binary", "edt", "outlines", "interfaces", "centroids", "skeletons")
        )
    
    # Nested function(s) ------------------------------------------------------
    
    def get_centroids(arr):
        if len(np.unique(arr)) <= 2:
            arr = label(arr, connectivity=1)
        coords = center_of_mass(
            np.ones_like(arr), arr, range(1, arr.max() + 1))
        coords = np.round(coords).astype(int)
        centroids = np.zeros_like(arr, dtype=bool)
        for y, x in coords:
            centroids[y, x] = True
        return centroids
    
    def _prepare_mask(arr):
        
        if method == "binary":
            prc = arr > 0
        elif method == "edt":
            prc = get_edt(arr, **edt_parameters)
        elif method == "outlines":
            prc = find_boundaries(arr)
        elif method == "interfaces":
            prc = lbl_conn(arr, conn=2) > 1
        elif method == "centroids":
            prc = get_centroids(arr)
        elif method == "skeletons":
            prc = get_skeletons(arr)
 
        return prc
    
    # Execute -----------------------------------------------------------------
    
    if isinstance(data, list):
        
        prc = Parallel(n_jobs=-1)(
            delayed(_prepare_mask)(arr)
            for arr in data
            )           
        
    elif isinstance(data, np.ndarray):
    
        if data.ndim == 3:
            
            prc = Parallel(n_jobs=-1)(
                delayed(_prepare_mask)(arr)
                for arr in data
                )           
            prc = np.stack(prc)
            
        elif data.ndim == 2:
            
            prc = _prepare_mask(data)
        
    return prc

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
    # dataset = "fluo_tissue"
    dataset = "fluo_nuclei_instance"
    # dataset = "fluo_nuclei_semantic"
    data_path = Path.cwd().parent / "_local" / dataset
    paths = list(data_path.rglob("*msk_trn.tif"))
    
    # Open data
    data = []
    for path in paths:
        data.append(io.imread(path))
    if len(data) == 1:
        data = data[0]
    if "wdisk" in dataset:
        data = ~data
    if "nuclei_semantic" in dataset:
        data_1 = label(data == 1)
        data_2 = label(data == 2)
        data_3 = label(data == 3) 
        data_2[data_2 > 0] += np.max(data_1)
        data_3[data_3 > 0] += np.max(data_2)
        data = data_1 + data_2 + data_3
    if "nuclei" in dataset:
        data = list(data)
    if isinstance(idx, int):
        data = data[idx]
    
#%% get_edt() -----------------------------------------------------------------
    
    # Parameters
    reference = "outlines"
    mode = "foreground" 
    normalize = "object"   
    invert = False
    
    # Initialize
    if isinstance(data, list):
        arr = data[0]
    else:
        arr = data
    
    t0 = time.time()
    print("get_edt() : ", end="", flush=True)
    
    edt = get_edt(
        arr, 
        reference=reference,
        mode=mode,
        normalize=normalize,
        invert=invert,
        )
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display
    vwr = napari.Viewer()
    vwr.add_labels(arr, visible=1, opacity=0.5)
    vwr.add_image(edt, blending="additive", visible=1) 
    
#%% get_skeletons() -----------------------------------------------------------
        
    # Initialize
    if isinstance(data, list):
        arr = data[0]
    else:
        arr = data
    
    t0 = time.time()
    print("get_skeletons() : ", end="", flush=True)
    
    skl = get_skeletons(arr, parallel=True)
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display
    vwr = napari.Viewer()
    vwr.add_labels(arr, visible=1, opacity=0.5)
    vwr.add_image(skl, blending="additive", visible=1) 
    
#%% process_masks() -----------------------------------------------------------
        
    # Parameters
    # method = "binary"
    method = "edt"
    # method = "outlines"
    # method = "interfaces"
    # method = "centroids"
    # method = "skeletons"
    
    # Initialize
    arr = data
    
    t0 = time.time()
    print("process_masks() : ", end="", flush=True)
    
    prc = process_masks(
        data, method=method,
        edt_parameters={
            "reference" : "outlines",
            "mode"      : "foreground",
            "normalize" : "object",   
            }
        )
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display
    vwr = napari.Viewer()
    if isinstance(arr, list):
        idx = 1
        vwr.add_labels(arr[idx], visible=1, opacity=0.5)
        vwr.add_image(prc[idx], blending="additive", visible=1)
    else:
        vwr.add_labels(arr, visible=1, opacity=0.5)
        vwr.add_image(prc, blending="additive", visible=1)   
