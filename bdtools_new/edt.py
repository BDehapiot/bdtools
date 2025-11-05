#%% Imports -------------------------------------------------------------------

import warnings
import numpy as np 

# bdtools
from bdtools.norm import norm_pct

# skimage
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries

# scipy
from scipy.ndimage import distance_transform_edt

#%% Comments ------------------------------------------------------------------

'''
- sampling factor to modify edt values acc. to pix/voxel size
- rescaling factor to speed up processing
- still very slow on 3D arrays
'''

#%% Function: get_edt() -------------------------------------------------------

def get_edt(
    arr,
    reference="outlines",
    process="both",
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

# if __name__ == "__main__":
    
#     # Imports
#     import time
#     import napari
    
#     # Generate random arrays() ------------------------------------------------
    
#     from edt_test import generate_random_array
    
#     # Inputs
#     nZ, nY, nX, = 1, 1024, 1024
#     nObj = 32
#     min_radius = 8
#     max_radius = 32
    
#     ismask = True
    
#     t0 = time.time()
#     print("generate_random_array() : ", end="", flush=True)

#     arr = generate_random_array(
#         nZ, nY, nX, 
#         nObj=nObj, 
#         min_radius=min_radius, 
#         max_radius=max_radius,
#         )
        
#     t1 = time.time()
#     print(f"{t1 - t0:.3f}s")
    
#     if ismask:
#         arr = arr > 0
    
#     # get_edt() ---------------------------------------------------------------
    
#     # Inputs 
#     reference = "outlines"
#     process = "background" 
#     normalize = "global"   
#     invert = False
    
#     t0 = time.time()
#     print("get_edt() : ", end="", flush=True)
    
#     edt = get_edt(
#         arr, 
#         reference=reference,
#         process=process,
#         normalize=normalize,
#         invert=invert,
#         )
    
#     t1 = time.time()
#     print(f"{t1 - t0:.3f}s")
    
#     # -------------------------------------------------------------------------
    
#     # Display
#     vwr = napari.Viewer()
#     vwr.add_labels(arr, opacity=0.5, visible=0)
#     vwr.add_image(edt, blending="additive", colormap="turbo", visible=1)
    
#%% Execute (data) ------------------------------------------------------------

if __name__ == "__main__":

    # Imports
    import time
    import napari
    from skimage import io
    from pathlib import Path

    # Parameters
    # dataset = "em_mito"
    dataset = "fluo_nuclei_instance"
    # dataset = "fluo_nuclei_semantic"
    
    # Paths
    data_path = Path.cwd().parent / "_local" / dataset
    msks_path = list(data_path.glob("*msk_trn.tif"))[0]
    
    # Load masks
    msks = io.imread(msks_path)
    if dataset == "fluo_nuclei_semantic":
        for i, msk in enumerate(msks):
            msk_1 = label(msk == 1)
            msk_2 = label(msk == 2)
            msk_3 = label(msk == 3) 
            msk_2[msk_2 > 0] += np.max(msk_1)
            msk_3[msk_3 > 0] += np.max(msk_2)
            msks[i, ...] = msk_1 + msk_2 + msk_3

    # get_edt() ---------------------------------------------------------------
    
    # Inputs 
    reference = "outlines"
    process = "foreground" 
    normalize = "none"   
    invert = False
    
    t0 = time.time()
    print("get_edt() : ", end="", flush=True)
    
    if dataset == "em_mito":
        
        edt = get_edt(
            msks, 
            reference=reference,
            process=process,
            normalize=normalize,
            invert=invert,
            )
        
    else:

        edt = []
        for msk in msks:
            edt.append(get_edt(
                msk, 
                reference=reference,
                process=process,
                normalize=normalize,
                invert=invert,
                ))
        edt = np.stack(edt)
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # -------------------------------------------------------------------------
    
    # Display
    vwr = napari.Viewer()
    vwr.add_labels(msks, opacity=0.5, visible=0)
    vwr.add_image(edt, blending="additive", colormap="turbo", visible=1)
    