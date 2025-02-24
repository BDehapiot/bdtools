#%% Imports -------------------------------------------------------------------

import numpy as np
from joblib import Parallel, delayed 

# bdtools
from bdtools.mask import get_edt
from bdtools.norm import norm_gcn, norm_pct
from bdtools.patch import extract_patches

# Skimage
from skimage.segmentation import find_boundaries 

#%% Comments ------------------------------------------------------------------

'''
- bug when images and masks are not of the same type
'''

#%% Function : preprocess() ---------------------------------------------------

def preprocess(
        imgs, msks=None,
        patch_size=256, 
        patch_overlap=0,
        img_norm="global",
        msk_type="normal", 
        ):
    
    """ 
    Preprocess images and masks for training or prediction procedures.
    
    If msks=None, only images will be preprocessed.
    Images and masks will be splitted into patches.
    
    Parameters
    ----------
    imgs : 2D ndarray or list of 2D ndarrays (int or float)
        Input image(s).
        
    msks : 2D ndarray or list of 2D ndarrays (bool or int), optional, default=None 
        Input corresponding mask(s).
        If None, only images will be preprocessed.
        
    img_norm : str, default="global"
        - "global" : 0 to 1 normalization considering the full stack.
        - "image"  : 0 to 1 normalization per image.
        
    msk_type : str, default="normal"
        - "normal" : No changes.
        - "edt"    : Euclidean distance transform of binary/labeled objects.
        - "bounds" : Boundaries of binary/labeled objects.

    patch_size : int, default=256
        Size of extracted patches.
        Should be int > 0 and multiple of 2.
    
    patch_overlap : int, default=0
        Overlap between patches.
        Should be int, from 0 to patch_size - 1.
        
    Returns
    -------  
    imgs : 3D ndarray (float32)
        Preprocessed images.
        
    msks : 3D ndarray (float32), optional
        Preprocessed masks.
        
    """
    
    valid_types = ["normal", "edt", "bounds"]
    if msk_type not in valid_types:
        raise ValueError(
            f"Invalid value for msk_type: '{msk_type}'."
            f" Expected one of {valid_types}."
            )

    valid_norms = ["none", "global", "image"]
    if img_norm not in valid_norms:
        raise ValueError(
            f"Invalid value for img_norm: '{img_norm}'."
            f" Expected one of {valid_norms}."
            )
        
    if patch_size <= 0 or patch_size % 2 != 0:
        raise ValueError(
            f"Invalid value for patch_size: '{patch_size}'."
            f" Should be int > 0 and multiple of 2."
            )

    if patch_overlap < 0 or patch_overlap >= patch_size:
        raise ValueError(
            f"Invalid value for patch_overlap: '{patch_overlap}'."
            f" Should be int, from 0 to patch_size - 1."
            )

    # Nested function(s) ------------------------------------------------------

    def normalize(arr, sample_fraction=0.1):
        arr = norm_gcn(arr, sample_fraction=sample_fraction)
        arr = norm_pct(arr, sample_fraction=sample_fraction)
        return arr      
            
    def _preprocess(img, msk=None):

        if msk is None:
            
            img = np.array(img).squeeze()
            
            img = extract_patches(img, patch_size, patch_overlap)
            
            return img
            
        else:
            
            img = np.array(img).squeeze()
            msk = np.array(msk).squeeze()
            
            if msk_type == "normal":
                msk = msk > 0
            elif msk_type == "edt":
                msk = get_edt(
                    msk, 
                    normalize="object", 
                    rescale_factor=1, # buggy if not 1
                    parallel=False,
                    )
            elif msk_type == "bounds":
                msk = find_boundaries(msk)           
            
            img = extract_patches(img, patch_size, patch_overlap)
            msk = extract_patches(msk, patch_size, patch_overlap)
                
            return img, msk
    
    # Execute -----------------------------------------------------------------        
           
    # Normalize images
    if img_norm == "none":
        pass
    if img_norm == "global":
        imgs = normalize(imgs)
    if img_norm == "image":
        if isinstance(imgs, np.ndarray) and imgs.ndim == 2: 
            imgs = normalize(imgs)
        else:
            imgs = [normalize(img) for img in imgs]
            
    # Preprocess
    if msks is None:
                
        if isinstance(imgs, np.ndarray):           
            if imgs.ndim == 2: imgs = [imgs]
            elif imgs.ndim == 3: imgs = list(imgs)
        
        if len(imgs) > 1:
               
            outputs = Parallel(n_jobs=-1)(
                delayed(_preprocess)(img)
                for img in imgs
                )
            imgs = [data for data in outputs]
            imgs = np.stack([arr for sublist in imgs for arr in sublist])
                
        else:
            
            imgs = _preprocess(imgs)
            imgs = np.stack(imgs)
        
        imgs = imgs.astype("float32")
        
        return imgs
    
    else:
        
        if isinstance(imgs, np.ndarray):
            if imgs.ndim == 2: imgs = [imgs]
            elif imgs.ndim == 3: imgs = list(imgs)
        if isinstance(msks, np.ndarray):
            if msks.ndim == 2: msks = [msks]
            elif msks.ndim == 3: msks = list(msks)
        
        if len(imgs) > 1:
            
            outputs = Parallel(n_jobs=-1)(
                delayed(_preprocess)(img, msk)
                for img, msk in zip(imgs, msks)
                )           
            imgs = [data[0] for data in outputs]
            msks = [data[1] for data in outputs]
            imgs = np.stack([arr for sublist in imgs for arr in sublist])
            msks = np.stack([arr for sublist in msks for arr in sublist])
            
        else:
            
            imgs, msks = _preprocess(imgs, msks)
            imgs = np.stack(imgs)
            msks = np.stack(msks)
            
        imgs = imgs.astype("float32")
        msks = msks.astype("float32")
        
        return imgs, msks

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__": 
    
    import time
    import napari
    from skimage import io
    from pathlib import Path
    
    train_path = Path.cwd().parent.parent / "_local" / "fluo_plants"
    rscale_paths = list(train_path.glob("*rscale*"))
    
    imgs, msks = [], []
    for path in rscale_paths:
        if "mask" in path.name:
            msks.append(io.imread(path))
        else:
            imgs.append(io.imread(path))
    imgs = np.stack(imgs)
    msks = np.stack(msks)
    
    # Parameters
    img_norm = "global"
    msk_type = "edt"
    patch_size = 128
    patch_overlap = 0
    
    #
    valid = []
    for i, msk in enumerate(msks):
        if np.max(msk) > 0:
            valid.append(i)
    imgs = imgs[valid]
    msks = msks[valid]
               
    # Preprocess tests
    print("preprocess : ", end="", flush=True)
    t0 = time.time()
    imgs_prp, msks_prp = preprocess(
        imgs, msks, 
        img_norm=img_norm,
        msk_type=msk_type, 
        patch_size=patch_size, 
        patch_overlap=patch_overlap,
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display
    viewer = napari.Viewer()
    viewer.add_image(imgs_prp)
    viewer.add_image(msks_prp) 

#%% Execute -------------------------------------------------------------------

# if __name__ == "__main__":
        
#     # Imports
#     import time
#     import napari
#     from skimage import io
#     from pathlib import Path

#     # Parameters
#     # dataset = "em_mito"
#     dataset = "fluo_nuclei"
#     img_norm = "global"
#     msk_type = "edt"
#     patch_size = 256
#     patch_overlap = 0
    
#     # Paths
#     local_path = Path.cwd().parent.parent / "_local"
#     X_path = local_path / f"{dataset}" / f"{dataset}_trn.tif"
#     y_path = local_path / f"{dataset}" / f"{dataset}_msk_trn.tif"
    
#     # Load images & masks
#     X = io.imread(X_path)
#     y = io.imread(y_path)
    
#     # Preprocess tests
#     print("preprocess : ", end="", flush=True)
#     t0 = time.time()
#     X_prp, y_prp = preprocess(
#         X, y, 
#         img_norm=img_norm,
#         msk_type=msk_type, 
#         patch_size=patch_size, 
#         patch_overlap=patch_overlap,
#         )
#     t1 = time.time()
#     print(f"{t1 - t0:.3f}s")
    
#     # Display
#     viewer = napari.Viewer()
#     viewer.add_image(X_prp)
#     viewer.add_image(y_prp) 
    
    