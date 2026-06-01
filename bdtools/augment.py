#%% Imports -------------------------------------------------------------------

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import numpy as np
import albumentations as A
from joblib import Parallel, delayed 

# bdtools
from bdtools.norm import norm_pct

# skimage
from skimage.filters import gaussian
from skimage.exposure import adjust_gamma

#%% Function : augment() ------------------------------------------------------

def augment(
        imgs, 
        iterations=100,
        msks=None,
        params=None,
        gamma_p=0.5,
        gblur_p=0.5,
        noise_p=0.5,
        flip_p=0.5,
        distort_p=0.5,
        balance=False,
        preserve_range=True,
        ):
    
    """
    Augment images and masks using random transformations.
    
    The following transformation are applied:
        
        - adjust gamma (image only)      
        - apply gaussian blur (image only) 
        - add noise (image only) 
        - flip (image & mask)
        - grid distort (image & mask)
    
    If required, image transformations are applied to their correponding masks.
    Transformation probabilities can be set with function arguments.
    Transformation random parameters can be tuned with the params dictionnary.
    Grid distortions are applied with the `albumentations` library.
    https://albumentations.ai/

    Parameters
    ----------
    imgs : 3D, 4D ndarray (int, float)
        Input image(s).
        
    msks (optional) : 1D (int) or 3D ndarray (float) 
        Input corresponding mask(s).
        Pass 1D ndarray of int for object classification.
        Pass None to only transform images without masks.
        
    iterations : int
        The number of augmented samples to generate.
    
    gamma_p, gblur_p, noise_p, flip_p, distort_p : float (0 to 1) 
        Probability to apply the transformation.
        
    balance : bool
        Evenly augment data according to classes
    
    Returns
    -------
    imgs : 3D, 4D ndarray (float)
        Augmented image(s).
        
    msks (optional) : 1D (int) or 3D ndarray (float) 
        Augmented corresponding mask(s).
    
    """
    
    # Parameters --------------------------------------------------------------
    
    if params is None:
        
        params = {
                   
            # Gamma
            "gamma_low"          : 0.75,
            "gamma_high"         : 1.25,
            "gamma_chn"          : "independent",
            
            # Gaussian blur
            "gblur_sigma_low"    : 1,
            "gblur_sigma_high"   : 3,
            "gblur_chn"          : "shared",
            
            # Noise
            "noise_gain_low"     : 30,
            "noise_gain_high"    : 60,
            "noise_std_low"      : 3,
            "noise_std_high"     : 6,
            "noise_chn"          : "independent",
            
            # Grid distort
            "distort_steps_low"  : 1,
            "distort_steps_high" : 10,
            "distort_limit_low"  : 0.1,
            "distort_limit_high" : 0.5,
            
            }
    
    # Nested function : _gamma() ----------------------------------------------
        
    def _gamma(img):

        g0 = params["gamma_low" ]
        g1 = params["gamma_high"]

        if img.ndim == 3:
            
            nC = img.shape[-1]
            
            if params["gamma_chn"] == "independent":
                gamma = np.random.uniform(g0, g1, size=nC)
            elif params["gamma_chn"] == "shared":
                gamma = np.repeat(np.random.uniform(g0, g1), nC)
            
            for c in range(nC):
                chn = img[..., c]
                chn_mean = np.mean(chn)
                chn = adjust_gamma(chn, gamma=gamma[c])
                chn = chn * (chn_mean / np.mean(chn))
                img[..., c] = chn
        
        elif img.ndim == 2:
            
            gamma = np.random.uniform(g0, g1)
            img_mean = np.mean(img)
            img = adjust_gamma(img, gamma=gamma)
            img = img * (img_mean / np.mean(img))
            
        return img
    
    # Nested function : _gblur() ----------------------------------------------
    
    def _gblur(img):
        
        s0 = params["gblur_sigma_low" ]
        s1 = params["gblur_sigma_high"]
        
        if img.ndim == 3:
            
            nC = img.shape[-1]
            
            if params["gblur_chn"] == "independent":
                sigma = np.random.randint(s0, s1, size=nC)
            elif params["gblur_chn"] == "shared":
                sigma = np.repeat(np.random.randint(s0, s1), nC)
                
            for c in range(nC):
                img[..., c] = gaussian(img[..., c], sigma[c])
                
        elif img.ndim == 2:
            
            sigma = np.random.randint(s0, s1)
            img = gaussian(img, sigma)
            
        return img
    
    # Nested function : _noise() ----------------------------------------------
    
    def _noise(img):
        
        g0 = params["noise_gain_low" ]
        g1 = params["noise_gain_high"]
        s0 = params["noise_std_low"  ]
        s1 = params["noise_std_high" ]
    
        if img.ndim == 3:
            
            nC = img.shape[-1]
            if params["noise_chn"] == "independent":
                gain = np.random.uniform(g0, g1, size=nC)
                std = np.random.randint(s0, s1, size=nC)
            elif params["noise_chn"] == "shared":
                gain = np.repeat(np.random.uniform(g0, g1), nC)
                std = np.repeat(np.random.randint(s0, s1), nC)
                
            for c in range(nC):
                chn = img[..., c]
                chn_std = np.std(chn)
                chn = np.random.poisson(chn * gain[c]) / gain[c]
                chn += np.random.normal(
                    loc=0.0, scale=chn_std / std[c], size=chn.shape)
                img[..., c] = chn
                
        elif img.ndim == 2:
            
            gain = np.random.uniform(g0, g1)
            std = np.random.randint(s0, s1)
            img_std = np.std(img)
            img = np.random.poisson(img * gain) / gain
            img += np.random.normal(
                loc=0.0, scale=img_std / std, size=img.shape)
            
        return img
    
    # Nested function : _flip() -----------------------------------------------
    
    def _flip(img, msk):

        if np.random.rand() < 0.5:
            img = np.flip(img, axis=0)
            if msk is not None : 
                msk = np.flip(msk, axis=0)
        if np.random.rand() < 0.5:
            img = np.flip(img, axis=1)
            if msk is not None : 
                msk = np.flip(msk, axis=1)
        
        if img.shape[0] == img.shape[1]:
            if np.random.rand() < 0.5:
                k = np.random.choice([-1, 1])
                img = np.rot90(img, k=k, axes=(0, 1))
                if msk is not None : 
                    msk = np.rot90(msk, k=k, axes=(0, 1))
        
        return img, msk
    
    # Nested function : _distort() --------------------------------------------
    
    def _distort(img, msk):
        
        s0 = params["distort_steps_low" ]
        s1 = params["distort_steps_high"]
        l0 = params["distort_limit_low" ]
        l1 = params["distort_limit_high"]
        
        steps = np.random.randint(s0, s1)
        limit = np.random.uniform(l0, l1)
        spatial_transforms = A.Compose([
            A.GridDistortion(num_steps=steps, distort_limit=limit, p=1)])
        if msk is not None:
            outputs = spatial_transforms(image=img, mask=msk)
            img, msk = outputs["image"], outputs["mask"]
        else:
            outputs = spatial_transforms(image=img)
            img, msk = outputs["image"], None
        
        return img, msk
                
    # Nested function : _augment() --------------------------------------------
    
    def _augment(img, msk):
        
        img = img.copy()
        if msk is not None:
            msk = msk.copy()
        
        # Stats (before augmentation)
        min_0, max_0 = np.min(img), np.max(img)
        pct_low_0, pct_high_0 = np.percentile(img, [0.01, 99.99])

        # Apply transformations -----------------------------------------------
        
        if np.random.rand() < gamma_p:
            img = _gamma(img)
            
        if np.random.rand() < gblur_p:
            img = _gblur(img)
            
        if np.random.rand() < noise_p:
            img = _noise(img)
            
        if np.random.rand() < flip_p:
            img, msk = _flip(img, msk)
            
        if np.random.rand() < distort_p:
            img, msk = _distort(img, msk)
        
        # Normalization -------------------------------------------------------
        
        if preserve_range:
            
            # Stats (after augmentation)
            pct_low_1, pct_high_1 = np.percentile(img, [0.01, 99.99])
            
            if max_0 > 0:
                img = np.interp(
                    img, (pct_low_1, pct_high_1), (pct_low_0, pct_high_0))
                img = np.clip(img, min_0, max_0)
        
        return img, msk
        
    # Execute -----------------------------------------------------------------
       
    # Initialize
    imgs = imgs.astype("float32")
    if isinstance(msks, np.ndarray) and msks.ndim == 1:
        clss = msks.copy()
        msks = None
        
    # Get random indexes (balance if required)
    if "clss" in locals() and balance:
        idxs = []
        cls_unique = np.unique(clss)
        for c in cls_unique:
            idxs.append(
                np.random.choice(
                    np.argwhere(clss == c).squeeze(),
                    size=iterations // len(cls_unique)
                    )
                )
        idxs = np.concatenate(idxs, axis=0)
    else:
        idxs = np.random.choice(
            np.arange(0, imgs.shape[0]), size=iterations)
    if "clss" in locals():
        clss = clss[idxs]
            
    # Augment
    outputs = Parallel(n_jobs=-1, backend="threading")(
        delayed(_augment)(imgs[i], msks[i] if msks is not None else None)
        for i in idxs
        )
    
    # output(s)
    imgs = np.stack([data[0] for data in outputs])
    if msks is not None:
        msks = np.stack([data[1] for data in outputs])
        return imgs, msks
    if "clss" in locals():
        return imgs, clss
    return imgs

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    from bdtools.test import load_data
    from skimage.transform import downscale_local_mean
    
    # Paths
    # dataset = "em_mito"
    # dataset = "fluo_tissue"
    # dataset = "fluo_nuclei_instance"
    dataset = "fluo_nuclei_semantic"
    # dataset = "sat_roads"
    # dataset = "chess_class"
    
    # Load data
    X, y = load_data(dataset)
    
    # Downscale (optional)
    if dataset == "chess_class":
        X = downscale_local_mean(X, (1, 4, 4, 1))
      
#%% augment() -----------------------------------------------------------------
    
    # Imports
    import time
    from bdtools.mask import process_masks

    # Parameters
    y = y
    msk_method = "binary"
    iterations = 1024
    gamma_p    = 0.5
    gblur_p    = 0.5
    noise_p    = 0.5
    flip_p     = 0.5
    distort_p  = 0.5
           
    # -------------------------------------------------------------------------
    
    params = {
               
        # Gamma
        "gamma_low"          : 0.75,
        "gamma_high"         : 1.25,
        "gamma_chn"          : "independent",
        
        # Gaussian blur
        "gblur_sigma_low"    : 1,
        "gblur_sigma_high"   : 3,
        "gblur_chn"          : "shared",
        
        # Noise
        "noise_gain_low"     : 30,
        "noise_gain_high"    : 60,
        "noise_std_low"      : 3,
        "noise_std_high"     : 6,
        "noise_chn"          : "independent",
        
        # Grid distort
        "distort_steps_low"  : 1,
        "distort_steps_high" : 10,
        "distort_limit_low"  : 0.1,
        "distort_limit_high" : 0.5,
        
        }
    
    # params = None

    # -------------------------------------------------------------------------
    
    # prepare_masks()
    print("prepare_masks() : ", end="", flush=True)
    t0 = time.time()
    
    if y is not None:
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = process_masks(y, method=msk_method).astype("float32")
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")

    # -------------------------------------------------------------------------

    # augment()
    print("augment() : ", end="", flush=True)
    t0 = time.time()
    
    outputs = augment(
        X, 
        iterations=iterations,
        msks=y, 
        params=params,
        gamma_p=gamma_p, 
        gblur_p=gblur_p, 
        noise_p=noise_p, 
        flip_p=flip_p, 
        distort_p=distort_p,
        preserve_range=True,
        balance=True,
        )
    
    if y is not None:
        X_aug, y_aug = outputs
    else:
        X_aug = outputs
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Stats -------------------------------------------------------------------
    
    print("---")
    print(f"avg = {np.mean(X):.3f}")
    print(f"std = {np.std(X):.3f}")
    print(f"min = {np.min(X):.3f}")
    print(f"max = {np.max(X):.3f}")
    print("---")
    print(f"aug avg = {np.mean(X_aug):.3f}")
    print(f"aug std = {np.std(X_aug):.3f}")
    print(f"aug min = {np.min(X_aug):.3f}")
    print(f"aug max = {np.max(X_aug):.3f}")
        
    # Display -----------------------------------------------------------------
    
    import napari
    
    vwr = napari.Viewer()
    contrast_limits = [
        np.percentile(X_aug, 0.01),
        np.percentile(X_aug, 99.99),
        ]
    if y is not None:
        if isinstance(y, np.ndarray) and y.ndim > 1:
            vwr.add_labels(
                y_aug.astype("uint8"), name="y_aug", visible=1,
                )
            vwr.grid.enabled = True
    vwr.add_image(
        X_aug, name="X_aug", visible=1,
        contrast_limits=contrast_limits,
        )
    vwr.reset_view()
