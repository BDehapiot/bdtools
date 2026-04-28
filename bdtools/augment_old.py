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
        imgs, msks, iterations,
        invert_p=0.5, 
        gamma_p=0.5,
        gblur_p=0.5,
        noise_p=0.5,
        flip_p=0.5,
        distord_p=0.5,
        preserve_range=True,
        ):
    
    """
    Augment images and masks using random transformations.
    
    The following transformation are applied:
        
        - adjust gamma (image only)      
        - apply gaussian blur (image only) 
        - add noise (image only) 
        - flip (image & mask)
        - grid distord (image & mask)
    
    If required, image transformations are applied to their correponding masks.
    Transformation probabilities can be set with function arguments.
    Transformation random parameters can be tuned with the params dictionnary.
    Grid distortions are applied with the `albumentations` library.
    https://albumentations.ai/

    Parameters
    ----------
    imgs : 3D ndarray (float)
        Input image(s).
        
    msks : 3D ndarray (float) 
        Input corresponding mask(s).
        
    iterations : int
        The number of augmented samples to generate.
    
    gamma_p, gblur_p, noise_p, flip_p, distord_p : float (0 to 1) 
        Probability to apply the transformation.
    
    Returns
    -------
    imgs : 3D ndarray (float)
        Augmented image(s).
        
    msks : 3D ndarray (float) 
        Augmented corresponding mask(s).
    
    """
    
    # Parameters --------------------------------------------------------------
    
    params = {
               
        # Gamma
        "gamma_low"  : 0.75,
        "gamma_high" : 1.25,
        
        # Gaussian blur
        "sigma_low"  : 1,
        "sigma_high" : 3,
        
        # Noise
        "sgain_low"   : 20,
        "sgain_high"  : 50,
        "rnoise_low"  : 2,
        "rnoise_high" : 4,
        
        # Grid distord
        "nsteps_low"  : 1,
        "nsteps_high" : 10,
        "dlimit_low"  : 0.1,
        "dlimit_high" : 0.5,
        
        }
    
    # Nested functions --------------------------------------------------------
    
    def _invert(img):
        img = 1 - img
        img = np.clip(img, 0.0, 1.0)
        return img
    
    def _gamma(img, gamma=1.0):
        img_mean = np.mean(img)
        if img_mean == 0:
            return img
        img = adjust_gamma(img, gamma=gamma)
        img = img * (img_mean / np.mean(img))
        return img
    
    def _noise(img, shot_gain=0.1, read_noise_std=5):
        img_std = np.std(img) 
        # img = np.random.poisson(img * shot_gain) / shot_gain
        img += np.random.normal(
            loc=0.0, scale=img_std / read_noise_std, size=img.shape)
        return img
    
    def _flip(img, msk):
        if np.random.rand() < 0.5:
            img, msk = np.flipud(img), np.flipud(msk)
        if np.random.rand() < 0.5:
            img, msk = np.fliplr(img), np.fliplr(msk)
        if img.shape[0] == img.shape[1]:
            if np.random.rand() < 0.5:
                k = np.random.choice([-1, 1])
                img = np.rot90(img, k=k)
                msk = np.rot90(msk, k=k)
        return img, msk
    
    def _augment(img, msk):
        
        img = img.copy()
        msk = msk.copy()
        
        if np.random.rand() < invert_p:
            img = _invert(img)
        
        if np.random.rand() < gamma_p:
            gamma = np.random.uniform(
                params["gamma_low"], params["gamma_high"])
            img = _gamma(img, gamma=gamma)
            
        if np.random.rand() < gblur_p:
            sigma = np.random.randint(
                params["sigma_low"], params["sigma_high"])
            img = gaussian(img, sigma=sigma)
            
        if np.random.rand() < noise_p:
            shot_gain = np.random.uniform(
                params["sgain_low"], params["sgain_high"])
            read_noise_std = np.random.randint(
                params["rnoise_low"], params["rnoise_high"])
            img = _noise(
                img, shot_gain=shot_gain, read_noise_std=read_noise_std)
            
        if np.random.rand() < flip_p:
            img, msk = _flip(img, msk)
            
        if np.random.rand() < distord_p:
            num_steps = np.random.randint(
                params["nsteps_low"], params["nsteps_high"])
            distort_limit = np.random.uniform(
                params["dlimit_low"], params["dlimit_high"])
            spatial_transforms = A.Compose([
                A.GridDistortion(
                    num_steps=num_steps, 
                    distort_limit=distort_limit, 
                    p=1
                    )
                ])
            outputs = spatial_transforms(image=img, mask=msk)
            img, msk = outputs["image"], outputs["mask"]
        
        if preserve_range:
            img = norm_pct(img, pct_low=0.01, pct_high=99.99)
        
        return img, msk
        
    # Execute -----------------------------------------------------------------
    
    # Initialize
    imgs = imgs.astype("float32")
    idxs = np.random.choice(
        np.arange(0, imgs.shape[0]), size=iterations)

    outputs = Parallel(n_jobs=-1, backend="threading")(
        delayed(_augment)(imgs[i], msks[i])
        for i in idxs
        )

    imgs = np.stack([data[0] for data in outputs])
    msks = np.stack([data[1] for data in outputs])
    
    return imgs, msks

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports
    import time
    import napari
    import numpy as np
    from skimage import io
    from pathlib import Path
    from skimage.measure import label
    from bdtools.mask import process_masks
    
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
    dataset = "fluo_nuclei_instance"
    # dataset = "fluo_nuclei_semantic"
    # dataset = "sat_roads"
    
    data_path = Path.cwd().parent / "_local" / dataset
    raw_trn_paths = list(data_path.rglob("*raw_trn.tif"))
    msk_trn_paths = list(data_path.rglob("*msk_trn.tif"))
    
    # Load data
    raw_trn = load_data(raw_trn_paths)
    msk_trn = load_data(msk_trn_paths)
    if "nuclei_semantic" in dataset:
        msk_trn = prep_mask(msk_trn)
        
    # Display -----------------------------------------------------------------
    
    # vwr = napari.Viewer()
    # if isinstance(raw_trn, list):
    #     idx = 2
    #     vwr.add_image(raw_trn[idx])
    #     vwr.add_image(msk_trn[idx])
    # else:
    #     vwr.add_image(raw_trn)
    #     vwr.add_image(msk_trn)
        
    # Normalization -----------------------------------------------------------
    
    # from bdtools.norm import norm_pct
        
    # if "sat_roads" in dataset:
    #     raw_trn = raw_trn.astype("float32") / 255
    # else:
    #     raw_trn = norm_pct(
    #         raw_trn, pct_low=0.01, pct_high=99.9, sample_fraction=1)
      
#%% augment() -----------------------------------------------------------------
    
    # Parameters
    mask_method = "binary"
    iterations = 500
    invert_p   = 0
    gamma_p    = 0.5
    gblur_p    = 0.5
    noise_p    = 0.5
    flip_p     = 0.5
    distord_p  = 0.5
           
    # prepare_masks()
    print("prepare_masks() : ", end="", flush=True)
    t0 = time.time()
    
    msk_trn = process_masks(msk_trn, method=mask_method).astype("float32")
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
        
    print(f"min = {np.min(raw_trn)}")
    print(f"max = {np.max(raw_trn)}")
    
    # augment()
    print("augment : ", end="", flush=True)
    t0 = time.time()
    
    raw_trn_aug, msk_trn_aug = augment(
        raw_trn, msk_trn, iterations,
        invert_p=invert_p, 
        gamma_p=gamma_p, 
        gblur_p=gblur_p, 
        noise_p=noise_p, 
        flip_p=flip_p, 
        distord_p=distord_p,
        preserve_range=False,
        )
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
        
    # # Display
    # viewer = napari.Viewer()
    # contrast_limits = [0, 1]
    # viewer.add_image(raw_trn_aug, contrast_limits=contrast_limits)
    # viewer.add_labels(msk_trn_aug.astype("uint8"), visible=0)
    
    print(f"aug min = {np.min(raw_trn_aug)}")
    print(f"aug max = {np.max(raw_trn_aug)}")
