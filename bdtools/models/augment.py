#%% Imports -------------------------------------------------------------------

import numpy as np
import albumentations as A
from joblib import Parallel, delayed 

# skimage
from skimage.filters import gaussian
from skimage.exposure import adjust_gamma

#%% Function : augment() ------------------------------------------------------

def augment(
        imgs, msks, iterations,
        gamma_p=0.5, gblur_p=0.5, noise_p=0.5, flip_p=0.5, distord_p=0.5,
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
        "sgain_low"   : 0.50,
        "sgain_high"  : 1.00,
        "rnoise_low"  : 6,
        "rnoise_high" : 10,
        
        # Grid distord
        "nsteps_low"  : 1,
        "nsteps_high" : 10,
        "dlimit_low"  : 0.1,
        "dlimit_high" : 0.5,
        
        }
    
    # Nested functions --------------------------------------------------------
    
    def _gamma(img, gamma=1.0):
        img_mean = np.mean(img)
        img = adjust_gamma(img, gamma=gamma)
        img = img * (img_mean / np.mean(img))
        return img
    
    def _noise(img, shot_gain=0.5, read_noise_std=5):
        img_std = np.std(img) 
        img = np.random.poisson(img * shot_gain) / shot_gain
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
                img = np.rot90(img, k=np.random.choice([-1, 1]))
                msk = np.rot90(msk, k=np.random.choice([-1, 1]))
        return img, msk
    
    def _augment(img, msk):
        
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
    from skimage import io
    from pathlib import Path

    # Parameters
    dataset = "em_mito"
    # dataset = "fluo_nuclei"
    iterations = 500 # n of augmented iterations 
    
    # Paths
    local_path = Path.cwd().parent.parent / "_local"
    img_path = local_path / f"{dataset}" / f"{dataset}_trn.tif"
    msk_path = local_path / f"{dataset}" / f"{dataset}_msk_trn.tif"
    
    # Load images & masks
    imgs = io.imread(img_path)
    msks = io.imread(msk_path)
        
    # Augment tests
    print(f"augment iterations = {iterations}", end=" ", flush=True)
    t0 = time.time()
    aug_imgs, aug_msks = augment(imgs, msks, iterations)
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")
        
    # Display
    viewer = napari.Viewer()
    contrast_limits = [0, 255]
    viewer.add_image(aug_imgs, contrast_limits=contrast_limits)
    viewer.add_labels(aug_msks)