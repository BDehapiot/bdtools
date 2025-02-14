#%% Imports -------------------------------------------------------------------

import warnings
import numpy as np
import albumentations as A
from joblib import Parallel, delayed 

#%% Comments ------------------------------------------------------------------

'''
High priority:
    - The operations should be nested in the parallel function 
'''

#%% Function : augment() ------------------------------------------------------

def augment(imgs, msks, iterations):
      
    """
    Augment images and masks using random transformations.
    
    The following transformation are applied:
        
        - vertical flip (p = 0.5)      
        - horizontal flip (p = 0.5)
        - rotate 90° (p = 0.5)
        - transpose (p = 0.5)
        - distord (p = 0.5)
    
    The same transformation is applied to an image and its correponding mask.
    Transformation can be tuned by modifying the `operations` variable.
    The function is based on the `albumentations` library.
    https://albumentations.ai/

    Parameters
    ----------
    imgs : 3D ndarray (float)
        Input image(s).
        
    msks : 3D ndarray (float) 
        Input corresponding mask(s).
        
    iterations : int
        The number of augmented samples to generate.
    
    Returns
    -------
    imgs : 3D ndarray (float)
        Augmented image(s).
        
    msks : 3D ndarray (float) 
        Augmented corresponding mask(s).
    
    """
    
    if iterations <= imgs.shape[0]:
        warnings.warn(f"iterations ({iterations}) is less than n of images")
        
    # Nested function(s) ------------------------------------------------------    
    def _augment(imgs, msks):      
        
        idx = np.random.randint(0, len(imgs) - 1)

        if imgs.shape[1] == imgs.shape[2]:
            p0, p1 = 0.5, 0.5
        else:
            p0, p1 = 0.5, 0    
            
        operations = A.Compose([
            
            # Geometric transformations
            # A.VerticalFlip(p=p1),              
            # A.HorizontalFlip(p=p1),
            # A.RandomRotate90(p=p1),
            # A.Transpose(p=p1),
            
            # Distortion-Based transformations
            A.ElasticTransform(p=p0),
            # A.GridDistortion(p=p0),

        ])
        
        outputs = operations(image=imgs[idx,...], mask=msks[idx,...])
        return outputs["image"], outputs["mask"]
    
    # Execute -----------------------------------------------------------------    
    
    outputs = Parallel(n_jobs=-1)(
        delayed(_augment)(imgs, msks)
        for i in range(iterations)
        )
    
    imgs = np.stack([data[0] for data in outputs])
    msks = np.stack([data[1] for data in outputs])
    
    return imgs, msks

    # ops = [
    #     # Distortion
    #     A.Compose([A.GridDistortion(num_steps=5, distort_limit=0.50, p=1)]),
    #     A.Compose([A.GridDistortion(num_steps=7, distort_limit=0.75, p=1)]),
    #     A.Compose([A.GridDistortion(num_steps=10, distort_limit=0.75, p=1)]),
    #     # Geometric Transformations
    #     A.Compose([A.HorizontalFlip(p=1)]),
    #     A.Compose([A.VerticalFlip(p=1)]),
    #     A.Compose([A.RandomRotate90(p=1)]),
    #     # Else
    #     A.Compose([A.Equalize(p=1)]),
    #     A.Compose([A.GaussNoise(std_range=(0.2, 0.3), p=1)]),
    #     A.Compose([A.MotionBlur(blur_limit=15, p=1)]),
    #     ]

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports
    import time
    import napari
    from skimage import io
    from pathlib import Path

    # Parameters
    dataset = "em_mito"
    n = 10 # n of subset images 
    iterations = 100 # n of augmented iterations 
    np.random.seed(42)
    
    # Paths
    local_path = Path.cwd().parent.parent / "_local"
    img_path = local_path / f"{dataset}" / f"{dataset}_trn.tif"
    msk_path = local_path / f"{dataset}" / f"{dataset}_msk_trn.tif"
    
    # Load images & masks
    imgs = io.imread(img_path)
    msks = io.imread(msk_path)
    
    # Subset images & masks
    idxs = np.random.choice(
        np.arange(imgs.shape[0]), size=n, replace=False)
    imgs = imgs[idxs]
    msks = msks[idxs]    
    
    # Augment tests
    print(f"augment iterations = {iterations}", end=" ", flush=True)
    t0 = time.time()
    imgs, msks = augment(imgs, msks, iterations)
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")
    
    # Display
    viewer = napari.Viewer()
    viewer.add_image(imgs)
    viewer.add_labels(msks)

    
        
    pass