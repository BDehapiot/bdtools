#%% Imports -------------------------------------------------------------------

import numpy as np

# bdtools
from bdtools.patch import get_patches
from bdtools.mask import process_masks
from bdtools.augment import augment

#%% Class(Prepare) ------------------------------------------------------------

class Prepare:
    
    def __init__(self, unet):
        self.unet = unet
        self.X, self.y = unet.X, unet.y
        self.parameters = unet.parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Run
        self.prepare_masks()
        self.prepare_patches()
        self.split_data()
        if self.augment_iterations > 0:
            self.augment_data()
        
        # Finalize
        self.unet.X_trn = self.X_trn
        self.unet.y_trn = self.y_trn
        self.unet.X_val = self.X_val
        self.unet.y_val = self.y_val
        self.unet.X_patches = self.X_patches
        self.unet.y_patches = self.y_patches
    
#%% Class(Prepare) function(s) ------------------------------------------------

    def prepare_masks(self):
        self.y = process_masks(self.y, method=self.mask_method)
        if isinstance(self.y, list):
            self.y = [
                arr.astype("float32") if not np.issubdtype(arr.dtype, np.floating) 
                else arr for arr in self.y
                ]
        if isinstance(self.y, np.ndarray):
            if not np.issubdtype(self.y.dtype, np.floating):
                self.y = self.y.astype("float32")

    def prepare_patches(self):
        if isinstance(self.X, list):
            self.X_patches, self.y_patches = [], []
            for arr_X, arr_y in zip(self.X, self.y):
                X_patches = get_patches(
                    arr_X, self.patch_size, self.patch_overlap)
                y_patches = get_patches(
                    arr_y, self.patch_size, self.patch_overlap)
                self.X_patches += X_patches
                self.y_patches += y_patches
        if isinstance(self.X, np.ndarray):
            self.X_patches = get_patches(
                self.X, self.patch_size, self.patch_overlap)
            self.y_patches = get_patches(
                self.y, self.patch_size, self.patch_overlap)
        self.X_patches = np.stack(self.X_patches)
        self.y_patches = np.stack(self.y_patches)

    def split_data(self):
        n_total = self.X_patches.shape[0]
        n_val = int(n_total * self.validation_split)
        idx = np.random.permutation(np.arange(0, n_total))
        self.X_trn = self.X_patches[idx[n_val:]] 
        self.y_trn = self.y_patches[idx[n_val:]]
        self.X_val = self.X_patches[idx[:n_val]]
        self.y_val = self.y_patches[idx[:n_val]]

    def augment_data(self):
        self.X_trn, self.y_trn = augment(
            self.X_trn, self.y_trn, self.augment_iterations,
            invert_p  = self.augment_invert_p,
            gamma_p   = self.augment_gamma_p,
            gblur_p   = self.augment_gblur_p,
            noise_p   = self.augment_noise_p,
            flip_p    = self.augment_flip_p,
            distord_p = self.augment_distord_p,
            preserve_range=True,
            )

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    pass