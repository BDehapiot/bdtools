#%% Imports -------------------------------------------------------------------

import numpy as np
import tensorflow as tf

# bdtools
from bdtools.augment import augment
from bdtools.mask import process_masks
from bdtools.patch import extract_patches

#%% Class(Prepare) ------------------------------------------------------------

class Prepare:
    
    def __init__(self, main, X, y=None, display=False):
        self.main = main
        self.X, self.y = X, y
        self.display = display
        self.parameters = main.parameters
        for key, val in self.parameters.items():
            setattr(self, key, val)
                
        # Run
        if self.y is not None:
            self.prepare_masks()
        self.prepare_patches()
        self.split_data()
        if self.augment_iterations is not None:
            self.augment_data()
        self.tensorize_data()
        if self.display:
            self.display_data()
        
        # Pass attributes to main class
        self.main.X = self.X        
        self.main.X_trn = self.X_trn
        self.main.y_trn = self.y_trn
        self.main.trn_tensor = self.trn_tensor
        self.main.y = self.y
        self.main.X_val = self.X_val
        self.main.y_val = self.y_val
        self.main.val_tensor = self.val_tensor
    
#%% Class(Prepare) prepare_masks() --------------------------------------------

    def prepare_masks(self):
        self.y = process_masks(self.y, method=self.mask_method)
        if isinstance(self.y, list):
            self.y = [
                arr.astype("float32") 
                if not np.issubdtype(arr.dtype, np.floating) 
                else arr for arr in self.y
                ]
        if isinstance(self.y, np.ndarray):
            if not np.issubdtype(self.y.dtype, np.floating):
                self.y = self.y.astype("float32")
        
#%% Class(Prepare) prepare_patches() ------------------------------------------
        
    def prepare_patches(self):
        multichannel = True if self.input_shape[-1] > 1 else False
        if isinstance(self.X, list):
            self.X_patches = []
            for arr_X in self.X:
                self.X_patches += extract_patches(
                    arr_X, self.patch_size, self.patch_overlap, 
                    multichannel=multichannel
                    )
            if self.y is not None:  
                self.y_patches = []
                for arr_y in self.y:
                    self.y_patches += extract_patches(
                        arr_y, self.patch_size, self.patch_overlap)
        if isinstance(self.X, np.ndarray):
            self.X_patches = extract_patches(
                self.X, self.patch_size, self.patch_overlap, 
                multichannel=multichannel
                )
            if self.y is not None: 
                self.y_patches = extract_patches(
                    self.y, self.patch_size, self.patch_overlap)
        self.X = np.stack(self.X_patches)
        if self.y is not None: 
            self.y = np.stack(self.y_patches)

#%% Class(Prepare) split_data() -----------------------------------------------

    def split_data(self):
        n_total = self.X.shape[0]
        n_val = int(n_total * self.validation_split)
        idx = np.random.permutation(np.arange(0, n_total))
        self.X_trn = self.X[idx[n_val:]]
        self.X_val = self.X[idx[:n_val]]
        if self.y is not None:  
            self.y_trn = self.y[idx[n_val:]]
            self.y_val = self.y[idx[:n_val]]
        else:
            self.y_trn = None
            self.y_val = None
            
#%% Class(Prepare) augment_data() ---------------------------------------------
            
    def augment_data(self):
        self.X_trn, self.y_trn = augment(
            self.X_trn, 
            msks=self.y_trn, 
            iterations=self.augment_iterations,
            params=self.augment_params,
            gamma_p=self.augment_gamma_p,
            gblur_p=self.augment_gblur_p,
            noise_p=self.augment_noise_p,
            flip_p=self.augment_flip_p,
            distort_p=self.augment_distort_p,
            preserve_range=True,
            )
        
#%% Class(Prepare) tensorize_data() -------------------------------------------

    def tensorize_data(self):

        # Build training & validation tensors
        trn_target = self.y_trn if self.y_trn is not None else self.X_trn
        val_target = self.y_val if self.y_val is not None else self.X_val
        self.trn_tensor = tf.data.Dataset.from_tensor_slices(
            (self.X_trn, trn_target))
        self.val_tensor = tf.data.Dataset.from_tensor_slices(
            (self.X_val, val_target))
        
        # Shuffle training tensors
        self.trn_tensor = self.trn_tensor.shuffle(
            buffer_size=min(len(self.X_trn), 1000))
        
        # Optimizations
        self.trn_tensor = (
            self.trn_tensor
            .cache()
            .shuffle(buffer_size=self.X_trn.shape[0])
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
            )
        self.val_tensor = (
            self.val_tensor
            .cache()
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
            )

#%% Class(Prepare) display_data() ---------------------------------------------

    def display_data(self):
        import napari
        vwr = napari.Viewer()
        vwr.add_image(self.X_trn, name="X_trn")
        if self.y is not None:
            vwr.add_image(self.y_trn, name="y_trn")
            vwr.grid.enabled = True