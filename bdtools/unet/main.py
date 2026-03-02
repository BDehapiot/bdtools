#%% Imports -------------------------------------------------------------------



#%% Class(UNet) ---------------------------------------------------------------

# Imports
from bdtools.check import Check_parameter

class UNet:
    
    def __init__(self, X, y, parameters):
        
        # Fetch
        self.X = X
        self.y = y
        self.parameters = parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Run
        self.initialize()
        Prepare(self)
        
# -----------------------------------------------------------------------------

    def initialize(self):
        
        # Check input data
        Check_parameter(
            self.X, name="X", 
            ctype=(np.ndarray, list), dtype=float,
            vrange=(0, 1),
            )
      
#%% Class(Prepare) ------------------------------------------------------------

# Imports 
from bdtools.patch import get_patches
from bdtools.mask import process_masks
from bdtools.augment import augment

class Prepare:
    
    def __init__(self, unet):
        self.unet = unet
        self.X = unet.X
        self.y = unet.y
        self.parameters = unet.parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
        
        # Run
        self.prepare_masks()
        self.prepare_patches()
        self.prepare_augment()
        
# -----------------------------------------------------------------------------
    
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

    def prepare_augment(self):
        
        self.X_patches, self.y_patches = augment(
            self.X_patches, self.y_patches, self.augment_iterations,
            invert_p  = self.augment_invert_p,
            gamma_p   = self.augment_gamma_p,
            gblur_p   = self.augment_gblur_p,
            noise_p   = self.augment_noise_p,
            flip_p    = self.augment_flip_p,
            distord_p = self.augment_distord_p,
            )
        
        Check_parameter(
            self.X_patches, name="X_patches", dtype=float, vrange=(0, 1))
        Check_parameter(
            self.y_patches, name="X_patches", dtype=float, vrange=(0, 1))
        
        self.unet.X_patches = self.X_patches
        self.unet.y_patches = self.y_patches
        
    # def augment(
    #         imgs, msks, iterations,
    #         invert_p=0.5, 
    #         gamma_p=0.5,
    #         gblur_p=0.5,
    #         noise_p=0.5,
    #         flip_p=0.5,
    #         distord_p=0.5,
    #         ):
        
        pass

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports
    import numpy as np
    from skimage import io
    from pathlib import Path
    from skimage.measure import label
    
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
    # dataset = "fluo_nuclei_instance"
    dataset = "fluo_nuclei_semantic"
    data_path = Path.cwd().parent.parent / "_local" / dataset
    raw_trn_paths = list(data_path.rglob("*raw_trn.tif"))
    msk_trn_paths = list(data_path.rglob("*msk_trn.tif"))
    raw_val_paths = list(data_path.rglob("*raw_val.tif"))
    msk_val_paths = list(data_path.rglob("*msk_val.tif"))
    
    # Load data
    raw_trn = load_data(raw_trn_paths)
    msk_trn = load_data(msk_trn_paths)
    raw_val = load_data(raw_val_paths)
    msk_val = load_data(msk_val_paths)
    if "nuclei_semantic" in dataset:
        msk_trn = prep_mask(msk_trn)
        msk_val = prep_mask(msk_val)
        
    # Normalization -----------------------------------------------------------
    
    from bdtools.norm import norm_pct
    
    raw_trn = norm_pct(raw_trn, pct_low=0.01, pct_high=99.9, sample_fraction=1)
    
#%% UNet() --------------------------------------------------------------------
    
    parameters = {
        
        # Patch
        "patch_size"         : 256,
        "patch_overlap"      : 0,
        
        # Masks
        "mask_method"        : "skeletons",
                
        # Augment
        "augment_iterations" : 2000,
        "augment_invert_p"   : 0.5,
        "augment_gamma_p"    : 0.5, 
        "augment_gblur_p"    : 0.5, 
        "augment_noise_p"    : 0.5, 
        "augment_flip_p"     : 0.5, 
        "augment_distord_p"  : 0.5,
        
        # Build
        "root_path"          : None,
        "model_name"         : None,
        "backbone"           : "resnet18",
        "activation"         : "sigmoid",
            
        # Train
        "epochs"             : 300,
        "batch_size"         : 4,
        "validation_split"   : 0.2,
        "metric"             : "soft_dice_coef",
        "learning_rate"      : 0.001,
        "patience"           : 100,

        }
    
    unet = UNet(raw_trn, msk_trn, parameters)
    X_patches = unet.X_patches
    y_patches = unet.y_patches
    
    # Display
    import napari
    vwr = napari.Viewer()
    vwr.add_image(np.stack(X_patches))
    vwr.add_image(np.stack(y_patches))