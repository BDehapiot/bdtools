#%% Imports -------------------------------------------------------------------

# bdtools
from bdtools.check import Check_parameter
from bdtools.unet.prepare import Prepare

#%% Class(UNet) ---------------------------------------------------------------

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
        
#%% Class(UNet) initialize() --------------------------------------------------

    def initialize(self):
        
        # Check data
        Check_parameter(
            self.X, name="X", 
            ctype=(np.ndarray, list), dtype=float,
            vrange=(0, 1),
            )

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
        "augment_iterations" : 500,
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