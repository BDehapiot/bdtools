#%% Imports -------------------------------------------------------------------

# bdtools
from bdtools.model import unet

#%% Class(Train) --------------------------------------------------------------

class Train:
    
    def __init__(self, X, y, parameters, X_val=None, y_val=None):
        
        # Fetch
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.parameters = parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
        
        pass

#%% Class(Train) initialize() -------------------------------------------------

#%% Class(Train) prepare() ----------------------------------------------------

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports
    import napari
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
    dataset = "fluo_nuclei_instance"
    # dataset = "fluo_nuclei_semantic"
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

#%% Train() -------------------------------------------------------------------
    
    parameters = {
        
        # Prepare
        
        
        # Build
        "root_path"        : None,
        "model_name"       : None,
        "backbone"         : "resnet18",
        "activation"       : "sigmoid",
            
        # Train
        "epochs"           : 300,
        "batch_size"       : 4,
        "validation_split" : 0.2,
        "metric"           : "soft_dice_coef",
        "learning_rate"    : 0.001,
        "patience"         : 100,

        }

    train = Train(raw_trn, msk_trn, parameters)

    
    