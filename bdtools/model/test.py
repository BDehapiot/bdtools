#%% Imports -------------------------------------------------------------------
    
import napari
import numpy as np
from skimage import io
from pathlib import Path

# bdtools
from bdtools.norm import norm_pct

# skimage
from skimage.measure import label

#%% Function : load_data() ----------------------------------------------------

def load_data(dataset="em_mito", display=False):
    
    # Nested function(s) ------------------------------------------------------
    
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
    
    # Execute -----------------------------------------------------------------
    
    # Paths 
    root_path = Path("C:\\Users\\bdeha\\Projects\\bdtools\\_local")
    data_path = root_path / dataset
    print(data_path)
    X_paths = list(data_path.glob("*raw_trn.tif"))
    y_paths = list(data_path.glob("*msk_trn.tif"))
    
    # Load data
    X = load_data(X_paths)
    y = load_data(y_paths)
    if dataset == "nuclei_semantic":
        y = prep_mask(y)
        
    # Normalize data
    if "sat_roads" in dataset:
        X = X.astype("float32") / 255
    else:
        X = norm_pct(
            X, pct_low=0.01, pct_high=99.9, sample_fraction=1)
        
    # Display
    if display:
        vwr = napari.Viewer()
        if isinstance(X, list):
            idx = 0
            vwr.add_image(X[idx])
            vwr.add_image(y[idx])
        else:
            vwr.add_image(X)
            vwr.add_image(y)
        
    return X, y

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Paths
    # dataset = "em_mito"
    # dataset = "fluo_tissue"
    # dataset = "fluo_nuclei_instance"
    dataset = "fluo_nuclei_semantic"
    # dataset = "sat_roads"
    
    # Load data
    X, y = load_data(dataset)