#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
from bdtools.norm import norm_gcn, norm_pct 

# Skimage
from skimage.transform import rescale 

#%% Function: -----------------------------------------------------------------

def preprocess(arr, cf=1, rf=1, norm=True):
    
    # Normalize
    arr = norm_gcn(arr, sample_fraction=0.01)
    arr = norm_pct(arr, pct_low=0.01, pct_high=99.99, sample_fraction=0.01)
    
    # Crop
    if cf < 1:
        nY, nX = arr.shape[-2], arr.shape[-1]
        y0 = int(nY / 2 - ((nY * cf) / 2))
        y1 = int(y0 + nY * cf)
        x0 = int(nX / 2 - ((nX * cf) / 2))
        x1 = int(x0 + nX * cf)  
        arr = arr[..., y0:y1, x0:x1]
    
    # Rescale
    if rf < 1:
        arr = rescale(arr, (1, rf, rf), order=0)
        
    return arr     

#%% Test ----------------------------------------------------------------------

idx0, idx1 = 0, 20

# -----------------------------------------------------------------------------

# Paths
data_path = Path('D:/local_Klimpel/data')
paths = list(data_path.glob("*.tif"))

# Read
arr = []
for i in range(idx0, idx1):
    arr.append(io.imread(paths[i]).astype("float32"))
arr = np.stack(arr)

# Preprocess
arr = preprocess(arr, cf=0.25, rf=0.25)

#%% Display -------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(arr)