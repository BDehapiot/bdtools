#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
from bdtools.piv import get_piv, filt_piv

#%% Function: get_transform_matrix --------------------------------------------

from scipy.optimize import minimize

# -----------------------------------------------------------------------------
    
def estimate_center_of_rotation(u, v):
    
    def divergence(center):
        cx, cy = center
        dx = np.arange(u.shape[1]) - cx
        dy = np.arange(u.shape[0])[:, None] - cy
        dist_squared = dx**2 + dy**2
        weights = 1 / (dist_squared + 1e-5)  # Add small value to avoid division by zero
        du = u * weights
        dv = v * weights
        return np.sum(du**2 + dv**2)

    result = minimize(divergence, x0=[u.shape[1] / 2, v.shape[0] / 2])
    
    return result.x

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

#%%

t0 = time.time()
print(" - get_piv : ", end='')

outputs = get_piv(arr, intSize=64, srcSize=128, binning=1)
vecU, vecV = outputs["vecU"], outputs["vecV"]

t1 = time.time()
print(f"{(t1-t0):<5.2f}s")

# -----------------------------------------------------------------------------

t0 = time.time()
print(" - filt_piv : ", end='')

outputs = filt_piv(
        outputs,
        outlier_cutoff=1.25,
        spatial_smooth=9, temporal_smooth=1, iterations_smooth=3,
        parallel=False,
        )
vecU_f, vecV_f = outputs["vecU"], outputs["vecV"]

t1 = time.time()
print(f"{(t1-t0):<5.2f}s")

# -----------------------------------------------------------------------------

rotCenter = []
for u, v in zip(vecU_f, vecV_f):
    rotCenter.append(estimate_center_of_rotation(u, v))

# -----------------------------------------------------------------------------

# viewer = napari.Viewer()
# viewer.add_image(vecU, contrast_limits=(-100, 100))
# viewer.add_image(vecU_f, contrast_limits=(-100, 100))
