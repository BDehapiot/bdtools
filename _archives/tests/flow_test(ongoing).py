#%% Imports -------------------------------------------------------------------

import sys
import time
import pytest
import numpy as np
from skimage import io
from pathlib import Path

# Skimage
from skimage.filters import gaussian

ROOT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_PATH / 'tests' / 'data' / 'flow'
sys.path.insert(0, str(ROOT_PATH))

from bdtools.flow import get_piv, filt_piv, plot_piv

#%% Function: (transform_coordinates) -----------------------------------------

def transform_coordinates(
        yCoords, xCoords, 
        translation=(0, 0), angle=0, center=(0, 0)
        ):
    
    # Initialize
    angle_rad = np.radians(angle)
    ty, tx = translation
    cy, cx = center
    
    # Rotation matrix
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    R = np.array([
        [cos_theta, -sin_theta, 0], 
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    
    # Translation matrix
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    
    # Combined transformation matrix
    M = np.dot(T, R)
    
    # Translate coordinates to origin
    yCoords -= cy
    xCoords -= cx
    
    # Convert to homogeneous coordinates
    ones = np.ones_like(yCoords)
    coords = np.vstack((xCoords, yCoords, ones))
    
    # Apply transformation matrix
    tCoords = np.dot(M, coords)
    
    # Extract the transformed coordinates
    txCoords = tCoords[0, :] + cx
    tyCoords = tCoords[1, :] + cy
    
    return tyCoords, txCoords

#%% Function: (generate_flow_stack) -------------------------------------------

def generate_flow_stack(
        nPoints=2048, nFrames=8, shape=(512, 512), sigma=5,
        dCoords=0.01, dValues=0.1, 
        dTrans=0.01, dAngle=5, dCenter=0.25, 
        noiseAvg=0.1, noiseStd=0.01,
        ):

    # Initialize
    shape = (shape[0] * 2, shape[1] * 2)
    stack = np.zeros((nFrames, shape[0], shape[1]))
    yCoordShift = int(shape[0] * dCoords)
    xCoordShift = int(shape[1] * dCoords)
    yTransShift = int(shape[0] * dTrans)
    xTransShift = int(shape[1] * dTrans)
    
    # Coordinates & values
    yCoords_ref = np.random.randint(yCoordShift, shape[0] - yCoordShift, nPoints)
    xCoords_ref = np.random.randint(xCoordShift, shape[1] - yCoordShift, nPoints)
    values_ref = np.random.uniform(0.5, 1, nPoints)
    
    # Transformations (translation & rotation)
    yTrans = np.random.uniform(-1, 1, nPoints) * yTransShift
    xTrans = np.random.uniform(-1, 1, nPoints) * xTransShift
    rotAngles = np.random.uniform(-1, 1, nPoints) * dAngle 
    rotCenters = (        
        np.random.randint(-shape[0] * dCenter, shape[0] * (1 + dCenter), nFrames),
        np.random.randint(-shape[1] * dCenter, shape[1] * (1 + dCenter), nFrames),
        )
    
    for f in range(nFrames):
        
        # Update coordinates & values (fluctuations)
        yCoords = yCoords_ref + np.random.randint(-yCoordShift, yCoordShift, nPoints)
        xCoords = xCoords_ref + np.random.randint(-xCoordShift, xCoordShift, nPoints)
        values = values_ref + np.random.uniform(-dValues, dValues, nPoints)
    
        # Transform coordinates
        yCoords, xCoords = transform_coordinates(
                yCoords, xCoords, 
                translation=(yTrans[f], xTrans[f]), 
                angle=rotAngles[f], center=(rotCenters[0][f], rotCenters[1][f])
                )
        yCoords = yCoords.astype(int)
        xCoords = xCoords.astype(int)
        
        # Keep valid coordinates
        yIdx = (yCoords > 1) & (yCoords < shape[0] - 1)
        xIdx = (xCoords > 1) & (xCoords < shape[1] - 1)
        idx = yIdx & xIdx
        yCoords = yCoords[idx]
        xCoords = xCoords[idx]
        values = values[idx]
        
        # Generate random image
        img = np.zeros(shape)
        img[yCoords, xCoords] = values
        img = gaussian(img, sigma=sigma, preserve_range=True)
        img = img / np.max(img)
        
        # Add noise
        img = img + np.random.normal(noiseAvg, noiseStd, shape)
        
        # Append stack
        stack[f, ...] = img
        
    # Crop stack
    stack = stack[
        :, 
        int(shape[0] * 0.25) : int(shape[0] * 0.75), 
        int(shape[1] * 0.25) : int(shape[1] * 0.75),
        ]
    
    # Transformation data
    tData = {
        "yTrans"     : yTrans,
        "xTrans"     : xTrans,
        "rotAngles"  : rotAngles,
        "rotCenters" : rotCenters,
        }
    
    return stack, tData

#%% Test cases ----------------------------------------------------------------

# Random data
stack, tData = generate_flow_stack(
        nPoints=2048, nFrames=8, shape=(512, 512), sigma=5,
        dCoords=0.001, dValues=0.1, 
        dTrans=0.001, dAngle=1, dCenter=0.25, 
        noiseAvg=0.1, noiseStd=0.05,
        )
mask = None

# -----------------------------------------------------------------------------

# # Real data
# stack = io.imread(DATA_PATH / "GBE_eCad.tif")
# mask = io.imread(DATA_PATH / "GBE_eCad_mask.tif")

# -----------------------------------------------------------------------------

t0 = time.time(); 
print(" - get_piv : ", end='')

outputs = get_piv(
        stack,
        intSize=32, srcSize=64, binning=1,
        mask=mask, maskCutOff=0.5,
        parallel=True
        )

t1 = time.time()
print(f"{(t1-t0):<5.2f}s")

# -----------------------------------------------------------------------------

t0 = time.time(); 
print(" - filt_piv : ", end='')

outputs = filt_piv(
        outputs,
        outlier_cutoff=1.5,
        spatial_smooth=5, temporal_smooth=3, iterations_smooth=1,
        parallel=False,
        )

t1 = time.time()
print(f"{(t1-t0):<5.2f}s")

# -----------------------------------------------------------------------------

t0 = time.time(); 
print(" - plot_piv : ", end='')

plot = plot_piv(stack, outputs)

t1 = time.time()
print(f"{(t1-t0):<5.2f}s")

#%%

import napari
viewer = napari.Viewer()
viewer.add_image(plot)
# viewer = napari.Viewer()
# viewer.add_image(outputs["vecU"], name="vecU", contrast_limits=(-5, 5))
# viewer.add_image(outputs["vecV"], name="vecV", contrast_limits=(-5, 5))