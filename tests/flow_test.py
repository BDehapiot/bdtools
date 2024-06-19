#%% Imports -------------------------------------------------------------------

import sys
import pytest
import numpy as np
from pathlib import Path

# Skimage
from skimage.filters import gaussian

ROOT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_PATH / 'tests' / 'data' / 'patch'
sys.path.insert(0, str(ROOT_PATH))

from bdtools.flow import get_piv

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
        nPoints=1024, nFrames=8, shape=(512, 512), sigma=5,
        dCoords=0.005, dValues=0.1, 
        dTrans=0.01, dAngle=5, dCenter=0.25, 
        noiseAvg=0.1, noiseStd=0.001,
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
    values_ref = np.random.random(nPoints)
    
    # Transformations (translation & rotation)
    yTrans = (np.random.random(nFrames) - 0.5) * 2 * yTransShift
    xTrans = (np.random.random(nFrames) - 0.5) * 2 * xTransShift
    rotAngles = (np.random.random(nFrames) - 0.5) * 2 * dAngle 
    rotCenters = (        
        np.random.randint(-shape[0] * dCenter, shape[0] * (1 + dCenter), nFrames),
        np.random.randint(-shape[1] * dCenter, shape[1] * (1 + dCenter), nFrames),
        )
    
    for f in range(nFrames):
        
        # Update coordinates & values (fluctuations)
        yCoords = yCoords_ref + np.random.randint(-yCoordShift, yCoordShift, nPoints)
        xCoords = xCoords_ref + np.random.randint(-xCoordShift, xCoordShift, nPoints)
        values = values_ref + (np.random.random(nPoints) - 0.5) * dValues
    
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
        img = gaussian(img, sigma=sigma)
        
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

stack, tData = generate_flow_stack(
        nPoints=4096, nFrames=8, shape=(512, 512), sigma=2,
        dCoords=0.001, dValues=0.1, 
        dTrans=0.005, dAngle=2, dCenter=0.25, 
        noiseAvg=0.01, noiseStd=0.001,
        )

outputs = get_piv(
        stack,
        intSize=32, srcSize=64, binning=1,
        mask=None, maskCutOff=1,
        parallel=True
        )

#%%

import napari
viewer = napari.Viewer()
viewer.add_image(stack)
viewer = napari.Viewer()
viewer.add_image(outputs["vecU"])