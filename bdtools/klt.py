#%% Imports -------------------------------------------------------------------

import cv2
import napari
import warnings
import numpy as np
from joblib import Parallel, delayed

# bdtools
from bdtools.norm import norm_pct

# skimage
from skimage.draw import line
from skimage.morphology import disk, binary_dilation
from skimage.filters.rank import gradient

# scipy
from scipy.ndimage import shift

# matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec

#%% Comments ------------------------------------------------------------------

# Feature detection
feat_params={
    "maxCorners"        : 1000,
    "qualityLevel"      : 1e-4,
    "minDistance"       : 3,
    "blockSize"         : 3,
    "useHarrisDetector" : True,
    "k"                 : 0.04,
    }

# Optical flow
flow_params={
    "winSize"           : (9, 9),
    "maxLevel"          : 3,
    "criteria"          : (5, 0.01),
    "minEigThreshold"   : 1e-4,
    }

'''

Feature Detection Parameters
----------------------------

    maxCorners : int
        Maximum number of features to detect. 
        If more corners exist, only the strongest are returned.
    
    qualityLevel : float
        Minimum accepted quality of features (as a fraction of the best feature). 
        Lower values allow more features.
    
    minDistance : int
        Minimum Euclidean distance between detected features to avoid clustering.
    
    blockSize : int
        Size of the neighborhood (in pixels) used for computing the feature
        quality.
    
    useHarrisDetector : bool
        Indicates whether to use the Harris feature detection method instead of
        the default Shi-Tomasi.
    
    k : float
        Free parameter for the Harris detector.
        Controls the sensitivity of the feature detection
        (commonly between 0.04 and 0.06).

Optical Flow Parameters
-----------------------

    winSize : tuple of int
        Size of the search window at each pyramid level.
        Defines the patch size used to track features between frames.
    
    maxLevel : 3
        Maximum number of pyramid levels to use.
        0 means only the original image is used.
    
    criteria ((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.01)):
        Termination criteria for the iterative search algorithm.
        stops after 5 iterations or when the change is below 0.01.
    
    flags (cv2.OPTFLOW_LK_GET_MIN_EIGENVALS):
        Instructs the algorithm to return the minimum eigenvalue of the gradient 
        matrix as a quality measure instead of the usual tracking error.
    
    minEigThreshold (1e-2):
        Minimum eigenvalue threshold. Features with a value below this are 
        rejected as they are considered too weak for reliable tracking.

'''

#%% Class : KLT ---------------------------------------------------------------

class KLT:
        
    def __init__(
            
            self, arr, msk=None, replace=False,
            
            feat_params={
                "maxCorners"        : 100,
                "qualityLevel"      : 1e-3,
                "minDistance"       : 3,
                "blockSize"         : 3,
                "useHarrisDetector" : True,
                "k"                 : 0.04,
                }, 
            
            flow_params={
                "winSize"         : (9, 9),
                "maxLevel"        : 3,
                "criteria"        : (5, 0.01),
                "minEigThreshold" : 1e-4,
                },
            
            ):
        
        # Fetch
        self.arr = arr
        self.msk = msk
        self.replace = replace
        self.feat_params = feat_params
        self.flow_params = flow_params
        self.format_flow_params()
        
        # Initialize
        self.shape = arr.shape
        self.nT, self.nY, self.nX = arr.shape
        
        # Procedure
        self.preprocess()
        self.process()
        self.get_stats()
        
    def format_flow_params(self):
                
        criteria = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT
        
        self.flow_params = {
            "winSize"         : self.flow_params["winSize"],
            "maxLevel"        : self.flow_params["maxLevel"],
            "criteria"        : (criteria, *self.flow_params["criteria"]),
            "flags"           : cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            "minEigThreshold" : self.flow_params["minEigThreshold"],
            }
        
        return 
    
#%% Method : preprocess() -----------------------------------------------------

    def preprocess(self):
        
        # Convert to uint8
        if self.arr.dtype != "uint8":
            self.arr = norm_pct(self.arr, sample_fraction=0.01)
            self.arr = (self.arr * 255).astype("uint8") 
        if self.msk.dtype != "uint8":
            self.msk = norm_pct(self.msk, sample_fraction=0.01)
            self.msk = (self.msk * 255).astype("uint8")
            
        # Format mask
        if self.msk is None:
            self.msk = np.full_like(self.arr, 255, dtype="uint8")
        elif self.msk.ndim == 2:
            self.msk = np.stack([self.msk] * self.nT, axis=0)
        
#%% Method : process() --------------------------------------------------------
        
    def process(self):
        
        self.n      = []
        self.y      = []
        self.x      = []
        self.dy     = []
        self.dx     = []
        self.norm   = []
        self.status = []
        self.error  = []
        
        # Nested functions ----------------------------------------------------
        
        def invalid_features(f, msk):
            x, y = f[:, 0], f[:, 1]
            out_frm = (x <= 0) | (x >= self.nX) | (y <= 0) | (y >= self.nY)
            valid = ~np.isnan(x) & ~np.isnan(y)
            out_msk = np.zeros_like(x, dtype=bool)
            out_msk[valid] = (
                msk[y[valid].astype(int), x[valid].astype(int)] == 0)
            return out_frm | out_msk
        
        def format_outputs(data, status):
            
            def sort_outputs(data):
                start = np.argmax(~np.isnan(data), axis=0)
                length = np.sum(~np.isnan(data), axis=0)
                sort_idx = np.lexsort((length, start))
                return data[:, sort_idx]
            
            fmt_data = []
            for c, col in enumerate(status.T):
                idxs = np.where(col == 0)[0]
                ends = np.append(idxs[1:], len(col))
                for start, end in zip(idxs, ends):
                    tmp_data = np.full_like(col, np.nan, dtype=float)
                    tmp_data[start:end] = data[start:end, c]
                    fmt_data.append(tmp_data)
            fmt_data = sort_outputs(np.stack(fmt_data).T)
            return fmt_data

        # Execute -------------------------------------------------------------
        
        # Get frame & features (t0)
        img0 = self.arr[0, ...]
        f0 = cv2.goodFeaturesToTrack(
            img0, mask=self.msk[0, ...], **self.feat_params)

        for t in range(1, self.nT):
            
            # Get current image
            img1 = self.arr[t, ...]
            
            # Compute optical flow (between f0 and f1)
            f1, status, error = cv2.calcOpticalFlowPyrLK(
                img0, img1, f0, None, **self.flow_params
                )
            
            # Format data #1
            status, error, f0, f1 = [
                data.squeeze() for data in (status, error, f0, f1)]

            # Remove invalid features
            idx = invalid_features(f1, self.msk[t, ...])
            status[idx] = 0
            f1[status == 0] = np.nan
            
            # Replace lost features *******************************************
            
            if self.replace:
            
                lost_idx = np.where(np.isnan(f1[:, 0]))[0]
                n_lost = len(lost_idx)
                # print(t, n_lost)
                           
                if n_lost > 0:
                    tmp_msk = np.zeros_like(self.msk[t, ...])
                    valid_f1 = f1[~np.isnan(f1[:, 0])].astype(int)
                    tmp_msk[valid_f1[:, 1], valid_f1[:, 0]] = 255
                    tmp_msk = self.msk[t, ...] ^ tmp_msk
                    
                    new_feat_params = self.feat_params.copy()
                    new_feat_params["maxCorners"] = n_lost
                    new_feats = cv2.goodFeaturesToTrack(
                        img1, mask=tmp_msk, **new_feat_params)
                    new_feats = new_feats.squeeze()
                    
                    f1[lost_idx] = new_feats
                    status[lost_idx] = 0
                                                  
            # *****************************************************************
            
            # Append data
            if t == 1:
                self.n.append(np.nansum(f0[:, 1] > 0))
                self.status.append(np.full_like(status, 0))
                self.error.append(error)
                self.x.append(f0[:, 0])
                self.y.append(f0[:, 1])
            self.n.append(np.nansum(f1[:, 1] > 0))  
            self.status.append(status)
            self.error.append(error)
            self.x.append(f1[:, 0])
            self.y.append(f1[:, 1])
           
            # Update previous frame & features 
            img0 = img1
            f0 = f1.reshape(-1, 1, 2)
            
        # Format data #2
        self.status, self.error, self.x, self.y = [
            np.stack(data) for data in (self.status, self.error, self.x, self.y)]
        self.x = format_outputs(self.x, self.status)
        self.y = format_outputs(self.y, self.status)
        self.error = format_outputs(self.error, self.status)
        self.status = format_outputs(self.status, self.status)
        
        # Remove 
        idx = np.where(np.nansum(self.status, axis=0) > 0) 
        self.x = self.x[:, idx[0]]
        self.y = self.y[:, idx[0]]
        self.error = self.error[:, idx[0]]
        self.status = self.status[:, idx[0]]

#%% Method : get_stats() ------------------------------------------------------

    def get_stats(self):
        
        self.dx = np.diff(self.x, axis=0)
        self.dy = np.diff(self.y, axis=0)
        self.dx = np.vstack((np.full((1, self.dx.shape[1]), np.nan), self.dx))
        self.dy = np.vstack((np.full((1, self.dy.shape[1]), np.nan), self.dy))
        self.norm = np.hypot(self.dx, self.dy)
            
#%% Method : get_maps() -------------------------------------------------------

    def get_maps(self):
        
        self.coords_map = np.zeros(self.shape, dtype=bool)
        self.labels_map = np.zeros(self.shape, dtype="uint16")
        self.speeds_map = np.zeros(self.shape, dtype=float)
        self.tracks_map = np.zeros(self.shape, dtype=bool)
        
        for t in range(self.nT):

            # Extract data  
            y1s = self.y[t, :]
            x1s = self.x[t, :]
            labels = np.arange(y1s.shape[0]) + 1
            speeds = self.norm[t]

            # Remove non valid data
            valid_idx = ~np.isnan(y1s)
            y1s = y1s[valid_idx].astype(int)
            x1s = x1s[valid_idx].astype(int)
            labels = labels[valid_idx]
            speeds = speeds[valid_idx]
            
            # Fill maps
            self.coords_map[t, y1s, x1s] = True
            self.labels_map[t, y1s, x1s] = labels
            self.speeds_map[t, y1s, x1s] = speeds
            if t > 0:
                y0s = self.y[t - 1, :]
                x0s = self.x[t - 1, :]
                # valid_idx = ~np.isnan(y0s)
                y0s = y0s[valid_idx].astype(int)
                x0s = x0s[valid_idx].astype(int)
                for x0, y0, x1, y1 in zip(x0s, y0s, x1s, y1s):
                    rr, cc = line(y0, x0, y1, x1)
                    self.tracks_map[t,rr,cc] = True

#%% Method : display() --------------------------------------------------------

    def display(self):
        
        if not hasattr(self, "coords_map"):
            self.get_maps()
        
        viewer = napari.Viewer()
        viewer.add_image(
            self.arr, name="arr", visible=1,
            opacity=0.5
            )
        viewer.add_image(
            self.coords_map, name="coords", visible=1,
            blending='additive'
            )
        viewer.add_image(
            self.tracks_map, name="tracks", visible=1,
            blending='additive'
            )
                
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    import time
    from skimage import io
    from pathlib import Path

    # Paths
    data_path = Path.cwd().parent / "_local" / "flow"
    
    arr_name = "GBE_eCad_40x.tif"
    msk_name = "GBE_eCad_40x_mask.tif"
    
    # dataset = "DC_UtrCH_100x.tif"
    
    # Load
    arr = io.imread(data_path / arr_name)
    msk = io.imread(data_path / msk_name)

#%%
    
    # KLT
    t0 = time.time()
    print("KLT : ", end="", flush=False)
    klt = KLT(
        arr, msk=msk, replace=True,
        feat_params=feat_params, 
        flow_params=flow_params,
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")

    klt.display()

    msk = klt.msk
    status, error = klt.status, klt.error
    n, y, x = klt.n, klt.y, klt.x
    dy, dx, norm = klt.dy, klt.dx, klt.norm
    
    # tmp_msk = klt.tmp_msk
    # valid_f1 = klt.valid_f1
    # new_feats = klt.new_feats
    
    # viewer = napari.Viewer()
    # viewer.add_image(tmp_msk)
    
#%%

# idx = np.where(np.nansum(status, axis=0) > 0) 
# test = status[:, idx[0]]

# def format_outputs(data, status):
    
#     def sort_outputs(data):
#         start = np.argmax(~np.isnan(data), axis=0)
#         length = np.sum(~np.isnan(data), axis=0)
#         sort_idx = np.lexsort((length, start))
#         return data[:, sort_idx]
    
#     fmt_data = []
#     for c, col in enumerate(status.T):
#         idxs = np.where(col == 0)[0]
#         ends = np.append(idxs[1:], len(col))
#         for start, end in zip(idxs, ends):
#             tmp_data = np.full_like(col, np.nan, dtype=float)
#             tmp_data[start:end] = data[start:end, c]
#             fmt_data.append(tmp_data)
#     fmt_data = sort_outputs(np.stack(fmt_data).T)
#     return fmt_data
   
# new_x = format_outputs(x, status)
# new_y = format_outputs(y, status)
# new_status = format_outputs(status, status)
# new_error = format_outputs(error, status)

# dx = np.gradient(new_x, axis=0)
# dy = np.gradient(new_y, axis=0)
# norm = np.hypot(dx, dy)   