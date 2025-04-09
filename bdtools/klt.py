#%% Imports -------------------------------------------------------------------

import cv2
import napari
import warnings
import numpy as np

# bdtools
from bdtools.norm import norm_pct

# skimage
from skimage.draw import line

# matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec

#%% Comments ------------------------------------------------------------------

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
        matrix as a quality measure instead of the usual tracking egval.
    
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
                "winSize"           : (9, 9),
                "maxLevel"          : 3,
                "criteria"          : (5, 0.01),
                "minEigThreshold"   : 1e-4,
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
        
        if self.arr.dtype != "uint8":
            self.arr = norm_pct(self.arr, sample_fraction=0.01)
            self.arr = (self.arr * 255).astype("uint8") 
        if self.msk is None:
            self.msk = np.full_like(self.arr, 255, dtype="uint8") 
        else:
            self.msk = norm_pct(self.msk, sample_fraction=0.01)
            self.msk = (self.msk * 255).astype("uint8")
        if self.msk.ndim == 2:
            self.msk = np.stack([self.msk] * self.nT, axis=0)
        
#%% Method : process() --------------------------------------------------------
        
    def process(self):
        
        self.y      = []
        self.x      = []
        self.status = []
        self.egval  = []       
        
        # Nested functions ----------------------------------------------------
        
        def invalid_features(f, msk):
            x, y = f[:, 0], f[:, 1]
            out_frm = (x <= 1) | (x >= self.nX - 1) | (y <= 1) | (y >= self.nY - 1)
            f[out_frm] = np.nan
            valid = ~np.isnan(x) & ~np.isnan(y)
            out_msk = np.zeros_like(x, dtype=bool)
            out_msk[valid] = (
                msk[y[valid].astype(int), x[valid].astype(int)] == 0)
            return out_frm | out_msk
        
        def replace_features(f1, status, msk):
            
            # Identify lost tracks
            lost_idx = np.where(np.isnan(f1[:, 0]))[0]
            n_lost = len(lost_idx)
            
            if n_lost > 0:
                
                # Create mask
                tmp_msk = np.zeros_like(msk)
                valid = f1[~np.isnan(f1[:, 0])].astype(int)
                tmp_msk[valid[:, 1], valid[:, 0]] = 255
                tmp_msk = msk ^ tmp_msk

                # Create new features
                feat_params = self.feat_params.copy()
                feat_params["maxCorners"] = n_lost
                fnew = cv2.goodFeaturesToTrack(
                    img1, mask=tmp_msk, **feat_params)
                fnew = fnew.squeeze()
                
                # Replace 
                f1[lost_idx] = fnew
                status[lost_idx] = 2
                
        def format_data(data, status):
        
            def sort_outputs(data):
                start = np.argmax(~np.isnan(data), axis=0)
                length = np.sum(~np.isnan(data), axis=0)
                sort_idx = np.lexsort((length, start))
                return data[:, sort_idx]
        
            fmt_data = []
            for c, col in enumerate(status.T):
                idxs = np.where(col == 2)[0]
                for i, start in enumerate(idxs):
                    tmp_data = np.full_like(col, np.nan, dtype=float)
                    if len(idxs) == 1:
                        end = np.nonzero(col)[0][-1] + 1
                    elif i + 1 < len(idxs):
                        end = idxs[i + 1]
                    else:
                        end = len(col)
                    tmp_data[start:end] = data[start:end, c]
                    fmt_data.append(tmp_data)
            return sort_outputs(np.stack(fmt_data).T)

        # Execute -------------------------------------------------------------
        
        # Get frame & features (t0)
        img0 = self.arr[0, ...]
        f0 = cv2.goodFeaturesToTrack(
            img0, mask=self.msk[0, ...], **self.feat_params)

        for t in range(1, self.nT):
            
            # Get current image
            img1 = self.arr[t, ...]
            
            # Compute optical flow (between f0 and f1)
            f1, status, egval = cv2.calcOpticalFlowPyrLK(
                img0, img1, f0, None, **self.flow_params
                )
            status, egval, f0, f1 = [
                data.squeeze() for data in (status, egval, f0, f1)]

            # Remove invalid features
            idx = invalid_features(f1, self.msk[t, ...])
            status[idx] = 0
            f1[status == 0] = np.nan

            # Replace lost features
            if self.replace:
                replace_features(f1, status, msk)

            # Append data
            if t == 1:
                self.status.append(np.full_like(status, 2))
                self.egval.append(egval)
                self.x.append(f0[:, 0])
                self.y.append(f0[:, 1])
            self.status.append(status)
            self.egval.append(egval)
            self.x.append(f1[:, 0])
            self.y.append(f1[:, 1])
           
            # Update previous frame & features 
            img0 = img1
            f0 = f1.reshape(-1, 1, 2)
            
        # Format data
        self.status, self.egval, self.x, self.y = [
            np.stack(data) for data in (self.status, self.egval, self.x, self.y)]
        self.x = format_data(self.x, self.status)
        self.y = format_data(self.y, self.status)
        self.egval = format_data(self.egval, self.status)
        self.status = format_data(self.status, self.status)
        
        # Remove 
        idx = np.where(np.nansum(self.status, axis=0) > 2) 
        self.x = self.x[:, idx[0]]
        self.y = self.y[:, idx[0]]
        self.egval = self.egval[:, idx[0]]
        self.status = self.status[:, idx[0]]

#%% Method : get_stats() ------------------------------------------------------

    def get_stats(self):
        
        def get_diff(data):
            diff = np.diff(data, axis=0)
            diff = np.vstack((np.full((1, data.shape[1]), np.nan), diff))
            return diff
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.n = np.sum(~np.isnan(self.status), axis=1)
            self.lost = np.sum(self.status == 2, axis=1)
            self.lost_cum = np.nancumsum(self.lost[1:], axis=0) 
            self.dx = get_diff(self.x)
            self.dy = get_diff(self.y)
            self.dx_avg = np.nanmean(self.dx, axis=1)
            self.dy_avg = np.nanmean(self.dy, axis=1)
            self.dx_avg_cum = np.nancumsum(self.dx_avg, axis=0) 
            self.dy_avg_cum = np.nancumsum(self.dy_avg, axis=0) 
            self.norm = np.hypot(self.dx, self.dy)
            self.norm_avg = np.nanmean(self.norm, axis=1)
            self.egval_avg = np.nanmean(self.egval, axis=1)

#%% Method : get_maps() -------------------------------------------------------

    def get_maps(self):
        
        def expand_coordinates(ys, xs):
            offsets = np.array([
                [-1, -1], [-1,  0], [-1,  1],  
                [ 0, -1], [ 0,  0], [ 0,  1],  
                [ 1, -1], [ 1,  0], [ 1,  1],   
                ])
            yss = (ys[:, np.newaxis] + offsets[:, 0]).reshape(-1)
            xss = (xs[:, np.newaxis] + offsets[:, 1]).reshape(-1)
            return yss, xss
    
        self.status_map = np.zeros(self.shape, dtype="uint8")
        self.labels_map = np.zeros(self.shape, dtype="uint16")
        self.tracks_norm_map = np.full(self.shape, np.nan, dtype=float)
        self.tracks_egval_map = np.full(self.shape, np.nan, dtype=float)
        
        for t in range(self.nT):
    
            # Extract data  
            y1s = self.y[t, :]
            x1s = self.x[t, :]
            status = self.status[t, :]
            egval = self.egval[t, :]
            labels = np.arange(y1s.shape[0]) + 1
            norm = self.norm[t]
    
            # Remove non valid data
            valid_idx = ~np.isnan(y1s)
            y1s = y1s[valid_idx].astype(int)
            x1s = x1s[valid_idx].astype(int)
            status = status[valid_idx]
            egval = egval[valid_idx]
            labels = labels[valid_idx]
            norm = norm[valid_idx]
            
            # Fill maps
            y1ss, x1ss = expand_coordinates(y1s, x1s)
            self.status_map[t, y1ss, x1ss] = np.repeat(status, 9)
            self.labels_map[t, y1ss, x1ss] = np.repeat(labels, 9)
            if t > 0:
                y0s = klt.y[t - 1, :]
                x0s = klt.x[t - 1, :]
                y0s = y0s[valid_idx]
                x0s = x0s[valid_idx]
                for i, (x0, y0, x1, y1) in enumerate(zip(x0s, y0s, x1s, y1s)):
                    if ~np.isnan(x0):
                        x0, y0 = int(x0), int(y0)
                        rr, cc = line(y0, x0, y1, x1)
                        self.tracks_norm_map[t,rr,cc] = norm[i]
                        self.tracks_egval_map[t,rr,cc] = egval[i]

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
            self.status_map == 2, name="status == 2", visible=1,
            colormap="gray", blending='additive', opacity=1.0
            )
        viewer.add_image(
            self.status_map == 1, name="status == 1", visible=1,
            colormap="red", blending='additive', opacity=0.33
            )
        viewer.add_labels(
            self.labels_map, name="labels", visible=0,
            blending='additive'
            )
        viewer.add_image(
            self.tracks_norm_map, name="track norm", visible=0,
            colormap="turbo", blending='additive'
            )
        viewer.add_image(
            self.tracks_egval_map, name="track egval", visible=0,
            colormap="turbo", blending='additive'
            )

#%% Method : plot() -----------------------------------------------------------

    def plot(self):
        
        # rcParams
           
        mpl.rcParams.update({
        
        # Font
        "font.family"        : "Consolas",
        "axes.titlesize"     : 8,
        "axes.labelsize"     : 6,
        "xtick.labelsize"    : 5,
        "ytick.labelsize"    : 5,
        "legend.fontsize"    : 5,
    
        # Padding
        "axes.titlepad"      : 4,  
        "axes.labelpad"      : 2,  
        "xtick.major.pad"    : 2,  
        "ytick.major.pad"    : 2,          
        
        # Linewidth
        "axes.linewidth"     : 0.5,
        "xtick.major.width"  : 0.5, 
        "ytick.major.width"  : 0.5, 
        "xtick.major.size"   : 2,
        "ytick.major.size"   : 2,
        
        # Saving
        "savefig.dpi"         : 300,
        "savefig.transparent" : False,
        
        })
        
        # Initialize
        self.get_stats()
        nmax = self.feat_params["maxCorners"]
        
        # Create figure
    
        fig = plt.figure(figsize=(4, 4), layout="tight")
        gs = GridSpec(3, 3, figure=fig)
                
        # Track count ---------------------------------------------------------
    
        # Plot
        ax_cnt = fig.add_subplot(gs[0, 0]) 
        ax_cnt.plot(self.n, linewidth=0.5)
        ax_cnt.plot(self.lost_cum, linewidth=0.5)
        ax_cnt.axhline(y=nmax, linewidth=0.5, linestyle="--", color="k") 
    
        # Format
        ax_cnt.set_title("Track count/lost")
        ax_cnt.set_ylim(0, nmax * 1.1)
        ax_cnt.set_ylabel("Count")
        ax_cnt.set_xlabel("Timepoint")
        
        # Average eigenvalue --------------------------------------------------
    
        # Plot
        ax_err = fig.add_subplot(gs[0, 1]) 
        ax_err.plot(self.egval_avg, linewidth=0.5)
        # ax_err.axhline(y=nmax, linewidth=0.5, linestyle="--", color="k") 
    
        # Format
        ax_err.set_title("Avg. eigenvalue")
        ax_err.set_ylim(0, np.nanmax(self.egval_avg) * 1.1)
        ax_err.set_ylabel("Eigenvalue")
        ax_err.set_xlabel("Timepoint")
        
        # Average speed -------------------------------------------------------
    
        # Plot
        ax_nrm = fig.add_subplot(gs[1, 0]) 
        ax_nrm.plot(self.norm_avg, linewidth=0.5)
        
        # Format
        ax_nrm.set_title("Avg. speed")
        ax_nrm.set_ylim(0, np.nanmax(self.norm_avg) * 1.1)
        ax_nrm.set_ylabel("Speed (pix.tp-1)")
        ax_nrm.set_xlabel("Timepoint")
        
        # Average dy/dx -------------------------------------------------------
        
        # Plot
        ax_dyx = fig.add_subplot(gs[1, 1]) 
        ax_dyx.plot(self.dy_avg, linewidth=0.5, label="dy")
        ax_dyx.plot(self.dx_avg, linewidth=0.5, label="dx")
        ax_dyx.axhline(y=0, linewidth=0.5, linestyle="--", color="k") 
        
        # Format
        ax_dyx.set_title("Avg. dy/dx")
        ax_dyx.set_ylabel("dy/dx (pix.tp-1)")
        ax_dyx.set_xlabel("Timepoint")
        # ax_dyx.legend(loc="lower left")
        
        # Cumulative average dy/dx --------------------------------------------
    
        # Plot
        ax_cyx = fig.add_subplot(gs[1, 2]) 
        ax_cyx.plot(self.dy_avg_cum, linewidth=0.5, label="cum_dy")
        ax_cyx.plot(self.dx_avg_cum, linewidth=0.5, label="cum_dx")
        ax_cyx.axhline(y=0, linewidth=0.5, linestyle="--", color="k")
        
        # Format
        ax_cyx.set_title("Cum. avg. dy/dx")
        ax_cyx.set_ylabel("Cum. dy/dx (pix.tp-1)")
        ax_cyx.set_xlabel("Timepoint")
        # ax_cyx.legend(loc="lower left")


                
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    import time
    from skimage import io
    from pathlib import Path

    # Parameters
    
    replace = 1
    
    feat_params={
        "maxCorners"        : 2000,
        "qualityLevel"      : 1e-4,
        "minDistance"       : 3,
        "blockSize"         : 3,
        "useHarrisDetector" : True,
        "k"                 : 0.04,
        }
    
    flow_params={
        "winSize"           : (9, 9),
        "maxLevel"          : 3,
        "criteria"          : (5, 0.01),
        "minEigThreshold"   : 1e-2,
        }


    # Paths
    data_path = Path.cwd().parent / "_local" / "flow"
    arr_name = "GBE_eCad_40x.tif"
    msk_name = "GBE_eCad_40x_mask.tif"
    # arr_name = "DC_UtrCH_100x.tif"
    # msk_name = None
       
    # Load
    arr = io.imread(data_path / arr_name)
    if msk_name is not None:
        msk = io.imread(data_path / msk_name)
    else:
        msk = None
    
    # klt.process()
    t0 = time.time()
    print("klt.process() : ", end="", flush=False)
    klt = KLT(
        arr, msk=msk, replace=replace,
        feat_params=feat_params, 
        flow_params=flow_params,
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # klt.display()
    t0 = time.time()
    print("klt.display() : ", end="", flush=False)
    klt.display()
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # klt.plot()
    t0 = time.time()
    print("klt.plot() : ", end="", flush=False)
    klt.plot()
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")

    # Fetch attributes
    x, y, egval, status = klt.x, klt.y, klt.egval, klt.status
    n, lost, lost_cum = klt.n, klt.lost, klt.lost_cum
    normn, norm_avg, egval_avg = klt.norm, klt.norm_avg, klt.egval_avg
    dx, dy, dx_avg, dy_avg, dx_avg_cum, dy_avg_cum = (
        klt.dx, klt.dy, klt.dx_avg, klt.dy_avg, klt.dx_avg_cum, klt.dy_avg_cum
        )
