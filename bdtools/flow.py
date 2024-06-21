#%% Imports -------------------------------------------------------------------

import shutil
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed
from bdtools.nan import nanfilt, nanreplace

# Skimage
from skimage.transform import rescale

# Scipy
from scipy.stats import zscore
from scipy.signal import correlate

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

#%% Function: get_piv ---------------------------------------------------------

def get_piv(
        arr,
        intSize=32, srcSize=64, binning=1,
        mask=None, maskCutOff=1,
        parallel=True
        ):
    
    # Nested function(s) ------------------------------------------------------
    
    def _get_piv(img, ref, mask):
        
        # Create empty arrays
        vecU = np.full((intYn, intXn), np.nan)
        vecV = np.full((intYn, intXn), np.nan)
        
        for y, (iYi, sYi) in enumerate(zip(intYi, srcYi)):
            for x, (iXi, sXi) in enumerate(zip(intXi, srcXi)):
                
                # Extract mask int. window 
                maskWin = mask[iYi:iYi + intSize, iXi:iXi + intSize]
                
                if np.mean(maskWin) >= maskCutOff:
                
                    # Extract int. & src. window
                    intWin = ref[iYi:iYi + intSize, iXi:iXi + intSize]
                    srcWin = img[sYi:sYi + srcSize, sXi:sXi + srcSize]           
        
                    # Compute 2D correlation
                    corr2D = correlate(
                        srcWin - np.mean(srcWin), 
                        intWin - np.mean(intWin),
                        method="fft"
                        )
                    
                    # Find max corr. and infer uv components
                    y_max, x_max = np.unravel_index(corr2D.argmax(), corr2D.shape)            
                    vecU[y, x] = x_max - (intSize - 1) - (srcSize // 2 - intSize // 2)
                    vecV[y, x] = y_max - (intSize - 1) - (srcSize // 2 - intSize // 2)
                    
                else:
                    
                    vecU[y, x] = np.nan
                    vecV[y, x] = np.nan
        
        return vecU, vecV, corr2D
        
    # Execute -----------------------------------------------------------------

    # Mask operations
    if mask is None:
        mask = np.full_like(arr, True, dtype=bool)
    else:
        mask = mask.astype(bool)
        if mask.ndim == 2: 
            mask = np.expand_dims(mask, 0)
            mask = np.repeat(mask, arr.shape[0], axis=0)
            
    # Adjust parameters/data acc. to binning
    if binning > 1:
    
        # Parameters
        intSize = intSize // binning
        srcSize = srcSize // binning 
        if intSize % 2 != 0:
            intSize += intSize % 2
            print(f"interrogation window size adjusted to {intSize * binning}")
        if srcSize % 2 != 0:
            srcSize += srcSize % 2
            print(f"search window size adjusted to {srcSize * binning}")  
    
        # Data
        arr = rescale(arr, (1, 1 / binning, 1 / binning), preserve_range=True)
        if mask.ndim == 2: 
            mask = rescale(mask, (1 / binning, 1 / binning), order=0)
        if mask.ndim == 3: 
            mask = rescale(mask, (1, 1 / binning, 1 / binning), order=0)
    
    # Define src. pad
    srcPad = (srcSize - intSize) // 2
    
    # Count number of int. window
    intYn = (arr.shape[1] - srcPad*2) // intSize
    intXn = (arr.shape[2] - srcPad*2) // intSize
    
    # Setup int. & src. window coordinates
    intYi = np.arange(
        (arr.shape[1] - intYn*intSize) // 2, 
        (arr.shape[1] - intYn*intSize) // 2 + intYn*intSize, 
        intSize,
        )
    intXi = np.arange(
        (arr.shape[2] - intXn*intSize) // 2, 
        (arr.shape[2] - intXn*intSize) // 2 + intXn*intSize, 
        intSize,
        )
    srcYi = intYi - srcPad
    srcXi = intXi - srcPad 

    # _getPIV
    if parallel:
        
        output_list = Parallel(n_jobs=-1)(
            delayed(_get_piv)(
                arr[t, ...], arr[t - 1, ...], mask[t, ...])
            for t in range(1, arr.shape[0])
            )
                
    else:
        
        output_list = [_get_piv(
            arr[t, ...], arr[t - 1, ...], mask[t, ...])
            for t in range(1, arr.shape[0])
            ]
        
    # Fill output dictionary    
    output_dict = {
    
        # Parameters
        "intSize" : intSize * binning,
        "srcSize" : srcSize * binning,
        "binning" : binning,
        "maskCutOff" : maskCutOff,
        
        # Data
        "intYi" : intYi * binning,
        "intXi" : intXi * binning,
        "vecU" : np.stack([data[0] for data in output_list], axis=0) * binning,
        "vecV" : np.stack([data[1] for data in output_list], axis=0) * binning,
        "corr2D" : np.stack([data[2] for data in output_list], axis=0), 
        "mask" : mask

    }
        
    return output_dict

#%% Function: filt_piv --------------------------------------------------------

def filt_piv(
        output_dict,
        outlier_cutoff=1.5,
        spatial_smooth=3, temporal_smooth=1, iterations_smooth=1,
        parallel=False,
        ):
    
    # Nested function(s) ------------------------------------------------------
    
    def smooth_piv(vec):
        
        vec = nanreplace(
            vec, 
            mask=nanmask,
            kernel_size=kernel_size,
            kernel_shape="ellipsoid",
            filt_method="mean", 
            iterations="inf",
            parallel=parallel,
            )

        vec = nanfilt(
            vec, 
            mask=nanmask,
            kernel_size=kernel_size,
            kernel_shape="ellipsoid",
            filt_method="mean", 
            iterations=iterations_smooth,
            parallel=parallel,
            )
        
        return vec
    
    # Execute -----------------------------------------------------------------

    # Extract data & parameters
    vecU = output_dict["vecU"]
    vecV = output_dict["vecV"]
    kernel_size = (temporal_smooth, spatial_smooth, spatial_smooth)

    # Extract nanmask 
    nanmask = ~np.isnan(vecU)

    # Replace outliers with NaNs
    for u, v in zip(vecU, vecV):
        z_u = np.abs(zscore(u, axis=None, nan_policy="omit"))
        z_v = np.abs(zscore(v, axis=None, nan_policy="omit"))
        u[(z_u > outlier_cutoff) | (z_v > outlier_cutoff)] = np.nan
        v[(z_u > outlier_cutoff) | (z_v > outlier_cutoff)] = np.nan

    # Smooth data
    vecU = smooth_piv(vecU)
    vecV = smooth_piv(vecV)
    
    # Updating output dictionary 
    output_dict.update({"vecU": vecU})
    output_dict.update({"vecV": vecV})
    output_dict.update({"outlier_cutoff": outlier_cutoff})
    output_dict.update({"spatial_smooth": spatial_smooth})
    output_dict.update({"temporal_smooth": temporal_smooth})
    output_dict.update({"iterations_smooth": iterations_smooth})
    
    return output_dict

#%% Function: plot_piv --------------------------------------------------------

def plot_piv(
        
        stack, outputs,
        
        # Main 
        axes = True,
        colorbar = True,
        background_image = True,
        cmap = "viridis",
        
        # Appearance
        dpi = 300,
        plotSize = 0.6,
        linewidth = 0.5,
        fontSize = 8,
        title = "flow",
        
        # Units
        pixel_size = 0.08,
        space_unit = "µm",
        time_interval = 1 / 12,
        time_unit = "min",
        
        # Axes
        xTick_min = 0,
        xTick_max = "auto",
        xTick_interval = 50,
        yTick_min = 0,
        yTick_max = "auto",
        yTick_interval = 50,
        reference_vector = 1,
        
        ):
    
    # rcParams ----------------------------------------------------------------

    rcParams["axes.linewidth"] = linewidth
    rcParams["axes.titlesize"] = fontSize * 1.5
    rcParams["axes.labelsize"] = fontSize
    rcParams["xtick.major.width"] = linewidth
    rcParams["ytick.major.width"] = linewidth
    rcParams["xtick.minor.visible"] = True
    rcParams["ytick.minor.visible"] = True
    rcParams["xtick.labelsize"] = fontSize * 0.75
    rcParams["ytick.labelsize"] = fontSize * 0.75
    rcParams["figure.facecolor"] = 'white'
    rcParams["axes.facecolor"] = 'white'
    
    # Nested function(s) ------------------------------------------------------
           
    def _plot_piv(t):
    
        # Plot quiver
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi) 
        cMax = np.nanmax(norm)
        
        plot = ax.quiver(
            
            # Data
            xCoords * pixel_size,
            yCoords * pixel_size,
            vecU[t,...] * pixel_size / time_interval,
            vecV[t,...] * pixel_size / time_interval * -1,
            norm[t,...] * pixel_size / time_interval,
            
            # Appearance
            cmap=cmap,
            pivot='mid',
            scale=50,
            scale_units="width",
            clim=(0, cMax),
            width=0.01,
            headwidth=3,
            headlength=5,
            headaxislength=4.5,
            minshaft=1,
            minlength=1,

            )
    
        # Set xy axes limits
        plt.ylim([0, height * pixel_size])
        plt.xlim([0, width * pixel_size])
        ax.invert_yaxis()
        
        # Axes
        if axes:
            fig.subplots_adjust(top=top, bottom=bottom, right=right, left=left)
            if xTick_max == 'auto': 
                xTick_maxx = width * pixel_size
            if yTick_max == 'auto': 
                yTick_maxx = height * pixel_size
            ax.set_xticks(np.arange(xTick_min, xTick_maxx + 1, xTick_interval))
            ax.set_yticks(np.arange(yTick_min, yTick_maxx + 1, yTick_interval))
            ax.set_xlabel(f'x position ({space_unit})')    
            ax.set_ylabel(f'y position ({space_unit})')
        else:
            fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
            ax.set_axis_off()

        # Background image     
        if background_image:
            ax.imshow(
                np.flip(stack[t, ...], axis=0), 
                extent=[0, width * pixel_size, 0, height * pixel_size], 
                cmap='gray'
                )

        # Reference vector
        if reference_vector:
            font_props = FontProperties(size=fontSize * 0.75)
            ax.quiverkey(
                plot, 0, 1.075, reference_vector, 
                label=f'{reference_vector} {space_unit}.{time_unit}-1', 
                labelpos='N', labelsep=0.075,
                coordinates='axes',
                fontproperties=font_props,
                )

        # Title
        if title is not None and axes:
            plt.title(title, pad=10)
            
        # Colorbar
        if colorbar and axes:
            cbax = fig.add_axes([right + 0.025, bottom, 0.025, plotSize])
            fig.colorbar(plot, orientation='vertical', cax=cbax)
            cbax.set_ylabel(f'{space_unit}.{time_unit}-1')
            
        # Save plot and close figure
        plt.savefig(save_path / f"plot_{t:03d}.tif", dpi=dpi)
        plt.close(fig)

    # Initialize --------------------------------------------------------------

    # Extract data
    vecU = outputs["vecU"]
    vecV = outputs["vecV"]
    intSize = outputs["intSize"]
    intYi = outputs["intYi"]
    intXi = outputs["intXi"]

    # Set figure layout
    width = stack.shape[2]
    height = stack.shape[1]
    fig_width = width / dpi
    fig_height = height / dpi
    if axes:
        fig_width /= plotSize
        fig_height /= plotSize
        bottom = (1 - plotSize) * 0.5
        top = bottom + plotSize
        left = (1 - plotSize) * 0.5
        right = left + plotSize
        
    # Get vector xy coordinates
    xCoords, yCoords = np.meshgrid(intXi + intSize // 2, intYi + intSize // 2)

    # Get vector norm
    norm = np.hypot(vecU, vecV)

    # Execute -----------------------------------------------------------------

    # Plot & save
    save_path = Path(Path.cwd(), "tmp")
    save_path.mkdir(exist_ok=True)   
    Parallel(n_jobs=-1)(
        delayed(_plot_piv)(t)
        for t in range(vecU.shape[0])
        )
    
    #
    plot = []
    for path in list(save_path.glob("*.tif")):
        plot.append(io.imread(path))
    plot = np.stack(plot)
    shutil.rmtree(save_path)
        
    return plot
    
#%%

import time

# -----------------------------------------------------------------------------

# Paths
ROOT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_PATH / 'tests' / 'data' / 'flow'

# Read
# stack = io.imread(DATA_PATH / "GBE_eCad_40x.tif")
# mask = io.imread(DATA_PATH / "GBE_eCad_40x_mask.tif")
stack = io.imread(DATA_PATH / "DC_UtrCH_100x.tif")
mask = None

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
        spatial_smooth=5, temporal_smooth=5, iterations_smooth=1,
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

# -----------------------------------------------------------------------------

import napari
viewer = napari.Viewer()
viewer.add_image(plot)
