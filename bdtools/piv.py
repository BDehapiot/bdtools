#%% Imports

import numpy as np
from scipy.stats import zscore
from scipy.signal import correlate
from joblib import Parallel, delayed 
from skimage.transform import rescale
from bdtools.nan import nanreplace, nanfilt

#%% Function: getpiv 

def getpiv(
        stack,
        intSize=32,
        srcSize=64,
        binning=1,
        mask=None,
        maskCutOff=1,
        parallel=True
        ):
    
    # Nested function ---------------------------------------------------------
    
    def _getpiv(img, ref, mask):
        
        # Create empty arrays
        vecU = np.full((intYn, intXn), np.nan)
        vecV = np.full((intYn, intXn), np.nan)
        
        for y, (iYi, sYi) in enumerate(zip(intYi, srcYi)):
            for x, (iXi, sXi) in enumerate(zip(intXi, srcXi)):
                
                # Extract mask int. window 
                maskWin = mask[iYi:iYi+intSize,iXi:iXi+intSize]
                
                if np.mean(maskWin) >= maskCutOff:
                
                    # Extract int. & src. window
                    intWin = ref[iYi:iYi+intSize,iXi:iXi+intSize]
                    srcWin = img[sYi:sYi+srcSize,sXi:sXi+srcSize]           
        
                    # Compute 2D correlation
                    corr2D = correlate(
                        srcWin - np.mean(srcWin), 
                        intWin - np.mean(intWin),
                        method='fft'
                        )
                    
                    # Find max corr. and infer uv components
                    y_max, x_max = np.unravel_index(corr2D.argmax(), corr2D.shape)            
                    vecU[y,x] = x_max-(intSize-1)-(srcSize//2-intSize//2)
                    vecV[y,x] = y_max-(intSize-1)-(srcSize//2-intSize//2)
                    
                else:
                    
                    vecU[y,x] = np.nan
                    vecV[y,x] = np.nan
        
        return vecU, vecV
        
    # Execute -----------------------------------------------------------------

    # Mask operations
    if mask is None:
        mask = np.full_like(stack, True, dtype=bool)
    else:
        mask = mask.astype(bool)
        if mask.ndim == 2: 
            mask = np.expand_dims(mask, 0)
            mask = np.repeat(mask, stack.shape[0], axis=0)
            
    # Adjust parameters/data acc. to binning
    if binning > 1:
    
        # Parameters
        intSize = intSize//binning
        srcSize = srcSize//binning 
        if intSize % 2 != 0:
            intSize += intSize % 2
            print(f'interrogation window size adjusted to {intSize*binning}')
        if srcSize % 2 != 0:
            srcSize += srcSize % 2
            print(f'search window size adjusted to {srcSize*binning}')  
    
        # Data
        stack = rescale(stack, (1, 1/binning, 1/binning), preserve_range=True)
        if mask.ndim == 2: mask = rescale(mask, (1/binning, 1/binning), order=0)
        if mask.ndim == 3: mask = rescale(mask, (1, 1/binning, 1/binning), order=0)
    
    # Define src. pad
    srcPad = (srcSize-intSize)//2
    
    # Count number of int. window
    intYn = (stack.shape[1]-srcPad*2)//intSize
    intXn = (stack.shape[2]-srcPad*2)//intSize
    
    # Setup int. & src. window coordinates
    intYi = np.arange(
        (stack.shape[1]-intYn*intSize)//2, 
        (stack.shape[1]-intYn*intSize)//2 + intYn*intSize, 
        intSize,
        )
    intXi = np.arange(
        (stack.shape[2]-intXn*intSize)//2, 
        (stack.shape[2]-intXn*intSize)//2 + intXn*intSize, 
        intSize,
        )
    srcYi = intYi - srcPad
    srcXi = intXi - srcPad 

    # _getPIV
    if parallel:
        
        output_list = Parallel(n_jobs=-1)(
        delayed(_getpiv)(
            stack[t,...],
            stack[t-1,...], 
            mask[t,...],
            )
        for t in range(1, stack.shape[0])
        )
        
    else:
        
        output_list = [_getpiv(
            stack[t,...],
            stack[t-1,...],
            mask[t,...],
            )
        for t in range(1, stack.shape[0])
        ]
        
    # Fill output dictionary    
    output_dict = {
    
    # Parameters
    'intSize': intSize,
    'srcSize': srcSize,
    'binning': binning,
    'maskCutOff': maskCutOff,
    
    # Data
    'intYi': intYi,
    'intXi': intXi,
    'vecU': np.stack([data[0] for data in output_list], axis=0),
    'vecV': np.stack([data[1] for data in output_list], axis=0),
    'mask': mask

    }
        
    return output_dict

#%% Function: filtpiv 

def filtpiv(
        output_dict,
        outlier_cutoff=1.5,
        spatial_smooth=3,
        temporal_smooth=3,
        iterations_smooth=1,
        parallel=False,
        ):
    
    # Execute -----------------------------------------------------------------

    # Extract data & parameters
    vecU = output_dict['vecU']
    vecV = output_dict['vecV']
    kernel_size = (temporal_smooth,spatial_smooth,spatial_smooth)

    # Extract nanmask 
    nanmask = ~np.isnan(vecU)

    # Replace outliers with NaNs
    for u, v in zip(vecU, vecV):
        z_u = np.abs(zscore(u, axis=None, nan_policy='omit'))
        z_v = np.abs(zscore(v, axis=None, nan_policy='omit'))
        u[(z_u>outlier_cutoff) | (z_v>outlier_cutoff)] = np.nan
        v[(z_u>outlier_cutoff) | (z_v>outlier_cutoff)] = np.nan

    vecU = nanreplace(
        vecU, 
        mask=nanmask,
        kernel_size=kernel_size,
        kernel_shape='ellipsoid',
        filt_method='mean', 
        iterations='inf',
        parallel=parallel,
        )

    vecU = nanfilt(
        vecU, 
        mask=nanmask,
        kernel_size=kernel_size,
        kernel_shape='ellipsoid',
        filt_method='mean', 
        iterations=iterations_smooth,
        parallel=parallel,
        )

    vecV = nanreplace(
        vecV, 
        mask=nanmask,
        kernel_size=kernel_size,
        kernel_shape='ellipsoid',
        filt_method='mean', 
        iterations='inf',
        parallel=parallel,
        )

    vecV = nanfilt(
        vecV, 
        mask=nanmask,
        kernel_size=kernel_size,
        kernel_shape='ellipsoid',
        filt_method='mean', 
        iterations=iterations_smooth,
        parallel=parallel,
        )
    
    # Updating output dictionary 
    output_dict.update({'vecU': vecU})
    output_dict.update({'vecV': vecV})
    output_dict.update({'outlier_cutoff': outlier_cutoff})
    output_dict.update({'spatial_smooth': spatial_smooth})
    output_dict.update({'temporal_smooth': temporal_smooth})
    output_dict.update({'iterations_smooth': iterations_smooth})
    
    return output_dict
        
#%% Test

import time
from skimage import io 
from pathlib import Path

# -----------------------------------------------------------------------------

stack_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8.tif'
# mask_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8_mask-all.tif'
mask_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8_mask-proj.tif'
stack = io.imread(Path('../data/piv', stack_name))
mask = io.imread(Path('../data/piv', mask_name))
# mask = None

# -----------------------------------------------------------------------------

# stack_name = '18-07-03_100x_UtrCH_Ctrl_a2_uint8.tif'
# stack = io.imread(Path('../data/piv', stack_name))
# mask = None

# -----------------------------------------------------------------------------

# getpiv parameters
intSize = 36 # size of interrogation window (pixels)
srcSize = 72 # size of search window (pixels)
binning = 1 # reduce image size to speed up computation (1, 2, 4, 8...)
maskCutOff = 1 # mask out interrogation windows (???) 

# filtpiv parameters
outlier_cutoff = 1.5 # remove outliers from vector field
spatial_smooth = 3 # spatial smoothening of vector field
temporal_smooth = 3 # temporal smoothening of vector field
iterations_smooth = 3 # iterations of smoothening process

# -----------------------------------------------------------------------------

start = time.time()
print('getpiv')
        
output_dict = getpiv(
    stack,
    intSize=intSize,
    srcSize=srcSize,
    binning=binning,
    mask=mask,
    maskCutOff=maskCutOff,
    parallel=True,
    )

end = time.time()
print(f'  {(end-start):5.3f} s') 

# -----------------------------------------------------------------------------

start = time.time()
print('filtpiv')
        
output_dict = filtpiv(
    output_dict,
    outlier_cutoff=outlier_cutoff,
    spatial_smooth=spatial_smooth,
    temporal_smooth=temporal_smooth,
    iterations_smooth=iterations_smooth,
    parallel=False,
    )

end = time.time()
print(f'  {(end-start):5.3f} s') 

#%% 

from skimage.morphology import dilation
from skimage.morphology import disk

intYi = output_dict['intYi']
intXi = output_dict['intXi']

vecROI = np.zeros_like(stack[0,...])
for y, iYi in enumerate(intYi):
    for x, iXi in enumerate(intXi):
        
        if iYi == intYi[-1] and iXi == intXi[-1]:
            color = 255
        else:
            color = 128

        vecROI[iYi+intSize//2,iXi+intSize//2] = color
           
vecROI = dilation(vecROI, footprint=disk(3))

#%% dispiv

import matplotlib.pyplot as plt
from matplotlib import rcParams

# Parameters ------------------------------------------------------------------

t = 45
axes = True
colorbar = True
background_image = False
title = 'Flow'
pixel_size = 0.2
space_unit = 'µm'
time_interval = 1/3
time_unit = 'min'
renference_vector = 5 # set to 0 to deactivate
xTick_interval = 20
yTick_interval = 20
cmap = 'gray'

# Advanced parameters ---------------------------------------------------------

dpi = 300
plotSize = 0.6
xTick_min = 0
xTick_max = 'auto'
yTick_min = 0
yTick_max = 'auto'
stack.shape[2] * pixel_size
stack.shape[1] * pixel_size

# rcParams --------------------------------------------------------------------

rcParams_keys = rcParams
rcParams['axes.linewidth'] =0.5
rcParams['axes.labelsize'] = 50 # ????
rcParams['contour.linewidth'] = 0.5 # ????
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5
rcParams['xtick.minor.visible'] = True
rcParams['ytick.minor.visible'] = True
rcParams['xtick.labelsize'] = 6
rcParams['ytick.labelsize'] = 6
rcParams['figure.facecolor'] = 'white'
rcParams['axes.facecolor'] = 'white'

# Initialize ------------------------------------------------------------------

# Extract data
vecU = output_dict['vecU']
vecV = output_dict['vecV']
intSize = output_dict['intSize']
intYi = output_dict['intYi']
intXi = output_dict['intXi']

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

# -----------------------------------------------------------------------------

# Get vector xy coordinates
xCoords, yCoords = np.meshgrid(intXi + intSize // 2, intYi + intSize // 2)

# Get vector norm
norm = np.hypot(vecU, vecV)

# Plot quiver
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi) 
plt.ylim([0, height * pixel_size])
plt.xlim([0, width * pixel_size])

plot = ax.quiver(
    xCoords * pixel_size,
    yCoords * pixel_size,
    vecU[t,...] * pixel_size / time_interval,
    vecV[t,...] * pixel_size / time_interval * -1,
    norm[t,...] * pixel_size / time_interval, 
    cmap=cmap,
    pivot='tail'
    )

ax.invert_yaxis()

# -----------------------------------------------------------------------------

if axes:
    
    fig.subplots_adjust(top=top, bottom=bottom, right=right, left=left)
    if xTick_max == 'auto': xTick_max = width * pixel_size
    if yTick_max == 'auto': yTick_max = height * pixel_size
    ax.set_xticks(np.arange(xTick_min, xTick_max + 1, xTick_interval))
    ax.set_yticks(np.arange(yTick_min, yTick_max + 1, yTick_interval))
    ax.set_xlabel(f'x position ({space_unit})')    
    ax.set_ylabel(f'y position ({space_unit})')
     
else:
    
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
    ax.set_axis_off()

# -----------------------------------------------------------------------------
    
if background_image:
    ax.imshow(vecROI, extent=[0, width * pixel_size, 0, height * pixel_size], cmap='gray')

# -----------------------------------------------------------------------------
    
if renference_vector:
    ax.quiverkey(
        plot, 0, 1.075, renference_vector, 
        label=f'{renference_vector} {space_unit}.{time_unit}-1', 
        labelpos='N', labelsep=0.075,
        coordinates='axes',
        fontproperties={'size': 6}
        )

# -----------------------------------------------------------------------------

if title is not None and axes:
    plt.title(title, fontsize=10, pad=10)

# -----------------------------------------------------------------------------

if colorbar and axes:
    cbax = fig.add_axes([right + 0.025, bottom, 0.025, plotSize])
    fig.colorbar(plot, orientation='vertical', cax=cbax)
    cbax.set_ylabel(f'{space_unit}.{time_unit}-1', fontsize=6)
    cbax.tick_params(axis='y', labelsize=6)

# -----------------------------------------------------------------------------

# save figure
plt.savefig("output.tif", dpi=dpi)

# # -----------------------------------------------------------------------------

# fig.canvas.draw()
# w, h = fig.canvas.get_width_height()
# buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
# buf = buf.reshape((h, w, 3))
# vecArrows = buf[:,:,0]

# # -----------------------------------------------------------------------------

# from skimage.morphology import dilation
# from skimage.morphology import disk

# intYi = output_dict['intYi']
# intXi = output_dict['intXi']

# vecROI = np.zeros_like(stack[0,...])
# for y, iYi in enumerate(intYi):
#     for x, iXi in enumerate(intXi):
        
#         vecROI[iYi+intSize//2,iXi+intSize//2] = 255
        
# vecROI = dilation(vecROI, footprint=disk(3))

# # -----------------------------------------------------------------------------

#%% Save vecU & vecV 

# # Extract data
# vecU = output_dict['vecU']
# vecV = output_dict['vecV']

# io.imsave(
#     Path('../data/piv', stack_name.replace('.tif', '_vecU.tif')),
#     vecU.astype('float32'),
#     check_contrast=False,
#     )

# io.imsave(
#     Path('../data/piv', stack_name.replace('.tif', '_vecV.tif')),
#     vecV.astype('float32'),
#     check_contrast=False,
#     )