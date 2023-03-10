#%% Imports

import numpy as np
from scipy.stats import zscore
from scipy.signal import correlate
from joblib import Parallel, delayed 
from skimage.transform import rescale
from bdtools.nan import nanreplace, nanfilt

#%%

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
        
    # Run ---------------------------------------------------------------------

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

#%% Test

import time
from skimage import io 
from pathlib import Path

# -----------------------------------------------------------------------------

stack_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8.tif'
# mask_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8_mask-all.tif'
mask_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8_mask-proj.tif'

# -----------------------------------------------------------------------------

stack = io.imread(Path('../data/piv', stack_name))
mask = io.imread(Path('../data/piv', mask_name))
# mask = None

# -----------------------------------------------------------------------------

intSize = 36 # size of interrogation window (pixels)
srcSize = 72 # size of search window (pixels)
binning = 2 # reduce image size to speed up computation (1, 2, 4, 8...)
maskCutOff = 1 # mask out interrogation windows (???) 
parallel = True

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
    parallel=True
    )

end = time.time()
print(f'  {(end-start):5.3f} s') 

#%% 

# -----------------------------------------------------------------------------

vecU = output_dict['vecU']
vecV = output_dict['vecV']

# -----------------------------------------------------------------------------

outTresh = 1.5
kernel_size = (3,3,3)
kernel_shape = 'ellipsoid'
filt_method = 'mean'
iterations = 3
parallel = False

# -----------------------------------------------------------------------------
   
start = time.time()
print('filtpiv')

# Extract nanmask ()
nanmask = ~np.isnan(vecU)

# Replace outliers with NaNs
for u, v in zip(vecU, vecV):
    z_u = np.abs(zscore(u, axis=None, nan_policy='omit'))
    z_v = np.abs(zscore(v, axis=None, nan_policy='omit'))
    u[(z_u>outTresh) | (z_v>outTresh)] = np.nan
    v[(z_u>outTresh) | (z_v>outTresh)] = np.nan

vecU = nanreplace(
    vecU, 
    mask=nanmask,
    kernel_size=kernel_size,
    kernel_shape=kernel_shape,
    filt_method=filt_method, 
    iterations='inf',
    parallel=parallel,
    )

vecU = nanfilt(
    vecU, 
    mask=nanmask,
    kernel_size=kernel_size,
    kernel_shape=kernel_shape,
    filt_method=filt_method, 
    iterations=iterations,
    parallel=parallel,
    )

vecV = nanreplace(
    vecV, 
    mask=nanmask,
    kernel_size=kernel_size,
    kernel_shape=kernel_shape,
    filt_method=filt_method, 
    iterations='inf',
    parallel=parallel,
    )

vecV = nanfilt(
    vecV, 
    mask=nanmask,
    kernel_size=kernel_size,
    kernel_shape=kernel_shape,
    filt_method=filt_method, 
    iterations=iterations,
    parallel=parallel,
    )

norm = np.hypot(vecU, vecV)

end = time.time()
print(f'  {(end-start):5.3f} s') 

# -----------------------------------------------------------------------------

# start = time.time()
# print('filtpiv')

# outTresh = 1.5
    
# for t, (u, v) in enumerate(zip(vecU, vecV)):
    
#     nanmask = ~np.isnan(u)
#     norm = np.hypot(u, v)
#     z_u = np.abs(zscore(u, axis=None, nan_policy='omit'))
#     z_v = np.abs(zscore(v, axis=None, nan_policy='omit'))
#     u[(z_u>outTresh) | (z_v>outTresh)] = np.nan
#     v[(z_u>outTresh) | (z_v>outTresh)] = np.nan
    
#     u = nanreplace(
#         u, 
#         kernel_size=3, 
#         method='mean', 
#         mask=nanmask,
#         )

#     v = nanreplace(
#         v, 
#         kernel_size=3, 
#         method='mean', 
#         mask=nanmask,
#         )
    
#     vecU[t,...] = nanfilt(
#         u, 
#         kernel_size=3, 
#         method='mean', 
#         iterations=3,
#         )

#     vecV[t,...] = nanfilt(
#         v, 
#         kernel_size=3, 
#         method='mean', 
#         iterations=3,
#         )   

# end = time.time()
# print(f'  {(end-start):5.3f} s') 

#%% Save vecU & vecV 

io.imsave(
    Path('../data/piv', stack_name.replace('.tif', '_vecU.tif')),
    vecU.astype('float32'),
    check_contrast=False,
    )

io.imsave(
    Path('../data/piv', stack_name.replace('.tif', '_vecV.tif')),
    vecV.astype('float32'),
    check_contrast=False,
    )

io.imsave(
    Path('../data/piv', stack_name.replace('.tif', '_vecNorm.tif')),
    norm.astype('float32'),
    check_contrast=False,
    )

io.imsave(
    Path('../data/piv', stack_name.replace('.tif', '_vecMask.tif')),
    nanmask.astype('uint8')*255,
    check_contrast=False,
    )

#%% Display (vector field)

# import matplotlib.pyplot as plt

# # -----------------------------------------------------------------------------

# t = 40
# u = vecU[t,...]
# v = vecV[t,...]
# norm = np.hypot(u, v)
# fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
# ax.quiver(u, v, norm)
