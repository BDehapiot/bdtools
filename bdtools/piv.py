#%% Imports -------------------------------------------------------------------

import numpy as np
from scipy.stats import zscore
from scipy.signal import correlate
from joblib import Parallel, delayed
from skimage.transform import rescale
from bdtools.nan import nanfilt, nanreplace

#%% Function: get_piv ---------------------------------------------------------

def get_piv(
        arr,
        intSize=32, srcSize=64, binning=1,
        mask=None, maskCutOff=1,
        parallel=True
        ):
    
    # Nested function ---------------------------------------------------------
    
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
                        method='fft'
                        )
                    
                    # Find max corr. and infer uv components
                    y_max, x_max = np.unravel_index(corr2D.argmax(), corr2D.shape)            
                    vecU[y, x] = x_max - (intSize - 1) - (srcSize // 2 - intSize // 2)
                    vecV[y, x] = y_max - (intSize - 1) - (srcSize // 2 - intSize // 2)
                    
                else:
                    
                    vecU[y, x] = np.nan
                    vecV[y, x] = np.nan
        
        return vecU, vecV
        
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
            print(f'interrogation window size adjusted to {intSize * binning}')
        if srcSize % 2 != 0:
            srcSize += srcSize % 2
            print(f'search window size adjusted to {srcSize * binning}')  
    
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
        'intSize': intSize * binning,
        'srcSize': srcSize * binning,
        'binning': binning,
        'maskCutOff': maskCutOff,
        
        # Data
        'intYi': intYi * binning,
        'intXi': intXi * binning,
        'vecU': np.stack([data[0] for data in output_list], axis=0) * binning,
        'vecV': np.stack([data[1] for data in output_list], axis=0) * binning,
        'mask': mask

    }
        
    return output_dict

#%% Function: filt_piv --------------------------------------------------------

def filt_piv(
        output_dict,
        outlier_cutoff=1.5,
        spatial_smooth=3, temporal_smooth=1, iterations_smooth=1,
        parallel=False,
        ):
    
    # Execute -----------------------------------------------------------------

    # Extract data & parameters
    vecU = output_dict['vecU']
    vecV = output_dict['vecV']
    kernel_size = (temporal_smooth, spatial_smooth, spatial_smooth)

    # Extract nanmask 
    nanmask = ~np.isnan(vecU)

    # Replace outliers with NaNs
    for u, v in zip(vecU, vecV):
        z_u = np.abs(zscore(u, axis=None, nan_policy='omit'))
        z_v = np.abs(zscore(v, axis=None, nan_policy='omit'))
        u[(z_u > outlier_cutoff) | (z_v > outlier_cutoff)] = np.nan
        v[(z_u > outlier_cutoff) | (z_v > outlier_cutoff)] = np.nan

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
