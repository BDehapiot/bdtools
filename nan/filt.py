#%% Imports

import warnings
import numpy as np

#%% Function

def nanfilt(img, kernel_size, method):

    """ 
    Filter image ignoring NaN
    
    Parameters
    ----------
    img : ndarray
        Image to be converted.
        
    percent_low : float
        Percentile to discard low values.
        Between 0 and 100 inclusive.
        
    percent_high : float
        Percentile to discard high values.
        Between 0 and 100 inclusive.
    
    Returns
    -------  
    img : ndarray
        Converted image.
    
    """

    # Warnings
    warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
    warnings.filterwarnings(action="ignore", message="Mean of empty slice")
    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd and > 1")

    # Add one dimension (if ndim = 2)
    ndim = img.ndim
    if ndim == 2:
        img = img.reshape((1, img.shape[0], img.shape[1]))

    # Pad img border with NaNs
    pad = (kernel_size - 1) // 2
    img_pad = np.pad(
        img,
        pad_width=((0, 0), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=np.nan,
    )

    # Find non-NaNs coordinates
    idx = rwhere(~np.isnan(img), True)
    idx_t = idx[0].squeeze().astype("int")
    idx_y = idx[1].squeeze().astype("int")
    idx_x = idx[2].squeeze().astype("int")

    if idx:

        # Define kernels
        kernel_t = np.zeros([kernel_size, kernel_size], dtype=int)
        kernel_x, kernel_y = np.meshgrid(
            np.arange((-kernel_size // 2) + 1, (kernel_size // 2) + 1),
            np.arange((-kernel_size // 2) + 1, (kernel_size // 2) + 1),
        )
        idx_tt = idx_t[:, None, None] + kernel_t
        idx_xx = idx_x[:, None, None] + kernel_x
        idx_yy = idx_y[:, None, None] + kernel_y

        # Filter image
        img_nanfilt = img.copy()
        functions = {
            'mean': np.nanmean, 
            'median': np.nanmedian, 
            'std': np.nanstd, 
            'max': np.nanmax, 
            'min': np.nanmin
            } 
        all_kernels = img_pad[idx_tt, idx_yy + pad, idx_xx + pad]
        all_kernels = functions[method](all_kernels, axis=(1, 2))
        for i in range(len(all_kernels)):
            img_nanfilt[idx_t[i], idx_y[i], idx_x[i]] = all_kernels[i]

    # Remove one dimension (if ndim = 2)
    if ndim == 2:
        img = img.squeeze()
        img_nanfilt = img_nanfilt.squeeze()

    return img_nanfilt

