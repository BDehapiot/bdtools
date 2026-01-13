#%% Imports -------------------------------------------------------------------

import numpy as np

#%% Function: norm_gcn() ------------------------------------------------------

def norm_gcn(arr, sample_fraction=1, mask=None):
    
    """ 
    Global contrast normalization.

    Array(s) is(are) normalized by substracting the mean and dividing by the 
    standard deviation. Calculation can be restricted to a random fraction 
    (sample_fraction) and/or a given selection (mask). NaNs are ignored.
    
    Parameters
    ----------
    arr : ndarray or list of ndarrays (uint8, uint16 or float)
        Array(s) to be normalized.
        
    sample_fraction : float
        Fraction of the array to be considered for mean and standard deviation
        calculation. Must be between 0 and 1.
        
    mask : ndarray or list of ndarrays (bool)
        Selection of the array(s) to be considered for mean and standard deviation
        calculation. Mask(s) and array(s) must be of the same shape.
                
    Returns
    -------  
    arr : ndarray or list of ndarrays (float)
        Normalized array(s)
    
    """
    
    # Check inputs
    if sample_fraction < 0 or sample_fraction > 1:
        raise ValueError("sample_fraction should be float between 0 and 1")
    if isinstance(arr, np.ndarray):
        arr = arr.astype("float32").copy()
        if mask is not None and mask.shape != arr.shape:
            raise ValueError("array and mask should have the same shape")
    elif isinstance(arr, list):
        arr = [ar.astype("float32").copy() for ar in arr]  # Copy each array
        if mask is not None:
            for i in range(len(arr)):
                if mask[i].shape != arr[i].shape:
                    raise ValueError("array and mask should have the same shape")
    else:
        raise ValueError("array must be np.ndarray or list of np.ndarray")
    
    # Extract values
    if isinstance(arr, np.ndarray):
        val = arr.ravel()
        if mask is not None:
            val = val[mask.ravel()]
    elif isinstance(arr, list):
        val = np.concatenate([ar.ravel() for ar in arr])
        if mask is not None:
            val = val[np.concatenate([msk.ravel() for msk in mask])]
    if sample_fraction < 1:
        val = np.random.choice(val, size=int(val.size * sample_fraction))
    val = val[~np.isnan(val)]
    
    # Normalize
    if isinstance(arr, np.ndarray):
        arr -= np.mean(val)
        arr /= np.std(val) 
    elif isinstance(arr, list):
        for i in range(len(arr)): 
            arr[i] -= np.mean(val)
            arr[i] /= np.std(val) 
    
    return arr

#%% Function: norm_pct() ------------------------------------------------------

def norm_pct(
        arr,
        pct_low=0.01,
        pct_high=99.99,
        sample_fraction=1,
        mask=None
        ):
    
    """ 
    Percentile normalization.

    Array(s) is(are) normalized from 0 to 1 considering a range determined by 
    a low and a high percentile value (pct_low and pct_high). Out of range 
    values are clipped and NaNs are ignored. Calculation can be restricted to 
    a random fraction (sample_fraction) and/or a given selection (mask).

    Parameters
    ----------
    arr : ndarray or list of ndarrays (uint8, uint16 or float)
        Array(s) to be normalized.
        
    pct_low : float
        Percentile to determine the low value of the normalization range.
        pct_low must be >= 0 and < pct_high. If pct_low == 0, low value is 
        equal to the minimum value of the array. 

    pct_high : float
        Percentile to determine the high value of the normalization range.
        pct_high must be > pct_low and <= 100. If pct_high == 100, high value 
        is equal to the maximum value of the array.
        
    sample_fraction : float
        Fraction of the array to be considered for mean and standard deviation
        calculation. Must be between 0 and 1.
        
    mask : ndarray or list of ndarrays (bool)
        Selection of the array(s) to be considered for mean and standard deviation
        calculation. Mask(s) and array(s) must be of the same shape.
                
    Returns
    -------  
    arr : ndarray or list of ndarrays (float)
        Normalized array(s)
    
    """
    
    # Check inputs
    if pct_low < 0 or pct_low >= pct_high:
        raise ValueError("pct_low should be >= 0 and < pct_high")
    if pct_high > 100 or pct_high <= pct_low:
        raise ValueError("pct_high should be <= 100 and > pct_low")
    if sample_fraction < 0 or sample_fraction > 1:
        raise ValueError("sample_fraction should be float between 0 and 1")
    if isinstance(arr, np.ndarray):
        arr = arr.astype("float32").copy()
        if mask is not None and mask.shape != arr.shape:
            raise ValueError("array and mask should have the same shape")
    elif isinstance(arr, list):
        arr = [ar.astype("float32").copy() for ar in arr]  # Copy each array
        if mask is not None:
            for i in range(len(arr)):
                if mask[i].shape != arr[i].shape:
                    raise ValueError("array and mask should have the same shape")
    else:
        raise ValueError("array must be np.ndarray or list of np.ndarray")
    
    # Extract values
    if isinstance(arr, np.ndarray):
        val = arr.ravel()
        if mask is not None:
            val = val[mask.ravel()]
    elif isinstance(arr, list):
        val = np.concatenate([ar.ravel() for ar in arr])
        if mask is not None:
            val = val[np.concatenate([msk.ravel() for msk in mask])]
    if sample_fraction < 1:
        val = np.random.choice(val, size=int(val.size * sample_fraction))
    val = val[~np.isnan(val)]
    
    # Normalize
    if pct_low == 0: pLow = np.nanmin(arr)
    else: pLow = np.percentile(val, pct_low)
    if pct_high == 100: pHigh = np.nanmax(arr)
    else: pHigh = np.percentile(val, pct_high)
    if isinstance(arr, np.ndarray):
        np.clip(arr, pLow, pHigh, out=arr)
        arr -= pLow
        with np.errstate(invalid='ignore'):
            arr /= (pHigh - pLow)
    elif isinstance(arr, list):
        for i in range(len(arr)):
           np.clip(arr[i], pLow, pHigh, out=arr[i]) 
           arr[i] -= pLow
           with np.errstate(invalid='ignore'):
               arr[i] /= (pHigh - pLow)
    
    return arr