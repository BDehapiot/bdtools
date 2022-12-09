#%% Imports

import numpy as np
from skimage.exposure import rescale_intensity

#%% Functions

def ranged_conversion(
        img, 
        intensity_range=(5,95), 
        spread=1.2, 
        dtype='float'
        ):

    """ 
    Ranged image dtype conversion. 
    
    Parameters
    ----------
    img : ndarray
        Image to be converted.
        
    intensity_range : tuple of float
        Range of input intensities to be considered (percentiles).
        
    spread : float
        values < 1 reduce range of input intensities
        values > 1 increase range of input intensities
    
    Returns
    -------  
    img : ndarray
        Converted image.
    
    """    

    # Define input min and max intensity
    int_min = np.percentile(img, intensity_range[0])
    int_max = np.percentile(img, intensity_range[1])
    int_range = int_max-int_min
    int_min -= int_range - (int_range*spread)/2
    int_max += int_range - (int_range*spread)/2
    if int_max > np.max(img): int_max = np.max(img)
    
    # Rescale image
    img = rescale_intensity(img, 
        in_range=(int_min, int_max),
        out_range=dtype
        )
    
    return img