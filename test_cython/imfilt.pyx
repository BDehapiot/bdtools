import numpy as np
cimport numpy as cnp

cpdef cnp.ndarray[cnp.float64_t, ndim=2] imfilt(
    cnp.ndarray[cnp.float64_t, ndim=2] img,
    int kernel_size
    ):

    # Pad img
    cdef int pad = kernel_size//2
    cdef cnp.ndarray[cnp.float64_t, ndim=2] img_pad = img.copy()   
    img_pad = np.pad(img.astype(float), pad, constant_values=np.nan)
    
    # Filt img
    cdef int x, y
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img[y,x] = np.nanmean(
                img_pad[y:y+kernel_size,x:x+kernel_size]
                )
        
    return img