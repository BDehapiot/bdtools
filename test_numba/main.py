#%%

import time
import numpy as np
from numba import jit, prange
from skimage import io 
from pathlib import Path

# -----------------------------------------------------------------------------

stack_name = 'noise_3d_256.tif'

# -----------------------------------------------------------------------------

img1 = io.imread(Path('../data', stack_name)).astype(float)
img2 = io.imread(Path('../data', stack_name)).astype(float)
kernel_size = 5

#%%

# Pad img
pad = kernel_size//2
img1_pad = np.pad(img1.astype(float), pad, constant_values=np.nan)
img1_filt = img1_pad.copy()

# -----------------------------------------------------------------------------

start = time.time()
print('python')

# Filt img
for z in range(img1.shape[0]):
    for y in range(img1.shape[1]):
        for x in range(img1.shape[2]):
    
            img1_filt[z,y,x] = np.nanmean(
                img1_pad[z:z+kernel_size,y:y+kernel_size,x:x+kernel_size]
                )
                
end = time.time()
print(f'{(end-start):5.3f} s') 

# -----------------------------------------------------------------------------

io.imsave(
    Path('../data', stack_name.replace('.tif', '_filt1.tif')),
    img1_filt.astype('float32'),
    check_contrast=False,
    )

#%%
   
# Pad img
pad = kernel_size//2
img2_pad = np.pad(img2.astype(float), pad, constant_values=np.nan)
img2_filt = img2_pad.copy()

# -----------------------------------------------------------------------------

start = time.time()
print('numba')

@jit(nopython=True, parallel=False)
def imfilt(img2_pad, img2_filt, kernel_size):

    for z in range(img2.shape[0]):
        for y in range(img2.shape[1]):
            for x in range(img2.shape[2]):
        
                img2_filt[z,y,x] = np.nanmean(
                    img2_pad[z:z+kernel_size,y:y+kernel_size,x:x+kernel_size]
                    )
        
    return img2_filt

# Filt img
img2_filt = imfilt(img2_pad, img2_filt, kernel_size)
end = time.time()
print(f'{(end-start):5.3f} s') 

# -----------------------------------------------------------------------------

io.imsave(
    Path('../data', stack_name.replace('.tif', '_filt2.tif')),
    img2_filt.astype('float32'),
    check_contrast=False,
    )

#%%

print(np.array_equal(img1_filt, img2_filt, equal_nan=True))