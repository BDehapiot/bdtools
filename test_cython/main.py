#%%

import time
import numpy as np
from skimage import io 
from pathlib import Path

# -----------------------------------------------------------------------------

stack_name = 'noisy.tif'

# -----------------------------------------------------------------------------

img = io.imread(Path('../data', stack_name)).astype(float)
kernel_size = 5

#%%

# def imfilt(img, kernel_size):

#     # Pad img
#     pad = kernel_size//2
#     img_pad = img.copy()   
#     img_pad = np.pad(img.astype(float), pad, constant_values=np.nan)
    
#     # Filt img
#     for y in range(img.shape[0]):
#         for x in range(img.shape[1]):
#             img[y,x] = np.nanmean(
#                 img_pad[y:y+kernel_size,x:x+kernel_size]
#                 )
        
#     return img
   
#%%

# start = time.time()
# print('python')
# img = imfilt(img, kernel_size)
# end = time.time()
# print(f'{(end-start):5.3f} s') 
    
#%%

from imfilt import imfilt

start = time.time()
print('cython')
img = imfilt(img, kernel_size)
end = time.time()
print(f'{(end-start):5.3f} s') 

#%%
    
io.imsave(
    Path('../data', stack_name.replace('.tif', '_filt.tif')),
    img.astype('float32'),
    check_contrast=False,
    )
