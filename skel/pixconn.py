#%% Imports

import numpy as np

#%% Function

def pixconn(img, conn=2):

    """ 
    Count number of connected pixels.
    
    Parameters
    ----------
    img : ndarray (bool)
        Skeleton/binary image.
        
    conn: int
        conn = 1, horizontal + vertical connected pixels.
        conn = 2, horizontal + vertical + diagonal connected pixels.
    
    Returns
    -------  
    img : ndarray (uint8)
        Processed image.
        Pixel intensity representing number of connected pixels.
    
    """    

    conn1 = np.array(
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]])
    
    conn2 = np.array(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]])
    
    # Convert img as bool
    img = img.astype('bool')
    
    # Pad img with False
    img = np.pad(img, pad_width=1, constant_values=False)
    
    # Find True coordinates
    idx = np.where(img == True) 
    idx_y = idx[0]; idx_x = idx[1]
    
    # Define all kernels
    mesh_range = np.arange(-1, 2)
    mesh_x, mesh_y = np.meshgrid(mesh_range, mesh_range)
    kernel_y = idx_y[:, None, None] + mesh_y
    kernel_x = idx_x[:, None, None] + mesh_x
    
    # Filter image
    all_kernels = img[kernel_y,kernel_x]
    if conn == 1:
        all_kernels = np.sum(all_kernels*conn1, axis=(1, 2))
    if conn == 2:    
        all_kernels = np.sum(all_kernels*conn2, axis=(1, 2))
    img = img.astype('uint8')
    img[idx] = all_kernels
    
    # Unpad img
    img = img[1:-1,1:-1]
    
    return img