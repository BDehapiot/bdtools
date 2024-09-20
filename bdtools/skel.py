#%% Imports

import numpy as np

# Skimage
from skimage.morphology import label

#%% Function: pix_conn() ------------------------------------------------------

def pix_conn(arr, conn=2):

    """ 
    Count number of connected pixels.
    
    Parameters
    ----------
    arr : 2D ndarray (bool)
        Skeleton/binary image.
        
    conn: int
        conn = 1, horizontal + vertical connected pixels.
        conn = 2, horizontal + vertical + diagonal connected pixels.
    
    Returns
    -------  
    arr : 2D ndarray (uint8)
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
    
    # Convert arr as bool
    arr = arr.astype('bool')
    
    # Pad arr with False
    arr = np.pad(arr, pad_width=1, constant_values=False)
    
    # Find True coordinates
    idx = np.where(arr == True) 
    idx_y = idx[0]; idx_x = idx[1]
    
    # Define all kernels
    mesh_range = np.arange(-1, 2)
    mesh_x, mesh_y = np.meshgrid(mesh_range, mesh_range)
    kernel_y = idx_y[:, None, None] + mesh_y
    kernel_x = idx_x[:, None, None] + mesh_x
    
    # Filter image
    all_kernels = arr[kernel_y,kernel_x]
    if conn == 1:
        all_kernels = np.sum(all_kernels*conn1, axis=(1, 2))
    if conn == 2:    
        all_kernels = np.sum(all_kernels*conn2, axis=(1, 2))
    arr = arr.astype('uint8')
    arr[idx] = all_kernels
    
    # Unpad arr
    arr = arr[1:-1,1:-1]
    
    return arr

#%% Function: lab_conn() ------------------------------------------------------

def lab_conn(arr, conn=2):

    """ 
    Count number of connected different labels.
    
    Parameters
    ----------
    arr : 2D ndarray (bool)
        Skeleton/binary image.
        
    conn: int
        conn = 1, horizontal + vertical connected pixels.
        conn = 2, horizontal + vertical + diagonal connected pixels.
    
    Returns
    -------  
    arr : 2D ndarray (uint8)
        Processed image.
        Pixel intensity representing number of connected different labels.
    
    """       

    conn1 = np.array(
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]])
    
    # Convert arr as bool
    arr = arr.astype('bool')
    
    # Create labels
    labels = label(np.invert(arr), connectivity=1)
    
    # Pad arr and labels with False and 0
    arr = np.pad(arr, pad_width=1, constant_values=False)
    labels = np.pad(labels, pad_width=1, constant_values=0)
    
    # Find True coordinates
    idx = np.where(arr == True) 
    idx_y = idx[0]; idx_x = idx[1]
    
    # Define all kernels
    mesh_range = np.arange(-1, 2)
    mesh_x, mesh_y = np.meshgrid(mesh_range, mesh_range)
    kernel_y = idx_y[:, None, None] + mesh_y
    kernel_x = idx_x[:, None, None] + mesh_x
    
    # Filter image
    all_kernels = labels[kernel_y,kernel_x]
    if conn == 1:
        all_kernels = all_kernels*conn1
    all_kernels = all_kernels.reshape((all_kernels.shape[0], -1))
    all_kernels.sort(axis=1)  
    arr = arr.astype('uint8')
    arr[idx] = (np.diff(all_kernels) > 0).sum(axis=1)
    
    # Unpad arr
    arr = arr[1:-1,1:-1]
    
    return arr