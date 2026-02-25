#%% Imports -------------------------------------------------------------------

import numpy as np
from numba import njit

# bdtools
from bdtools.check import Check_parameter

# skimage
from skimage.measure import label

#%% Comments ------------------------------------------------------------------

'''
- works only for 2D images, consider implementing 3D?
'''

#%% Function: pix_conn() ------------------------------------------------------

def pix_conn(arr, conn=2):

    """ 
    Count number of non-zero connected pixels for non-zero pixels.
    
    Parameters
    ----------
    arr : 2D ndarray
        Skeleton/binary image.
        
    conn: int
        conn = 1, horizontal + vertical connected pixels.
        conn = 2, horizontal + vertical + diagonal connected pixels.
    
    Returns
    -------  
    pconn : 2D ndarray (uint8)
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
    
    # Checks
    Check_parameter(
        arr, name="arr", ctype=np.ndarray, dtype=(int, bool), ndim=2)
        
    # Initialize
    arr = arr > 0
    arr = np.pad(arr, pad_width=1, constant_values=0) # pad
    idx = np.where(arr > 0) 
    pconn = np.zeros_like(arr, dtype="uint8")
    
    # Define kernels
    mesh_range = np.arange(-1, 2)
    mesh_x, mesh_y = np.meshgrid(mesh_range, mesh_range)
    kernel_y = idx[0][:, None, None] + mesh_y
    kernel_x = idx[1][:, None, None] + mesh_x
    
    # Process kernels
    all_kernels = arr[kernel_y,kernel_x]
    if conn == 1:
        all_kernels = np.sum(all_kernels * conn1, axis=(1, 2))
    if conn == 2:    
        all_kernels = np.sum(all_kernels * conn2, axis=(1, 2))
    
    # Fill output (pconn)
    pconn[idx] = all_kernels
    
    return pconn[1:-1, 1:-1] # un-pad

#%% Function: lbl_conn() ------------------------------------------------------

@njit
def count_unique_nonzero_rows(arr):
    n_rows = arr.shape[0]
    out = np.zeros(n_rows, dtype=np.uint8)
    for i in range(n_rows):
        seen = set()
        for val in arr[i]:
            if val != 0:
                seen.add(val)
        out[i] = len(seen)
    return out

def lbl_conn(arr, conn=2):

    """ 
    Count number of connected different labels for non-zero pixels.
    
    Parameters
    ----------
    arr : 2D ndarray (bool)
        Skeleton/binary image.
        
    conn: int
        conn = 1, horizontal + vertical connected pixels.
        conn = 2, horizontal + vertical + diagonal connected pixels.
    
    Returns
    -------  
    lconn : 2D ndarray (uint8)
        Processed image.
        Pixel intensity representing number of connected labels.
    
    """           
    
    conn1 = np.array(
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]])
    
    # Checks
    Check_parameter(
        arr, name="arr", ctype=np.ndarray, dtype=(int, bool), ndim=2)
        
    # Initialize
    msk = arr > 0
    if np.issubdtype(arr.dtype, np.bool_):
        lbl = label(~arr, connectivity=1)
    else:
        lbl = arr.copy()
    msk = np.pad(msk, pad_width=1, constant_values=0) # pad
    lbl = np.pad(lbl, pad_width=1, constant_values=0) # pad
    idx = np.where(msk > 0) 
    lconn = np.zeros_like(msk, dtype="uint8")
    
    if len(idx[0]) > 0:
    
        # Define kernels
        mesh_range = np.arange(-1, 2)
        mesh_x, mesh_y = np.meshgrid(mesh_range, mesh_range)
        kernel_y = idx[0][:, None, None] + mesh_y
        kernel_x = idx[1][:, None, None] + mesh_x
        
        # Process kernels
        all_kernels = lbl[kernel_y, kernel_x]
        if conn == 1:
            all_kernels = all_kernels * conn1
        all_kernels = all_kernels.reshape((all_kernels.shape[0], -1))
        
        # Fill output (lconn)
        lconn[idx] = count_unique_nonzero_rows(all_kernels)

    return lconn[1:-1, 1:-1] # un-pad

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
        
    # Imports
    import time
    import napari
    from skimage import io
    from pathlib import Path
    
    # Paths
    idx = "all"
    dataset = "skel_wdisk"
    # dataset = "fluo_tissue"
    # dataset = "fluo_nuclei_instance"
    # dataset = "fluo_nuclei_semantic"
    data_path = Path.cwd().parent / "_local" / dataset
    paths = list(data_path.rglob("*msk_trn.tif"))
    
    # Open data
    data = []
    for path in paths:
        data.append(io.imread(path))
    if len(data) == 1:
        data = data[0]
    if "nuclei_semantic" in dataset:
        data_1 = label(data == 1)
        data_2 = label(data == 2)
        data_3 = label(data == 3) 
        data_2[data_2 > 0] += np.max(data_1)
        data_3[data_3 > 0] += np.max(data_2)
        data = data_1 + data_2 + data_3
    if "nuclei" in dataset:
        data = list(data)
    if isinstance(idx, int):
        data = data[idx]
    
#%% pix_conn() ----------------------------------------------------------------
    
    # Parameters
    conn = 2
    
    # Initialize
    if isinstance(data, list):
        arr = data[0]
    else:
        arr = data

    t0 = time.time()
    print("pix_conn() : ", end="", flush=True)
    
    pconn = pix_conn(arr, conn=conn)

    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display
    vwr = napari.Viewer()
    vwr.add_labels(arr, visible=1)
    vwr.add_labels(pconn, blending="additive", visible=1)
    
#%% lab_conn() ----------------------------------------------------------------
    
    # Parameters
    conn = 2
    
    # Initialize
    if isinstance(data, list):
        arr = data[0]
    else:
        arr = data

    t0 = time.time()
    print("lbl_conn() : ", end="", flush=True)
    
    lconn = lbl_conn(arr, conn=conn)

    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display
    vwr = napari.Viewer()
    vwr.add_labels(arr, visible=1)
    vwr.add_labels(lconn, blending="additive", visible=1)