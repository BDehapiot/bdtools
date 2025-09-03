#%% Imports

import numpy as np
from numba import njit

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
    
    # Initialize
    arr = arr.astype(bool)
    arr = np.pad(arr, pad_width=1, constant_values=0) # pad
    pconn = np.zeros_like(arr, dtype="uint8")
    idx = np.where(arr > 0) 
    
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

#%% Function: lab_conn() ------------------------------------------------------

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
    lconn : 2D ndarray (uint8)
        Processed image.
        Pixel intensity representing number of connected labels.
    
    """       

    conn1 = np.array(
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]])

    # Initialize
    arr = np.pad(arr, pad_width=1, constant_values=0) # pad
    lconn = np.zeros_like(arr, dtype=arr.dtype)
    idx = np.where(arr > 0) 
    
    if len(idx[0]) > 0:
    
        # Define kernels
        mesh_range = np.arange(-1, 2)
        mesh_x, mesh_y = np.meshgrid(mesh_range, mesh_range)
        kernel_y = idx[0][:, None, None] + mesh_y
        kernel_x = idx[1][:, None, None] + mesh_x
        
        # Process kernels
        all_kernels = arr[kernel_y, kernel_x]
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
    local_path = Path.cwd().parent / "_local"
    path = list(local_path.rglob("fluo_nuclei_instance_msk_trn.tif"))[0]
    
    # Open data
    labels = io.imread(path)
    masks = labels > 0
    
    # pix_conn()
    print("pix_conn() : ", end="", flush=True)
    t0 = time.time()
    pconns = []
    for msk in masks:
        pconns.append(pix_conn(msk, conn=1))
    pconns = np.stack(pconns)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # lab_conn()
    print("lab_conn() : ", end="", flush=True)
    t0 = time.time()
    lconns = []
    for lbl in labels:
        lconns.append(lab_conn(lbl, conn=2))
    lconns = np.stack(lconns)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display
    vwr = napari.Viewer()
    vwr.add_labels(labels, visible=1)
    vwr.add_labels(pconns, visible=0)
    vwr.add_labels(lconns, visible=0)