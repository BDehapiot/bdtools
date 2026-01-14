#%% Imports -------------------------------------------------------------------

import numpy as np
from numba import njit

# skimage
from skimage.measure import label

#%% Comments ------------------------------------------------------------------

'''
- works only for 2D images, maybe implement 3D?
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
    if not (np.issubdtype(arr.dtype, np.integer) or
            np.issubdtype(arr.dtype, np.bool_)):
        raise TypeError("Input array must be bool or integers labels")
    
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

    # Parameters
    conn = 2
    
    # Paths
    # dataset = "wdisk_skel"
    dataset = "fluo_nuclei_instance"
    # dataset = "fluo_nuclei_semantic"
    data_path = Path.cwd().parent / "_local" / dataset
    if dataset == "wdisk_skel":
        msk_path = list(data_path.rglob(f"*{dataset}.tif"))[0]
        msk = io.imread(msk_path).astype(bool)
    else:
        msk_path = list(data_path.glob("*msk_trn.tif"))[0]
        msk = io.imread(msk_path)[0]
    
    # -------------------------------------------------------------------------
    
    t0 = time.time()
    print("pix_conn() : ", end="", flush=True)
    
    pconn = pix_conn(msk, conn=conn)

    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # -------------------------------------------------------------------------
    
    t0 = time.time()
    print("lbl_conn() : ", end="", flush=True)
    
    lconn = lbl_conn(msk, conn=conn)

    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # -------------------------------------------------------------------------
    
    # Display
    vwr = napari.Viewer()
    vwr.add_labels(msk, visible=1)
    vwr.add_labels(pconn, visible=0)
    vwr.add_labels(lconn, visible=0)