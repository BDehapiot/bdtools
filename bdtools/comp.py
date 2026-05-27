#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
np.random.seed(42)
from skimage import io
from pathlib import Path

# bdtools
from bdtools.check import Check

#%% Comments ------------------------------------------------------------------

#%% Function(s) : old ---------------------------------------------------------

def save_compressed_mask(arr, path):
    arr_bin = np.packbits(arr > 0)        
    np.savez_compressed(path, data=arr_bin, org_shape=arr.shape)

def load_compressed_mask(path):
    with np.load(path) as loader:
        arr_bin = loader["data"]
        shape = loader["org_shape"]
        unpacked = np.unpackbits(arr_bin)[:np.prod(shape)]
        return unpacked.reshape(shape).astype(np.uint8) * 255
    
def save_compressed_stack(arr, path):
    np.savez_compressed(path, data=arr)

def load_compressed_stack(path):
    with np.load(path) as loader:
        return loader["data"]
    
#%% Function(s) : new ---------------------------------------------------------

def save_arr(data, path):
    dtypes = [bool, "uint8", "uint16", "float32"]
    Check(data, name="data", ctype=np.ndarray, dtype=dtypes) 
    is_msk = data.dtype == bool or np.isin(np.unique(data), [0, 1]).all()
    if is_msk:
        arr_bin = np.packbits(arr.astype(bool))
        np.savez_compressed(
            path, 
            data=arr_bin, 
            shape=arr.shape, 
            dtype=arr.dtype.str, 
            is_mask=True
            )
    else:
        np.savez_compressed(path, data=arr, is_mask=False)

def load_arr(path):
    with np.load(path) as loader:
        if loader["is_mask"]:
            shape = loader["shape"]
            orig_dtype = loader["dtype"]
            unpacked = np.unpackbits(loader["data"])[:np.prod(shape)]
            return unpacked.reshape(shape).astype(orig_dtype)
        else:
            return loader["data"]

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    save_path = Path("C:/Users/bdeha/Desktop") 
    
#%% Synthetic data ------------------------------------------------------------

    arr = np.random.rand(*(64, 1024, 1024))
    msk = (arr > 0.5).astype("uint8") 
    ui8 = (arr * 255).astype("uint8") 
    u16 = (arr * 65535).astype("uint16")
    f32 = (arr * 65535).astype("float32")
    
#%% Save ----------------------------------------------------------------------

    raw_path = save_path / "msk.tif"
    cmp_path = save_path / "msk.npz"

    t0 = time.time()
    print("save (raw) : ", end="", flush=True)
    io.imsave(raw_path, msk, check_contrast=False)
    t1 = time.time()
    t_raw = t1 - t0
    print(f"{t_raw:.3f}s")

    t0 = time.time()
    print("save (cmp) : ", end="", flush=True)
    save_compressed_mask(msk, cmp_path)
    t1 = time.time()
    t_cmp = t1 - t0
    print(f"{t_cmp:.3f}s")

    s_raw = raw_path.stat().st_size / 1e6
    s_cmp = cmp_path.stat().st_size / 1e6
    
    s_ratio = s_cmp / s_raw
    t_ratio = t_cmp / t_raw
    
    print(f"raw msk size = {s_raw:.2f}")
    print(f"cmp msk size = {s_cmp:.2f}")
    print(f"ratio (size) = {s_ratio:.2f}")
    print(f"ratio (time) = {t_ratio:.2f}")