[![Test](https://github.com/BDehapiot/bdtools/actions/workflows/pytest.yml/badge.svg)](https://github.com/BDehapiot/bdtools/actions/workflows/pytest.yml)  

# bdtools
Collection of tools for recurring tasks

## 

### `mask`
- **get_edt()**  
Compute Euclidean distance tranform (edt) for binary or labelled masks stored in ndarrays.
### `nan`
- **nan_filt()**  
Filter ndarrays ignoring NaN values. Mean, median and standard deviation filters are implemented.
- **nan_replace()**  
Replace NaN values in ndarrays based on surrounding pixels. 
### `norm`
- norm_gcn() 
- norm_pct()
### `patch`
- extract_patches()
- merge_patches()
### `skel`
- pix_conn()
- lab_conn()

## Todo
- flow functions

## Bugs
- nanreplace : avoid error if no nan to replace (with iterations = "inf")


