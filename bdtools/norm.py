#%% Imports -------------------------------------------------------------------

import numpy as np

#%%

def normalize_gcn(img):
    mask = img != 0
    img = img - np.mean(img[mask])
    img = img / np.std(img[mask])     
    img[~mask] = np.nan
    return img

def normalize_pct(imgs, min_pct, max_pct):
    values = np.concatenate(
        [np.random.choice(img.ravel(), size=1000) for img in imgs])
    pMin = np.nanpercentile(values, min_pct)
    pMax = np.nanpercentile(values, max_pct)
    for img in imgs:
        np.clip(img, pMin, pMax, out=img)
        img -= pMin
        img /= (pMax - pMin)
    return imgs
