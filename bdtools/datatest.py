#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

# bdtools
from bdtools.norm import norm_pct

# skimage
from skimage.measure import label

# scipy
from scipy import ndimage as ndi




#%% Function : random_objects() -----------------------------------------------

def random_objects(
        shape,
        is3D=False,
        n_objects=16,
        radius=(8, 16),
        elongation=0.0,
        irregularity=0.0,
        output_type="labels",
        brightness=(2.0, 3.0),
        noise=0.1,
        seed=None,
        ):
    """
    Generate a random 2D or 3D image containing objects.

    radius, elongation and irregularity accept a single value (same for all
    objects) or a (min, max) range (drawn randomly for each object).

    Parameters
    ----------
    shape : tuple of int
        (nY, nX) or (nZ, nY, nX).
    is3D : bool
        Only used if shape is 3D. True -> real 3D objects.
        False -> stack of independent 2D planes (n_objects per plane,
        labels unique across the stack).
    output_type : str
        "labels"         : int32 label array, sequential (0 = background)
        "mask"           : boolean mask
        "image"          : float32 noisy image of objects over background
        "nan_objects"    : float32 noisy background, objects set to NaN
        "nan_background" : float32 noisy objects, background set to NaN
        Float outputs are scaled to 0-1 at the end by dividing by their
        max, so intensity ratios are preserved.
    n_objects : int
        Number of objects (may end up slightly lower: objects can overlap
        and a fully covered object disappears).
    radius : float or (float, float)
        Object radius in pixels (longest semi-axis).
    elongation : float or (float, float)
        0 = round, 0.5 = short axis is half the long axis, max 0.9.
        Elongated objects are randomly oriented.
    irregularity : float or (float, float)
        0 = smooth outline, ~0.5 = bumpy, ~1.5 = very irregular.
    brightness : float or (float, float)
        "image" and "nan_background" only.
        Object intensity as a ratio of the background
        (object = background * brightness):
        2 = twice as bright as the background, 1 = invisible,
        0.5 = half as bright (dark objects), 0 = black (minimum).
        A range gives each object its own random brightness.
    noise : float
        Poisson noise strength for float outputs, as the noise std
        relative to the background (0 = no noise, 0.1 = 10 %, 0.5 = 50 %).
        Brighter objects are noisier in absolute terms, as in real images.
    seed : int or None
        Random seed for reproducibility.
    """

    # Nested function(s) ------------------------------------------------------
    
    def _sample(value, rng):
        """Single value -> value, (min, max) -> random value in range."""
        if np.isscalar(value):
            return float(value)
        return rng.uniform(min(value), max(value))

    def _random_rotation(ndim, rng):
        q, r = np.linalg.qr(rng.standard_normal((ndim, ndim)))
        return q * np.sign(np.diag(r))

    def _make_object(ndim, radius, elongation, irregularity, rng):
        """Binary object centred in its own bounding box."""

        # Semi-axes : one axis = radius, one = radius * (1 - elongation),
        # (3D) third one in between
        radius = max(radius, 1)
        elongation = np.clip(elongation, 0, 0.9)
        axes = np.full(ndim, radius)
        axes[1] = radius * (1 - elongation)
        if ndim == 3:
            axes[2] = radius * (1 - elongation * rng.uniform())
        axes = np.maximum(axes, 1)

        # Coordinate grid
        half = int(np.ceil(radius * (1 + irregularity))) + 1
        grid = np.mgrid[(slice(-half, half + 1),) * ndim].astype(float)

        # Normalised distance field (randomly rotated)
        rot = np.einsum("ij,j...->i...", _random_rotation(ndim, rng), grid)
        dist = np.sqrt(sum((rot[i] / axes[i]) ** 2 for i in range(ndim)))

        # Irregularity : deform with smooth random noise, keep largest component
        if irregularity > 0:
            noise = ndi.gaussian_filter(
                rng.standard_normal(dist.shape), max(radius * 0.25, 1))
            dist += irregularity * 0.5 * noise / (noise.std() + 1e-12)
            lbl, n = ndi.label(dist <= 1)
            if n > 1:
                sizes = ndi.sum(lbl > 0, lbl, range(1, n + 1))
                return lbl == np.argmax(sizes) + 1

        return dist <= 1

    def _random_labels(
            shape, n_objects, radius, elongation, irregularity, rng):
        """Sequential label array with n_objects random (overlapping) objects."""

        lbl = np.zeros(shape, dtype="int32")
        for i in range(n_objects):
            obj = _make_object(
                len(shape),
                _sample(radius, rng),
                _sample(elongation, rng),
                _sample(irregularity, rng),
                rng,
                )

            # Paste centred on a random position, clipped to array bounds
            arr_sl, obj_sl = [], []
            for s, o in zip(shape, obj.shape):
                s0 = rng.integers(0, s) - o // 2
                s1 = s0 + o
                arr_sl.append(slice(max(s0, 0), min(s1, s)))
                obj_sl.append(slice(max(-s0, 0), o - max(s1 - s, 0)))
            lbl[tuple(arr_sl)][obj[tuple(obj_sl)]] = i + 1

        # Fully covered objects vanish -> make labels sequential again
        uniq = np.unique(lbl)
        lut = np.zeros(n_objects + 1, dtype="int32")
        lut[uniq] = np.arange(len(uniq))
        return lut[lbl]
    
    # Execute -----------------------------------------------------------------

    rng = np.random.default_rng(seed)
    shape = tuple(int(s) for s in shape)
    if len(shape) not in (2, 3):
        raise ValueError("shape must be (nY, nX) or (nZ, nY, nX)")
    output_types = (
        "labels", "mask", "image", "nan_objects", "nan_background")
    if output_type not in output_types:
        raise ValueError(f"output_type must be one of {output_types}")

    # Labels
    args = (n_objects, radius, elongation, irregularity, rng)
    if len(shape) == 3 and not is3D:
        lbl = np.stack(
            [_random_labels(shape[1:], *args) for _ in range(shape[0])])
        offsets = np.cumsum([0] + [p.max() for p in lbl[:-1]])
        lbl = np.where(lbl > 0, lbl + offsets[:, None, None], 0)
        lbl = lbl.astype("int32")
    else:
        lbl = _random_labels(shape, *args)

    if output_type == "labels":
        return lbl
    if output_type == "mask":
        return lbl > 0

    # Image : background = 1, objects = brightness (ratio, per object)
    img = np.ones(shape, dtype="float32")
    if output_type in ("image", "nan_background"):
        lut = np.ones(lbl.max() + 1, dtype="float32")
        lut[1:] = [_sample(brightness, rng) for _ in range(lbl.max())]
        img = np.maximum(lut, 0)[lbl]  # intensity can't go below 0
    if noise > 0:
        scale = noise ** 2  # higher noise -> fewer "photons"
        img = (rng.poisson(img / scale) * scale).astype("float32")
    if output_type == "nan_objects":
        img[lbl > 0] = np.nan
    elif output_type == "nan_background":
        img[lbl == 0] = np.nan

    # Scale to 0-1 by dividing by the max (keeps object/background ratios,
    # NaNs are ignored and kept)
    vmax = np.nanmax(img)
    if vmax > 0:
        img = img / vmax

    return img.astype("float32")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Paths
    # dataset = "em_mito"
    # dataset = "fluo_tissue"
    # dataset = "fluo_nuclei_instance"
    # dataset = "fluo_nuclei_semantic"
    # dataset = "sat_roads"
    dataset = "chess_class"
    
    # Load data
    X, y = load_dataset(dataset, display=False)
    
    pass

    # Imports -----------------------------------------------------------------
    
    import time 
    import napari

    # Initialize --------------------------------------------------------------

    # Parameters
    shape = (64, 256, 256)
    params = {
        "is3D"         : True,
        "output_type"  : "nan_background",
        "n_objects"    : 16,
        "radius"       : (8, 16),
        "elongation"   : (0.1, 0.7),
        "irregularity" : 0.2,
        "brightness"   : (1.5, 5.0),
        "noise"        : 0.5,
        "seed"         : 1,
        }
    
    # Run ---------------------------------------------------------------------
    
    t0 = time.time()
    print("random_objects() : ", end="", flush=True)

    arr = random_objects(shape, **params)

    t1 = time.time()
    print(f"{t1 - t0:.3f}s")

    # Display -----------------------------------------------------------------
    
    vwr = napari.Viewer()
    vwr.add_image(arr, name="nan")