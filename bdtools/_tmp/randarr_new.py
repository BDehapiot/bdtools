#%% Imports -------------------------------------------------------------------

import numpy as np
from scipy import ndimage as ndi

#%% Helpers -------------------------------------------------------------------

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

#%% Function: random_objects() ------------------------------------------------

def random_objects(
        shape,
        is3D=False,
        output_type="labels",
        n_objects=16,
        radius=(8, 16),
        elongation=0.0,
        irregularity=0.0,
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
        "labels" : int32 label array, sequential labels (0 = background)
        "mask"   : boolean mask
        "image"  : float32 noisy image, objects brighter than background
        "nan"    : float32 noisy image, objects set to NaN
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
    noise : float
        Poisson noise strength for "image" and "nan" (0 = no noise,
        ~0.1 = moderate, ~0.5 = strong).
    seed : int or None
        Random seed for reproducibility.
    """

    rng = np.random.default_rng(seed)
    shape = tuple(int(s) for s in shape)
    if len(shape) not in (2, 3):
        raise ValueError("shape must be (nY, nX) or (nZ, nY, nX)")
    if output_type not in ("labels", "mask", "image", "nan"):
        raise ValueError(
            "output_type must be 'labels', 'mask', 'image' or 'nan'")

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

    # Image : background = 0.2, objects = random intensity in [0.5, 1.2]
    img = np.full(shape, 0.2, dtype="float32")
    if output_type == "image":
        lut = np.zeros(lbl.max() + 1, dtype="float32")
        lut[1:] = rng.uniform(0.5, 1.2, lbl.max())
        img += lut[lbl]
    if noise > 0:
        scale = noise ** 2  # higher noise -> fewer "photons"
        img = (rng.poisson(img / scale) * scale).astype("float32")
    if output_type == "nan":
        img[lbl > 0] = np.nan

    return img

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    # Imports -----------------------------------------------------------------
    
    import time 
    import napari

    # Initialize --------------------------------------------------------------

    # Parameters
    shape = (64, 256, 256)
    params = {
        "is3D"         : True,
        "output_type"  : "image",
        "n_objects"    : 64,
        "radius"       : (8, 16),
        "elongation"   : 0.0,
        "irregularity" : 0.0,
        "noise"        : 0.05,
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