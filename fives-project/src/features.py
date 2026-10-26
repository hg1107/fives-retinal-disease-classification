"""Image -> feature vector. Uses ONLY the fundus image (never the masks).

Pipeline (preprocess): resize longest side to 512 + pad to square -> detect the field of view (FOV) ->
crop to the FOV box, pad to square, resize to 512 -> circular FOV mask -> CLAHE on the green channel.
Everything outside the FOV mask is set to 0, which also removes the corner overlay glyph found in EDA.
"""
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import frangi
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, skeletonize

FEATURE_VERSION = "v1"  # bump to invalidate the per-image cache when the feature code changes
FEATURE_VERSION_NORM = "norm_v1"  # separate cache for the normalised variant (never touches the v1 cache)
SIZE = 512
FOV_THRESH = 10  # gray level separating the black border from the retina (after blur)
CIRCLE_FRAC = 0.96  # circular mask radius as a fraction of SIZE/2 (trims the bright rim of the FOV)
ERODE_PX = 3
VESSEL_ERODE_PX = 10  # extra erosion for vessel filtering so the FOV edge is not detected as a ridge
FRANGI_SIGMAS = (1, 2, 3, 4)  # pixels at 512 px scale (vessels are ~2-12 px wide)
VESSEL_REL_THRESH = 0.10  # binarise Frangi response at this fraction of its 99th percentile
MIN_VESSEL_PX = 100  # connected vessel candidates smaller than this are speckle
GLCM_LEVELS = 32
GLCM_DISTANCES = (1, 3, 5)
GLCM_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
LBP_P, LBP_R = 8, 1
PERCENTILES = (5, 25, 50, 75, 95)
_CLAHE = dict(clipLimit=2.0, tileGridSize=(8, 8))
ILLUM_SIGMA = SIZE / 8  # large-scale Gaussian for illumination estimation (1/8 of image width)
NORM_MEAN, NORM_STD = 128.0, 48.0  # per-channel target inside the FOV after standardisation (std 48 rarely clips)


# ------------------------------------------------------------------ preprocessing
def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def as_rgb(x):
    """Path -> RGB uint8 array; array-like (ndarray / PIL image) -> RGB uint8 (gray is stacked, alpha dropped, other dtypes clipped to 0-255)."""
    if isinstance(x, (str, Path)):
        return read_rgb(x)
    a = np.asarray(x)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError(f"expected an HxW, HxWx3 or HxWx4 image, got shape {a.shape}")
    a = a[..., :3]
    return a if a.dtype == np.uint8 else np.clip(a, 0, 255).astype(np.uint8)


def _pad_square(img):
    h, w = img.shape[:2]
    s = max(h, w)
    out = np.zeros((s, s) + img.shape[2:], img.dtype)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return out


def _resize_square(img, size=SIZE):
    """Keep aspect ratio (longest side -> size), pad with black to a square."""
    h, w = img.shape[:2]
    k = size / max(h, w)
    if k != 1:
        img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    return _pad_square(img)


def _fov(rgb):
    """Boolean FOV mask: largest bright component, holes filled; thin protrusions (e.g. the notch) opened away."""
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (0, 0), 2)
    m = (gray > FOV_THRESH).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        raise ValueError("no field of view found")
    m = lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    return ndi.binary_fill_holes(m)


def preprocess(path, aux_path=None, perturb=None):
    """path (or RGB array) -> dict(rgb: masked uint8 RGB SIZExSIZE, green: CLAHE green (masked), mask: bool FOV mask).

    perturb (optional callable uint8 RGB SIZExSIZE -> same) degrades the image right after the first resize; used only by the
    robustness analysis (script 09) to re-extract features from corrupted images.
    aux_path (optional grayscale image of the same frame, e.g. a ground-truth mask used ONLY for sanity checks)
    is pushed through the identical geometry and returned as "aux"; it never influences the image features.
    """
    img = _resize_square(as_rgb(path))
    if perturb is not None:
        img = perturb(img)
    ys, xs = np.where(_fov(img))
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    img = cv2.resize(_pad_square(img[box]), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    yy, xx = np.ogrid[:SIZE, :SIZE]
    c = (SIZE - 1) / 2
    circle = (yy - c) ** 2 + (xx - c) ** 2 <= (CIRCLE_FRAC * SIZE / 2) ** 2
    mask = _fov(img) & circle
    mask = cv2.erode(mask.astype(np.uint8), np.ones((2 * ERODE_PX + 1,) * 2, np.uint8)).astype(bool)
    green = cv2.createCLAHE(**_CLAHE).apply(img[..., 1])
    out = {"rgb": (img * mask[..., None]).astype(np.uint8), "green": (green * mask).astype(np.uint8), "mask": mask}
    if aux_path is not None:
        aux = _resize_square(cv2.imread(str(aux_path), cv2.IMREAD_GRAYSCALE))
        out["aux"] = cv2.resize(_pad_square(aux[box]), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return out


# ------------------------------------------------------------------ colour
def color_features(rgb, mask):
    spaces = {"RGB": rgb, "HSV": cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), "LAB": cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)}
    names = {"RGB": "RGB", "HSV": "HSV", "LAB": ["L", "a", "b"]}
    out = {}
    for sp, arr in spaces.items():
        px = arr[mask].astype(np.float64)
        stats = {"mean": px.mean(0), "std": px.std(0), "skew": skew(px, axis=0), "kurt": kurtosis(px, axis=0)}
        pct = np.percentile(px, PERCENTILES, axis=0)
        for i, ch in enumerate(names[sp]):
            for k, v in stats.items():
                out[f"color_{sp}_{ch}_{k}"] = v[i]
            for p, v in zip(PERCENTILES, pct):
                out[f"color_{sp}_{ch}_p{p}"] = v[i]
    return out


# ------------------------------------------------------------------ texture
def texture_features(green, mask):
    c, h = SIZE // 2, int(0.7 * CIRCLE_FRAC * SIZE / 2)  # inscribed square: lies fully inside the circular FOV
    q = (green[c - h:c + h, c - h:c + h] // (256 // GLCM_LEVELS)).astype(np.uint8)
    glcm = graycomatrix(q, GLCM_DISTANCES, GLCM_ANGLES, levels=GLCM_LEVELS, symmetric=True, normed=True)
    out = {f"tex_glcm_{p}": float(graycoprops(glcm, p).mean()) for p in ("contrast", "homogeneity", "energy", "correlation")}
    lbp = local_binary_pattern(green, LBP_P, LBP_R, method="uniform")[mask]
    hist = np.bincount(lbp.astype(int), minlength=LBP_P + 2) / lbp.size
    out.update({f"tex_lbp_{i}": float(v) for i, v in enumerate(hist)})
    return out


# ------------------------------------------------------------------ vessels
def _neighbours(skel):
    return ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - skel


def vessel_region(mask):
    """FOV region where vessels are measured (FOV mask eroded so its rim is not detected as a ridge)."""
    return cv2.erode(mask.astype(np.uint8), np.ones((2 * VESSEL_ERODE_PX + 1,) * 2, np.uint8)).astype(bool)


def vessel_features(green, mask):
    g = green.astype(np.float64) / 255
    g[~mask] = g[mask].mean()  # flat fill: no artificial ridge at the FOV edge
    inner = vessel_region(mask)
    resp = frangi(g, sigmas=FRANGI_SIGMAS, black_ridges=True) * inner  # vessels are dark on the green channel
    binary = resp > VESSEL_REL_THRESH * np.percentile(resp[inner], 99)
    binary = remove_small_objects(binary, max_size=MIN_VESSEL_PX - 1)  # drops objects smaller than MIN_VESSEL_PX
    skel = skeletonize(binary)
    dist = ndi.distance_transform_edt(binary)
    nb = _neighbours(skel)
    branch = skel & (nb >= 3)
    seg = skel & ~cv2.dilate(branch.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    tort = []
    for r in regionprops(label(seg, connectivity=2)):
        if r.area < 10:
            continue
        xy = r.coords.astype(float)
        a = xy[np.argmax(np.linalg.norm(xy - xy[0], axis=1))]  # double sweep: ends of the segment
        b = xy[np.argmax(np.linalg.norm(xy - a, axis=1))]
        tort.append(r.area / max(np.linalg.norm(a - b), 1.0))
    th = 2 * dist[skel] if skel.any() else np.array([0.0])
    area = inner.sum()
    return {
        "vessel_density": binary.sum() / area,
        "vessel_thickness_mean": th.mean(),
        "vessel_thickness_std": th.std(),
        "vessel_skeleton_length": float(skel.sum()),
        "vessel_skeleton_density": skel.sum() / area,
        "vessel_branch_points": float(label(branch, connectivity=2).max()),
        "vessel_end_points": float((skel & (nb == 1)).sum()),
        "vessel_n_segments": float(len(tort)),
        "vessel_tortuosity_mean": float(np.mean(tort)) if tort else 0.0,
        "vessel_tortuosity_std": float(np.std(tort)) if tort else 0.0,
        "vessel_frangi_mean": float(resp[inner].mean()),
    }


# ------------------------------------------------------------------ bright regions (optic disc / lesions)
def disc_features(rgb, mask):
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (0, 0), 4).astype(np.float64)
    ref = max(np.percentile(gray[mask], 99), 1.0)  # near-peak brightness of this image
    out = {"disc_peak_to_mean": ref / max(gray[mask].mean(), 1.0)}
    for frac in (0.9, 0.8):
        out[f"disc_area_ratio_{int(frac * 100)}"] = float(((gray >= frac * ref) & mask).sum() / mask.sum())
    hi = (gray >= 0.9 * ref) & mask
    lab = label(hi, connectivity=2)
    regs = [r for r in regionprops(lab) if r.area >= 50]
    out["disc_n_components"] = float(len(regs))
    c, rad = (SIZE - 1) / 2, CIRCLE_FRAC * SIZE / 2
    if regs:
        r = max(regs, key=lambda r: r.area)
        cy, cx = r.centroid
        out.update({"disc_centroid_x": (cx - c) / rad, "disc_centroid_y": (cy - c) / rad,
                    "disc_dist_center": float(np.hypot(cx - c, cy - c) / rad), "disc_eccentricity": float(r.eccentricity),
                    "disc_solidity": float(r.solidity), "disc_largest_area_ratio": float(r.area / mask.sum())})
    else:  # no sizeable bright blob: neutral values (centre, round, zero area)
        out.update({"disc_centroid_x": 0.0, "disc_centroid_y": 0.0, "disc_dist_center": 0.0, "disc_eccentricity": 0.0,
                    "disc_solidity": 0.0, "disc_largest_area_ratio": 0.0})
    return out


def extract_handcrafted(path):
    p = preprocess(path)
    f = {}
    f.update(color_features(p["rgb"], p["mask"]))
    f.update(texture_features(p["green"], p["mask"]))
    f.update(vessel_features(p["green"], p["mask"]))
    f.update(disc_features(p["rgb"], p["mask"]))
    return {k: float(v) for k, v in f.items()}


def normalise(p):
    """Normalised variant of a preprocess() result, for the color_ and tex_ features.

    1. Illumination correction: green / (Gaussian blur of green, sigma = SIZE/8). The blur is computed inside the FOV
       only (masked convolution: blur(g*m)/blur(m)) so the black border does not darken the estimate near the rim.
    2. Per-channel standardisation inside the FOV (zero mean, unit variance), mapped to NORM_MEAN/NORM_STD and clipped
       to uint8, so HSV/LAB conversion still works. Consequence: RGB per-channel mean/std become (near-)constant and
       are dropped later by VarianceThreshold; the remaining signal is distribution shape and cross-channel relations.
    3. CLAHE on the normalised green channel (as in the base pipeline).
    """
    mask = p["mask"]
    rgb = p["rgb"].astype(np.float64)
    w = mask.astype(np.float64)
    bg = cv2.GaussianBlur(rgb[..., 1] * w, (0, 0), ILLUM_SIGMA) / np.maximum(cv2.GaussianBlur(w, (0, 0), ILLUM_SIGMA), 1e-6)
    rgb[..., 1] = rgb[..., 1] / np.maximum(bg, 1.0)
    out = np.zeros(rgb.shape, np.uint8)
    for c in range(3):
        v = rgb[..., c][mask]
        z = (rgb[..., c] - v.mean()) / max(v.std(), 1e-6)
        out[..., c] = np.clip(NORM_MEAN + NORM_STD * z, 0, 255)
    out *= mask[..., None].astype(np.uint8)
    green = cv2.createCLAHE(**_CLAHE).apply(out[..., 1]) * mask
    return {"rgb": out, "green": green.astype(np.uint8), "mask": mask}


def extract_handcrafted_norm(path, perturb=None):
    """Like extract_handcrafted, but color_ and tex_ come from the normalised image; vessel_/disc_ are unchanged."""
    p = preprocess(path, perturb=perturb)
    n = normalise(p)
    f = {}
    f.update(color_features(n["rgb"], n["mask"]))
    f.update(texture_features(n["green"], n["mask"]))
    f.update(vessel_features(p["green"], p["mask"]))
    f.update(disc_features(p["rgb"], p["mask"]))
    return {k: float(v) for k, v in f.items()}


def resnet_input(path, size=224):
    """Preprocessed (FOV-cropped, FOV-masked, no CLAHE) RGB resized to size x size, uint8 HWC."""
    return cv2.resize(preprocess(path)["rgb"], (size, size), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------ perturbations (robustness analysis only)
def perturb_blur(img, sigma):
    """Gaussian blur with std `sigma` pixels (at SIZE resolution)."""
    return cv2.GaussianBlur(img, (0, 0), sigma)


def perturb_brightness(img, gain):
    """Multiplicative brightness change. Multiplicative (not additive) so the black border stays black and FOV detection is unaffected."""
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def perturb_noise(img, sigma):
    """Additive Gaussian noise (std `sigma` gray levels) on the retina only (gray > FOV_THRESH); seeded from the image content."""
    rng = np.random.default_rng(int(img.sum()) % (2**32))
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape).astype(np.float32)
    retina = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > FOV_THRESH)[..., None]
    return np.where(retina, np.clip(noisy, 0, 255), img).astype(np.uint8)
