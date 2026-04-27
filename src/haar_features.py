"""
haar_features.py
----------------
Implements the integral image and Haar feature extraction pipeline
exactly as described in Nayar FPCV-2-5.

Key ideas from the lecture:
  - Integral image II[i,j] = sum of all pixel values above and left of (i,j)
  - Sum of any rectangle = 3 additions using II corner values (slide 23)
  - Haar filter response = sum(white region) - sum(black region)  (slide 17)
  - Computational cost per filter = only 7-8 additions (slide 25)
  - Cost is INDEPENDENT of filter size (the key advantage)

Haar filter types (slide 16):
  Type 0 - horizontal two-region  [white | black]
  Type 1 - vertical two-region    [white / black]
  Type 2 - horizontal three-region [white | black | white]  (Laplacian-like)
  Type 3 - four-region checkerboard
"""

import numpy as np
from itertools import product


# ── Integral Image ────────────────────────────────────────────────────────────

def compute_integral_image(img: np.ndarray) -> np.ndarray:
    """
    Compute the integral image of a 2D grayscale image.

    II[i,j] = sum of I[r,c] for all r<=i, c<=j

    Implementation: single raster scan using the recurrence from slide 26:
        II[A] = II[B] + II[C] - II[D] + I[A]

    Returns an array of shape (H+1, W+1) with a zero-padded border,
    so rectangle sums never need boundary checks.
    """
    H, W = img.shape
    ii = np.zeros((H + 1, W + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)
    return ii


def rect_sum(ii: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> float:
    """
    Compute the sum of pixels in the rectangle [r1:r2, c1:c2] using II.
    Uses the 3-addition formula from slide 23:
        Sum = II[P] - II[Q] - II[S] + II[R]
    where P=bottom-right, Q=top-right, S=bottom-left, R=top-left.
    (r1,c1) is top-left INCLUSIVE, (r2,c2) is bottom-right EXCLUSIVE.
    """
    return (ii[r2, c2] - ii[r1, c2] - ii[r2, c1] + ii[r1, c1])


# ── Haar Filter Bank Definition ───────────────────────────────────────────────

def _generate_haar_templates(window_size: int = 64, step: int = 8):
    """
    Generate a list of Haar filter descriptors covering the window.

    Each descriptor is a tuple:
        (filter_type, row, col, height, width)

    We sub-sample positions and sizes with 'step' to keep feature count
    manageable (otherwise there would be ~160,000 possible filters).

    Returns list of (ftype, r, c, h, w) tuples.
    """
    templates = []
    W = window_size

    for r in range(0, W, step):
        for c in range(0, W, step):
            for h in range(step, W - r + 1, step):
                for w in range(step, W - c + 1, step):
                    # Type 0: left-white / right-black  (vertical edge)
                    if w % 2 == 0:
                        templates.append((0, r, c, h, w))
                    # Type 1: top-white / bottom-black  (horizontal edge)
                    if h % 2 == 0:
                        templates.append((1, r, c, h, w))
                    # Type 2: left-white / center-black / right-white (Laplacian-X)
                    if w % 3 == 0:
                        templates.append((2, r, c, h, w))
                    # Type 3: 2x2 checkerboard (corner-like)
                    if h % 2 == 0 and w % 2 == 0:
                        templates.append((3, r, c, h, w))

    return templates


# Pre-build template list once (reused across all windows)
_TEMPLATES = None
_WINDOW_SIZE = None

def get_templates(window_size: int = 64, step: int = 8):
    global _TEMPLATES, _WINDOW_SIZE
    if _TEMPLATES is None or _WINDOW_SIZE != window_size:
        _TEMPLATES = _generate_haar_templates(window_size, step)
        _WINDOW_SIZE = window_size
    return _TEMPLATES


# ── Haar Feature Computation ──────────────────────────────────────────────────

def compute_haar_response(ii: np.ndarray,
                           ftype: int,
                           r: int, c: int,
                           h: int, w: int) -> float:
    """
    Compute a single Haar filter response using the integral image.

    Formula (slide 17, 25):
        response = sum(white pixels) - sum(black pixels)

    All four filter types are implemented here.
    """
    if ftype == 0:
        # Horizontal two-region: left half white, right half black
        half_w = w // 2
        white = rect_sum(ii, r, c,          r + h, c + half_w)
        black = rect_sum(ii, r, c + half_w, r + h, c + w)
        return white - black

    elif ftype == 1:
        # Vertical two-region: top half white, bottom half black
        half_h = h // 2
        white = rect_sum(ii, r,          c, r + half_h, c + w)
        black = rect_sum(ii, r + half_h, c, r + h,      c + w)
        return white - black

    elif ftype == 2:
        # Three-region horizontal: white | black | white
        third_w = w // 3
        white1 = rect_sum(ii, r, c,              r + h, c + third_w)
        black  = rect_sum(ii, r, c + third_w,    r + h, c + 2 * third_w)
        white2 = rect_sum(ii, r, c + 2*third_w,  r + h, c + w)
        return (white1 + white2) - black

    elif ftype == 3:
        # 2x2 checkerboard: (TL white, TR black, BL black, BR white)
        half_h = h // 2
        half_w = w // 2
        tl = rect_sum(ii, r,          c,          r + half_h, c + half_w)
        tr = rect_sum(ii, r,          c + half_w, r + half_h, c + w)
        bl = rect_sum(ii, r + half_h, c,          r + h,      c + half_w)
        br = rect_sum(ii, r + half_h, c + half_w, r + h,      c + w)
        return (tl + br) - (tr + bl)

    return 0.0


def extract_features(img: np.ndarray,
                      templates=None,
                      window_size: int = 64,
                      step: int = 8) -> np.ndarray:
    """
    Extract the full Haar feature vector from a single window image.

    Steps (following FPCV-2-5):
      1. Compute integral image   (slide 21)
      2. Apply all Haar filters   (slide 17, 25)
      3. Return feature vector f  (slide 11)

    img must be grayscale uint8 of shape (window_size, window_size).
    Returns a 1D float64 array of length len(templates).
    """
    assert img.shape == (window_size, window_size), \
        f"Expected {window_size}x{window_size}, got {img.shape}"

    if templates is None:
        templates = get_templates(window_size, step)

    ii = compute_integral_image(img)
    features = np.array([
        compute_haar_response(ii, ft, r, c, h, w)
        for (ft, r, c, h, w) in templates
    ], dtype=np.float64)

    # L2-normalize so responses are scale-independent (illumination robustness)
    norm = np.linalg.norm(features)
    if norm > 1e-8:
        features /= norm

    return features


def extract_features_batch(images: list,
                            window_size: int = 64,
                            step: int = 8) -> np.ndarray:
    """
    Extract features from a list of images.
    Returns array of shape (N, n_features).
    """
    templates = get_templates(window_size, step)
    return np.vstack([
        extract_features(img, templates, window_size, step)
        for img in images
    ])


if __name__ == "__main__":
    # Quick sanity check
    dummy = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    feats = extract_features(dummy)
    templates = get_templates()
    print(f"Number of Haar templates: {len(templates)}")
    print(f"Feature vector shape: {feats.shape}")
    print(f"Feature range: [{feats.min():.4f}, {feats.max():.4f}]")

    # Verify integral image
    img = np.ones((4, 4), dtype=np.uint8) * 2
    ii = compute_integral_image(img)
    total = rect_sum(ii, 0, 0, 4, 4)
    assert total == 32, f"Expected 32, got {total}"
    print("Integral image sanity check: PASSED")
