"""
data_generator.py
-----------------
Synthesizes training data for the drone silhouette detector.

Since we don't have a real labeled dataset, we procedurally generate:
  - Positive samples: simple drone silhouettes (X or + shaped body with rotors)
  - Negative samples: random texture patches (gradients, noise, lines)

In a real project you would replace this with actual labeled images
(e.g. from the MAV-VID or Det-Fly datasets).
"""

import numpy as np
import cv2
import os
from pathlib import Path

WINDOW_SIZE = 64   # all patches are 64x64 pixels
N_POSITIVES  = 800
N_NEGATIVES  = 800
SEED         = 42

rng = np.random.default_rng(SEED)


# ── Positive sample: synthetic drone silhouette ──────────────────────────────

def draw_drone(img, cx, cy, scale=1.0, angle_deg=0.0):
    """
    Draw a simple quadcopter silhouette onto img (in-place).
    Body: two crossed rectangles (+ shape).
    Rotors: four small filled ellipses at arm tips.
    """
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle_deg, scale)

    def rot(pts):
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        return (M @ pts_h.T).T.astype(np.int32)

    arm = int(22 * scale)
    thick = int(5 * scale)
    rotor_r = int(9 * scale)

    # Arms (two rectangles forming a + cross)
    corners_h = rot(np.array([
        [cx - arm, cy - thick], [cx + arm, cy - thick],
        [cx + arm, cy + thick], [cx - arm, cy + thick]]))
    corners_v = rot(np.array([
        [cx - thick, cy - arm], [cx + thick, cy - arm],
        [cx + thick, cy + arm], [cx - thick, cy + arm]]))
    cv2.fillPoly(img, [corners_h], 255)
    cv2.fillPoly(img, [corners_v], 255)

    # Central body (small square)
    b = int(7 * scale)
    body = rot(np.array([
        [cx - b, cy - b], [cx + b, cy - b],
        [cx + b, cy + b], [cx - b, cy + b]]))
    cv2.fillPoly(img, [body], 255)

    # Four rotors at arm tips
    for dx, dy in [(arm, 0), (-arm, 0), (0, arm), (0, -arm)]:
        pt = rot(np.array([[cx + dx, cy + dy]]))[0]
        cv2.ellipse(img, tuple(pt), (rotor_r, rotor_r), 0, 0, 360, 255, -1)


def make_positive(size=WINDOW_SIZE):
    """Return a 64x64 uint8 image with a drone silhouette on varied background."""
    bg_val  = rng.integers(20, 80)
    img     = np.full((size, size), bg_val, dtype=np.uint8)

    # Random silhouette brightness (simulate lighting variation)
    fg_val  = rng.integers(150, 256)
    mask    = np.zeros_like(img)

    cx      = rng.integers(size // 3, 2 * size // 3)
    cy      = rng.integers(size // 3, 2 * size // 3)
    scale   = rng.uniform(0.6, 1.1)
    angle   = rng.uniform(0, 360)

    draw_drone(mask, cx, cy, scale=scale, angle_deg=angle)
    img[mask > 0] = fg_val

    # Add Gaussian noise
    noise = rng.normal(0, rng.uniform(3, 12), img.shape)
    img   = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    return img


# ── Negative sample: background texture patch ────────────────────────────────

def make_negative(size=WINDOW_SIZE):
    """Return a 64x64 uint8 image with no drone — random natural-ish texture."""
    kind = rng.choice(['noise', 'gradient', 'lines', 'blobs'])

    if kind == 'noise':
        base = rng.integers(30, 200)
        img  = rng.normal(base, rng.uniform(10, 40), (size, size))

    elif kind == 'gradient':
        angle = rng.uniform(0, np.pi)
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        XX, YY = np.meshgrid(x, y)
        img = (np.cos(angle) * XX + np.sin(angle) * YY)
        img = img * rng.uniform(100, 200) + rng.uniform(20, 80)
        img += rng.normal(0, 8, img.shape)

    elif kind == 'lines':
        img = np.full((size, size), float(rng.integers(50, 180)))
        n_lines = rng.integers(3, 10)
        for _ in range(n_lines):
            pt1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            pt2 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            val = float(rng.integers(100, 255))
            cv2.line(img.astype(np.uint8), pt1, pt2, val, rng.integers(1, 4))
        img = img.astype(float) + rng.normal(0, 10, img.shape)

    else:  # blobs
        img = np.full((size, size), float(rng.integers(50, 150)))
        n_blobs = rng.integers(2, 6)
        for _ in range(n_blobs):
            cx = int(rng.integers(0, size))
            cy = int(rng.integers(0, size))
            rx = int(rng.integers(4, 18))
            ry = int(rng.integers(4, 18))
            val = int(rng.integers(100, 255))
            cv2.ellipse(img.astype(np.uint8), (cx, cy), (rx, ry),
                        int(rng.integers(0, 180)), 0, 360, val, -1)
        img += rng.normal(0, 8, img.shape)

    return np.clip(img, 0, 255).astype(np.uint8)


# ── Main: generate and save ───────────────────────────────────────────────────

def generate_dataset(pos_dir, neg_dir, n_pos=N_POSITIVES, n_neg=N_NEGATIVES):
    Path(pos_dir).mkdir(parents=True, exist_ok=True)
    Path(neg_dir).mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_pos} positive samples...")
    for i in range(n_pos):
        img = make_positive()
        cv2.imwrite(str(Path(pos_dir) / f"drone_{i:04d}.png"), img)

    print(f"Generating {n_neg} negative samples...")
    for i in range(n_neg):
        img = make_negative()
        cv2.imwrite(str(Path(neg_dir) / f"bg_{i:04d}.png"), img)

    print("Done.")


if __name__ == "__main__":
    base = Path(__file__).parent.parent / "data"
    generate_dataset(base / "positives", base / "negatives")
