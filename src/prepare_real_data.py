"""
prepare_real_data.py
--------------------
Converts a real labeled drone dataset (YOLO format) into 64x64 patches
ready for train_svm.py.

Crops each drone bounding box → resizes to 64x64 → saves to data/positives/
Mines background patches from same images   → saves to data/negatives/

Existing synthetic patches are preserved — real data is appended.

Usage:
    python src/prepare_real_data.py
"""

import cv2
import numpy as np
from pathlib import Path

WINDOW_SIZE  = 64
PADDING      = 0.15   # expand each box by 15% for context
MIN_BOX_PX   = 20     # skip boxes smaller than this (annotation noise)
NEG_PER_IMG  = 2      # background patches mined per image

DATASET_ROOT = Path(__file__).parent.parent / "drone_dataset" / "drone_yolov8"
OUT_POS      = Path(__file__).parent.parent / "data" / "positives"
OUT_NEG      = Path(__file__).parent.parent / "data" / "negatives"

# Use train + valid splits
SPLITS = ["train", "valid"]


def parse_yolo_label(label_path, img_w, img_h):
    """Parse YOLO .txt → list of (x1,y1,x2,y2) pixel coords."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = map(float, parts[:5])
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            boxes.append((x1, y1, x2, y2))
    return boxes


def add_padding(x1, y1, x2, y2, img_w, img_h, pad=PADDING):
    """Expand bounding box by pad fraction, clamped to image bounds."""
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0,     x1 - int(bw * pad))
    y1 = max(0,     y1 - int(bh * pad))
    x2 = min(img_w, x2 + int(bw * pad))
    y2 = min(img_h, y2 + int(bh * pad))
    return x1, y1, x2, y2


def mine_negatives(img_gray, drone_boxes, n=NEG_PER_IMG):
    """Sample background patches that don't overlap any drone box."""
    H, W = img_gray.shape
    patches, attempts = [], 0
    while len(patches) < n and attempts < 200:
        attempts += 1
        x1 = np.random.randint(0, max(1, W - WINDOW_SIZE))
        y1 = np.random.randint(0, max(1, H - WINDOW_SIZE))
        x2, y2 = x1 + WINDOW_SIZE, y1 + WINDOW_SIZE
        overlap = any(
            max(x1, bx1) < min(x2, bx2) and max(y1, by1) < min(y2, by2)
            for (bx1, by1, bx2, by2) in drone_boxes
        )
        if not overlap:
            patch = img_gray[y1:y2, x1:x2]
            if patch.shape == (WINDOW_SIZE, WINDOW_SIZE):
                patches.append(patch)
    return patches


def prepare():
    OUT_POS.mkdir(parents=True, exist_ok=True)
    OUT_NEG.mkdir(parents=True, exist_ok=True)

    pos_count = len(list(OUT_POS.glob("*.png")))
    neg_count = len(list(OUT_NEG.glob("*.png")))
    print(f"Existing patches — positives: {pos_count}, negatives: {neg_count}")

    skipped_small = skipped_no_label = 0
    total_images  = 0

    for split in SPLITS:
        img_dir   = DATASET_ROOT / split / "images"
        label_dir = DATASET_ROOT / split / "labels"
        img_paths = sorted(list(img_dir.glob("*.jpg")) +
                           list(img_dir.glob("*.png")))

        print(f"\nProcessing {split}: {len(img_paths)} images")
        total_images += len(img_paths)

        for img_path in img_paths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            H, W     = img_gray.shape

            label_path = label_dir / (img_path.stem + ".txt")
            boxes = parse_yolo_label(label_path, W, H)

            if not boxes:
                skipped_no_label += 1
                continue

            # Positive patches — one per bounding box
            for (x1, y1, x2, y2) in boxes:
                if (x2 - x1) < MIN_BOX_PX or (y2 - y1) < MIN_BOX_PX:
                    skipped_small += 1
                    continue
                x1p, y1p, x2p, y2p = add_padding(x1, y1, x2, y2, W, H)
                crop = img_gray[y1p:y2p, x1p:x2p]
                if crop.size == 0:
                    continue
                patch = cv2.resize(crop, (WINDOW_SIZE, WINDOW_SIZE),
                                   interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(OUT_POS / f"real_drone_{pos_count:05d}.png"), patch)
                pos_count += 1

            # Negative patches — background regions from same image
            for patch in mine_negatives(img_gray, boxes, n=NEG_PER_IMG):
                cv2.imwrite(str(OUT_NEG / f"real_bg_{neg_count:05d}.png"), patch)
                neg_count += 1

    print(f"\n{'='*50}")
    print(f"Total images processed : {total_images}")
    print(f"Positive patches total : {pos_count}")
    print(f"Negative patches total : {neg_count}")
    print(f"Skipped (too small)    : {skipped_small}")
    print(f"Skipped (no label)     : {skipped_no_label}")
    print(f"\nNext step — retrain the SVM:")
    print(f"  python src/train_svm.py")


if __name__ == "__main__":
    prepare()
