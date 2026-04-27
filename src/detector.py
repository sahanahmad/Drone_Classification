"""
detector.py
-----------
Runs the trained SVM drone detector on a test image using a sliding window.

Pipeline (mirrors FPCV-2-5 face detection pipeline, slides 10-11):
  1. Slide a window of fixed size across the image in a raster scan
  2. At each location, extract Haar features using the integral image
  3. Query the SVM: is this window a drone (+1) or background (-1)?
  4. Collect all positive windows with their confidence scores
  5. Apply non-maximal suppression (NMS) to collapse nearby detections
     (same NMS used for corners in FPCV-2-1, generalized here to boxes)

Multi-scale detection:
  Resize the image to multiple scales so we can detect drones of
  different sizes — exactly the "Slide windows of different sizes"
  approach described in slide 10 of FPCV-2-5.
"""

import numpy as np
import cv2
import pickle
from pathlib import Path

from haar_features import extract_features, get_templates

WINDOW_SIZE  = 64
HAAR_STEP    = 8
STRIDE       = 8          # sliding window step (pixels)
SCALES       = [1.0, 0.75, 0.5]   # image downscale factors for multi-scale
THRESHOLD    = 0.92        # SVM probability threshold for positive detection
NMS_IOU      = 0.3         # IoU threshold for non-maximal suppression


# ── Load model ──────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['svm'], data['scaler']


# ── Sliding window detection at one scale ──────────────────────────────────────

def detect_at_scale(img_gray: np.ndarray,
                     svm, scaler,
                     scale: float,
                     templates,
                     stride: int = STRIDE,
                     threshold: float = THRESHOLD):
    """
    Run the sliding window detector on a single scaled version of the image.

    Returns list of (x1, y1, x2, y2, score) bounding boxes in original
    image coordinates.
    """
    H_orig, W_orig = img_gray.shape
    H = int(H_orig * scale)
    W = int(W_orig * scale)

    if H < WINDOW_SIZE or W < WINDOW_SIZE:
        return []

    img_scaled = cv2.resize(img_gray, (W, H))
    detections = []

    for r in range(0, H - WINDOW_SIZE + 1, stride):
        for c in range(0, W - WINDOW_SIZE + 1, stride):
            patch = img_scaled[r:r + WINDOW_SIZE, c:c + WINDOW_SIZE]

            # Extract Haar features (FPCV-2-5 slide 17, 25)
            feat = extract_features(patch, templates, WINDOW_SIZE, HAAR_STEP)
            feat_sc = scaler.transform(feat.reshape(1, -1))

            # SVM classification (slide 49)
            prob = svm.predict_proba(feat_sc)[0]
            score = prob[1]  # probability of class +1 (drone)

            if score >= threshold:
                # Map back to original image coordinates
                x1 = int(c / scale)
                y1 = int(r / scale)
                x2 = int((c + WINDOW_SIZE) / scale)
                y2 = int((r + WINDOW_SIZE) / scale)
                detections.append((x1, y1, x2, y2, score))

    return detections


# ── Non-maximal suppression ────────────────────────────────────────────────────

def iou(box_a, box_b):
    """Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def non_maximal_suppression(detections, iou_thresh=NMS_IOU):
    """
    Non-maximal suppression: keeps only the highest-scoring box
    from overlapping groups.

    This is the same principle as the corner NMS in FPCV-2-1 (slide 49):
    slide a window, suppress anything that isn't the local maximum.
    Here we suppress based on bounding box overlap (IoU) instead.

    detections: list of (x1, y1, x2, y2, score)
    """
    if not detections:
        return []

    # Sort by score descending
    dets = sorted(detections, key=lambda d: d[4], reverse=True)
    kept = []

    while dets:
        best = dets.pop(0)
        kept.append(best)
        dets = [d for d in dets if iou(best[:4], d[:4]) < iou_thresh]

    return kept


# ── Full detection pipeline ────────────────────────────────────────────────────

def detect(img_path: str,
           model_path: str,
           output_path: str = None,
           scales: list = SCALES,
           threshold: float = THRESHOLD):
    """
    Full multi-scale sliding window detection on a single image.

    Returns list of final detections (x1, y1, x2, y2, score).
    """
    # Load
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    svm, scaler = load_model(model_path)
    templates   = get_templates(WINDOW_SIZE, HAAR_STEP)

    print(f"Image: {Path(img_path).name}  ({img_gray.shape[1]}x{img_gray.shape[0]})")
    print(f"Running detector at {len(scales)} scales: {scales}")

    # Collect detections across all scales
    all_detections = []
    for scale in scales:
        dets = detect_at_scale(img_gray, svm, scaler, scale, templates, threshold=threshold)
        all_detections.extend(dets)
        print(f"  Scale {scale:.2f}: {len(dets)} raw detections")

    # Non-maximal suppression
    final = non_maximal_suppression(all_detections)
    print(f"\nAfter NMS: {len(final)} detection(s)")

    # Draw results
    result = img_bgr.copy()
    for (x1, y1, x2, y2, score) in final:
        color = (0, 220, 50)   # green
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = f"drone {score:.2f}"
        cv2.putText(result, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Result saved to: {output_path}")

    return final, result


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    detect(
        img_path   = str(base / "data" / "test" / "test_scene.png"),
        model_path = str(base / "output" / "drone_svm.pkl"),
        output_path= str(base / "output" / "detection_result.png")
    )
