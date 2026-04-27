"""
run_pipeline.py
---------------
End-to-end script that:
  1. Generates synthetic training data
  2. Trains the SVM detector
  3. Synthesizes a test scene with drones embedded in background
  4. Runs detection
  5. Produces a summary visualization showing the FPCV-2-5 pipeline stages

Run this file to execute the full pipeline:
    python run_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from data_generator   import make_positive, make_negative, draw_drone
from haar_features    import compute_integral_image, extract_features, get_templates
from train_svm        import train
from detector         import detect, non_maximal_suppression, detect_at_scale, load_model

BASE       = Path(__file__).parent.parent
DATA_DIR   = BASE / "data"
OUTPUT_DIR = BASE / "output"
MODEL_PATH = OUTPUT_DIR / "drone_svm.pkl"
TEST_PATH  = DATA_DIR / "test" / "test_scene.png"

rng = np.random.default_rng(99)


# ── Synthesize test scene ──────────────────────────────────────────────────────

def make_test_scene(path: str, size: int = 512, n_drones: int = 3):
    """
    Create a test scene: sky-like background with several drones embedded.
    Returns list of ground-truth bounding boxes.
    """
    # Sky gradient background
    scene = np.zeros((size, size), dtype=np.uint8)
    for row in range(size):
        val = int(180 - row * 80 / size + rng.normal(0, 5))
        scene[row, :] = np.clip(val, 60, 200)

    # Add some cloud-like blobs (distractors)
    for _ in range(6):
        cx = int(rng.integers(50, size - 50))
        cy = int(rng.integers(50, size - 50))
        rx = int(rng.integers(20, 60))
        ry = int(rng.integers(15, 40))
        cv2.ellipse(scene, (cx, cy), (rx, ry), 0, 0, 360, 200, -1)

    scene = cv2.GaussianBlur(scene, (15, 15), 0)
    scene += rng.integers(0, 15, scene.shape, dtype=np.uint8)

    # Embed drone silhouettes
    gt_boxes = []
    for _ in range(n_drones):
        scale  = rng.uniform(0.7, 1.2)
        sz     = int(64 * scale)
        margin = sz // 2 + 5
        cx     = int(rng.integers(margin, size - margin))
        cy     = int(rng.integers(margin, size - margin))
        angle  = rng.uniform(0, 360)
        fg_val = int(rng.integers(30, 80))   # dark drone on lighter sky

        mask = np.zeros_like(scene)
        draw_drone(mask, cx, cy, scale=scale, angle_deg=angle)
        scene[mask > 0] = fg_val

        x1 = max(0, cx - sz // 2)
        y1 = max(0, cy - sz // 2)
        x2 = min(size, cx + sz // 2)
        y2 = min(size, cy + sz // 2)
        gt_boxes.append((x1, y1, x2, y2))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, scene)
    return gt_boxes


# ── Visualization: pipeline summary ───────────────────────────────────────────

def visualize_pipeline(model_path: str, output_path: str, gt_boxes: list):
    """
    Produce a multi-panel figure showing each stage of the FPCV-2-5 pipeline.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.patch.set_facecolor('#111111')
    for ax in axes.flat:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    title_kw = dict(color='#dddddd', fontsize=10, fontweight='bold', pad=6)
    label_kw = dict(color='#888888', fontsize=8)

    # Load model and scene
    svm, scaler = load_model(str(model_path))
    scene_gray  = cv2.imread(str(TEST_PATH), cv2.IMREAD_GRAYSCALE)
    scene_bgr   = cv2.cvtColor(scene_gray, cv2.COLOR_GRAY2BGR)
    templates   = get_templates(64, 8)

    # Panel 0: Input test scene
    ax = axes[0, 0]
    ax.imshow(scene_gray, cmap='gray', vmin=0, vmax=255)
    for (x1, y1, x2, y2) in gt_boxes:
        rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   lw=1.5, edgecolor='#00cc44', facecolor='none',
                                   linestyle='--')
        ax.add_patch(rect)
    ax.set_title("Input scene\n(green = ground truth)", **title_kw)
    ax.axis('off')

    # Panel 1: Integral image
    ax = axes[0, 1]
    ii = compute_integral_image(scene_gray)
    ax.imshow(ii, cmap='plasma')
    ax.set_title("Integral image II\n(FPCV-2-5 slide 21)", **title_kw)
    ax.axis('off')

    # Panel 2: Four sample Haar filters visualized
    ax = axes[0, 2]
    ax.set_title("Haar filter types\n(FPCV-2-5 slide 16)", **title_kw)
    filter_imgs = []
    for ftype, label in [(0, 'Type 0\nHoriz 2-region'),
                          (1, 'Type 1\nVert 2-region'),
                          (2, 'Type 2\n3-region'),
                          (3, 'Type 3\nCheckerboard')]:
        fimg = np.zeros((32, 32), dtype=np.float32)
        if ftype == 0:
            fimg[:, :16] = 1; fimg[:, 16:] = -1
        elif ftype == 1:
            fimg[:16, :] = 1; fimg[16:, :] = -1
        elif ftype == 2:
            fimg[:, :10] = 1; fimg[:, 10:22] = -1; fimg[:, 22:] = 1
        elif ftype == 3:
            fimg[:16, :16] = 1; fimg[16:, 16:] = 1
            fimg[:16, 16:] = -1; fimg[16:, :16] = -1
        filter_imgs.append(fimg)

    grid = np.block([[filter_imgs[0], np.zeros((32, 4)) - 2, filter_imgs[1]],
                     [np.zeros((4, 68)) - 2],
                     [filter_imgs[2], np.zeros((32, 4)) - 2, filter_imgs[3]]])
    ax.imshow(grid, cmap='RdBu_r', vmin=-1.5, vmax=1.5)
    ax.text(16, 16, 'H2', color='white', ha='center', va='center', fontsize=7)
    ax.text(54, 16, 'V2', color='white', ha='center', va='center', fontsize=7)
    ax.text(16, 54, 'H3', color='white', ha='center', va='center', fontsize=7)
    ax.text(54, 54, 'CB', color='white', ha='center', va='center', fontsize=7)
    ax.axis('off')

    # Panel 3: Haar response map for one filter type on the scene
    ax = axes[0, 3]
    stride = 8
    H, W   = scene_gray.shape
    wsize  = 64
    resp_map = np.zeros((H, W))
    ii_full  = compute_integral_image(scene_gray)
    # Apply a single Haar filter (type 0, 32x32) across the whole image
    for r in range(0, H - wsize, stride):
        for c in range(0, W - wsize, stride):
            from haar_features import rect_sum
            white = rect_sum(ii_full, r, c,      r+wsize, c+wsize//2)
            black = rect_sum(ii_full, r, c+wsize//2, r+wsize, c+wsize)
            resp_map[r + wsize//2, c + wsize//2] = white - black
    ax.imshow(resp_map, cmap='seismic', vmin=-resp_map.std()*3, vmax=resp_map.std()*3)
    ax.set_title("Haar response map\n(Type 0 filter)", **title_kw)
    ax.axis('off')

    # Panel 4: SVM decision boundary illustration (2D PCA projection)
    ax = axes[1, 0]
    ax.set_title("SVM decision boundary\n(FPCV-2-5 slide 47)", **title_kw)
    from sklearn.decomposition import PCA
    # Project support vectors to 2D for illustration
    sv = svm.support_vectors_
    sv_labels = svm.predict(sv)
    pca = PCA(n_components=2)
    sv_2d = pca.fit_transform(sv)
    pos_mask = sv_labels == 1
    neg_mask = sv_labels == -1
    ax.scatter(sv_2d[neg_mask, 0], sv_2d[neg_mask, 1],
               c='#4488ff', s=15, alpha=0.6, label='bg SV')
    ax.scatter(sv_2d[pos_mask, 0], sv_2d[pos_mask, 1],
               c='#ff6644', s=15, alpha=0.6, label='drone SV')
    ax.set_xlabel("PC 1", **label_kw)
    ax.set_ylabel("PC 2", **label_kw)
    ax.legend(fontsize=7, labelcolor='#aaaaaa',
              facecolor='#222222', edgecolor='#444444')

    # Panel 5: Sliding window detections BEFORE NMS
    ax = axes[1, 1]
    raw_dets = []
    for scale in [1.0, 0.75, 0.5]:
        raw_dets.extend(
            detect_at_scale(scene_gray, svm, scaler, scale, templates, stride=16))
    disp = scene_gray.copy()
    disp_bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    for (x1, y1, x2, y2, sc) in raw_dets:
        cv2.rectangle(disp_bgr, (x1, y1), (x2, y2), (50, 50, 200), 1)
    ax.imshow(cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Before NMS\n({len(raw_dets)} raw windows)", **title_kw)
    ax.axis('off')

    # Panel 6: After NMS
    ax = axes[1, 2]
    final_dets = non_maximal_suppression(raw_dets)
    disp2      = cv2.cvtColor(scene_gray, cv2.COLOR_GRAY2BGR)
    for (x1, y1, x2, y2, sc) in final_dets:
        cv2.rectangle(disp2, (x1, y1), (x2, y2), (0, 220, 60), 2)
        cv2.putText(disp2, f"{sc:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 60), 1)
    for (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(disp2, (x1, y1), (x2, y2), (255, 200, 0), 1)
    ax.imshow(cv2.cvtColor(disp2, cv2.COLOR_BGR2RGB))
    ax.set_title(f"After NMS — {len(final_dets)} detection(s)\n"
                 f"(green=detected, yellow=GT)", **title_kw)
    ax.axis('off')

    # Panel 7: Sample windows — detected vs not detected
    ax = axes[1, 3]
    ax.set_title("Sample patches\n(top=detected, bottom=background)", **title_kw)
    samples = []
    for (x1, y1, x2, y2, sc) in final_dets[:2]:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        r0 = max(0, cy - 32)
        c0 = max(0, cx - 32)
        patch = scene_gray[r0:r0+64, c0:c0+64]
        if patch.shape == (64, 64):
            samples.append((patch, f"drone {sc:.2f}", '#00cc44'))
    # add two background patches
    for _ in range(2):
        r0 = int(rng.integers(0, scene_gray.shape[0] - 64))
        c0 = int(rng.integers(0, scene_gray.shape[1] - 64))
        patch = scene_gray[r0:r0+64, c0:c0+64]
        samples.append((patch, "background", '#4488ff'))

    n = min(4, len(samples))
    grid_img = np.zeros((64, n * 68), dtype=np.uint8)
    for i, (patch, lbl, col) in enumerate(samples[:n]):
        grid_img[:, i*68:i*68+64] = patch
    ax.imshow(grid_img, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')

    plt.suptitle("Haar Feature + SVM Drone Detector  |  Following Nayar FPCV-2-5",
                 color='#eeeeee', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight',
                facecolor='#111111', edgecolor='none')
    plt.close()
    print(f"Pipeline visualization saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Train SVM
    print("\n" + "=" * 55)
    print("STEP 1: Training SVM on Haar features")
    print("=" * 55)
    train(
        pos_dir    = str(DATA_DIR / "positives"),
        neg_dir    = str(DATA_DIR / "negatives"),
        model_path = str(MODEL_PATH)
    )

    # Step 2: Make test scene
    print("\n" + "=" * 55)
    print("STEP 2: Creating test scene")
    print("=" * 55)
    gt_boxes = make_test_scene(str(TEST_PATH), size=400, n_drones=3)
    print(f"Test scene saved with {len(gt_boxes)} ground-truth drones")

    # Step 3: Detect
    print("\n" + "=" * 55)
    print("STEP 3: Running detector")
    print("=" * 55)
    detections, result_img = detect(
        img_path   = str(TEST_PATH),
        model_path = str(MODEL_PATH),
        output_path= str(OUTPUT_DIR / "detection_result.png"),
        threshold  = 0.92
    )

    # Step 4: Visualize pipeline
    print("\n" + "=" * 55)
    print("STEP 4: Generating pipeline visualization")
    print("=" * 55)
    visualize_pipeline(
        model_path  = MODEL_PATH,
        output_path = str(OUTPUT_DIR / "pipeline_summary.png"),
        gt_boxes    = gt_boxes
    )

    print("\n" + "=" * 55)
    print("ALL DONE")
    print("=" * 55)
    print(f"Outputs in: {OUTPUT_DIR}")
    print(f"  detection_result.png  — annotated test image")
    print(f"  pipeline_summary.png  — 8-panel FPCV-2-5 pipeline visualization")
    print(f"  drone_svm.pkl         — trained model (SVM + scaler)")


if __name__ == "__main__":
    main()
