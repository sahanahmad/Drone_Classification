"""
visualize_pipeline.py
---------------------
Data-first walkthrough: traces ONE training image through every stage
and saves a visualization at each step.

Output: output/stage_*.png
Run:    python src/visualize_pipeline.py
"""

import sys
import pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent))
from haar_features import (
    compute_integral_image,
    rect_sum,
    get_templates,
    compute_haar_response,
    extract_features,
)

ROOT   = Path(__file__).parent.parent
OUT    = ROOT / "output"
OUT.mkdir(exist_ok=True)

# ── Pick one image ─────────────────────────────────────────────────────────────
img_path = ROOT / "data" / "positives" / "drone_0000.png"
img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
assert img is not None, f"Could not load {img_path}"
print(f"Loaded: {img_path.name}  shape={img.shape}  dtype={img.dtype}")
print(f"  pixel range: [{img.min()}, {img.max()}]")
print(f"  mean={img.mean():.1f}  std={img.std():.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Raw pixel array
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Stage 1: Raw pixel array ──")
print(f"  Top-left 8×8 corner:\n{img[:8, :8]}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Stage 1 — Raw Pixel Array (64×64 uint8)", fontsize=13, fontweight='bold')

ax = axes[0]
im = ax.imshow(img, cmap='gray', vmin=0, vmax=255)
ax.set_title("Image")
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1]
# Show a 8x8 corner as a heatmap with numbers
corner = img[:8, :8]
ax.imshow(corner, cmap='gray', vmin=0, vmax=255)
for r in range(8):
    for c in range(8):
        ax.text(c, r, str(corner[r, c]), ha='center', va='center',
                fontsize=7, color='red' if corner[r, c] > 127 else 'lime')
ax.set_title("Top-left 8×8 pixel values")
ax.axis('off')

plt.tight_layout()
plt.savefig(OUT / "stage1_raw_pixels.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved → output/stage1_raw_pixels.png")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Integral Image
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Stage 2: Integral Image ──")
ii = compute_integral_image(img)
print(f"  Shape: {img.shape} → {ii.shape}  (padded by 1 on each side)")
print(f"  ii[0, :5] = {ii[0, :5]}  (zero padding row)")
print(f"  ii[1, :5] = {ii[1, :5]}  (cumsum of row 0)")
print(f"  ii[64,64] = {ii[64,64]:.0f}  (total image sum)")
print(f"  Manual check: 64×64×mean = {64*64*img.mean():.0f}  ← should match")

# Verify rect_sum on a known region
total = rect_sum(ii, 0, 0, 64, 64)
print(f"  rect_sum(full image) = {total:.0f}  ← must equal ii[64,64]")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Stage 2 — Integral Image (65×65 float64)", fontsize=13, fontweight='bold')

axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[0].set_title("Original (64×64)")
axes[0].axis('off')

im2 = axes[1].imshow(ii, cmap='hot')
axes[1].set_title("Integral Image (65×65)\nvalues grow to top-left corner →")
axes[1].axis('off')
plt.colorbar(im2, ax=axes[1], fraction=0.046)

# Show 5x5 top-left corner with numbers
ax = axes[2]
corner_ii = ii[:6, :6]
ax.imshow(corner_ii, cmap='hot')
for r in range(6):
    for c in range(6):
        ax.text(c, r, f"{int(corner_ii[r,c])}", ha='center', va='center',
                fontsize=7, color='white')
ax.set_title("Top-left 6×6 of II\n(row 0 = padding zeros)")
ax.axis('off')

plt.tight_layout()
plt.savefig(OUT / "stage2_integral_image.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved → output/stage2_integral_image.png")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Haar Filter Responses (response maps for all 4 types)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Stage 3: Haar Filter Responses ──")

# For each filter type, slide a fixed-size filter across the image and
# record the response at each position → gives a 2D response map.
def response_map(ii, ftype, h=16, w=16, stride=4):
    H_img = ii.shape[0] - 1
    W_img = ii.shape[1] - 1
    rows = range(0, H_img - h + 1, stride)
    cols = range(0, W_img - w + 1, stride)
    rmap = np.zeros((len(rows), len(cols)))
    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            rmap[ri, ci] = compute_haar_response(ii, ftype, r, c, h, w)
    return rmap

type_names = [
    "Type 0 — Vertical Edge\n(left bright / right dark)",
    "Type 1 — Horizontal Edge\n(top bright / bottom dark)",
    "Type 2 — Laplacian-X\n(bright | dark | bright)",
    "Type 3 — Checkerboard\n(corner detector)",
]
filter_params = [
    (0, 16, 16),  # (ftype, h, w)
    (1, 16, 16),
    (2, 16, 24),  # w must be divisible by 3
    (3, 16, 16),
]

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("Stage 3 — Haar Filter Responses (response map over image)", fontsize=13, fontweight='bold')

for i, ((ftype, h, w), name) in enumerate(zip(filter_params, type_names)):
    rmap = response_map(ii, ftype, h=h, w=w, stride=4)

    # Draw the filter pattern schematic
    ax_schema = axes[0, i]
    schema = np.zeros((h, w))
    if ftype == 0:
        schema[:, :w//2] = 1.0     # white left
        schema[:, w//2:] = -1.0    # black right
    elif ftype == 1:
        schema[:h//2, :] = 1.0
        schema[h//2:, :] = -1.0
    elif ftype == 2:
        third = w // 3
        schema[:, :third] = 1.0
        schema[:, third:2*third] = -1.0
        schema[:, 2*third:] = 1.0
    elif ftype == 3:
        schema[:h//2, :w//2] = 1.0
        schema[:h//2, w//2:] = -1.0
        schema[h//2:, :w//2] = -1.0
        schema[h//2:, w//2:] = 1.0

    ax_schema.imshow(schema, cmap='RdYlGn', vmin=-1, vmax=1)
    ax_schema.set_title(name, fontsize=8)
    ax_schema.axis('off')
    white_p = mpatches.Patch(color='green', label='+1 (white)')
    black_p = mpatches.Patch(color='red',   label='−1 (black)')
    ax_schema.legend(handles=[white_p, black_p], fontsize=6, loc='lower right')

    # Response map
    ax_resp = axes[1, i]
    vabs = np.abs(rmap).max() or 1
    im_r = ax_resp.imshow(rmap, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
    ax_resp.set_title(f"Response map\n(h={h}, w={w}, stride=4)\nrange [{rmap.min():.0f}, {rmap.max():.0f}]", fontsize=7)
    ax_resp.axis('off')
    plt.colorbar(im_r, ax=ax_resp, fraction=0.046)

axes[0, 0].set_ylabel("Filter Schematic", fontsize=9)
axes[1, 0].set_ylabel("Response Map", fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "stage3_haar_responses.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved → output/stage3_haar_responses.png")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Feature Vector (4,212 values)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Stage 4: Feature Vector ──")
templates = get_templates(64, 8)
print(f"  Number of templates: {len(templates)}")

features = extract_features(img, templates)
print(f"  Feature vector shape: {features.shape}")
print(f"  Range: [{features.min():.4f}, {features.max():.4f}]")
print(f"  L2 norm (should be 1.0): {np.linalg.norm(features):.6f}")
print(f"  First 10 values: {features[:10].round(4)}")

# Show which templates produced the strongest responses
top_idx  = np.argsort(np.abs(features))[-10:][::-1]
print("\n  Top-10 strongest features:")
for rank, idx in enumerate(top_idx):
    ft, r, c, h, w = templates[idx]
    tname = ['Type0-VEdge','Type1-HEdge','Type2-LapX','Type3-Check'][ft]
    print(f"    #{rank+1:2d}  idx={idx:4d}  val={features[idx]:+.4f}  "
          f"{tname} at r={r} c={c} h={h} w={w}")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Stage 4 — Haar Feature Vector (4,212 values, L2-normalized)",
             fontsize=13, fontweight='bold')

# Full feature vector as line plot
ax = axes[0, 0]
ax.plot(features, linewidth=0.4, color='steelblue')
ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
ax.set_title("Full feature vector (all 4,212 values)")
ax.set_xlabel("Feature index")
ax.set_ylabel("Value (L2-normalized)")
ax.set_xlim(0, len(features))

# Histogram of values
ax = axes[0, 1]
ax.hist(features, bins=80, color='steelblue', edgecolor='none')
ax.axvline(0, color='k', linewidth=1)
ax.set_title("Value distribution")
ax.set_xlabel("Feature value")
ax.set_ylabel("Count")

# Reshape to 2D: group by filter type
for i, (ftype_name, color) in enumerate(zip(
        ['Type0', 'Type1', 'Type2', 'Type3'],
        ['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])):
    mask = np.array([t[0] == i for t in templates])
    vals = features[mask]
    axes[1, 0].plot(vals, linewidth=0.6, label=ftype_name, color=color, alpha=0.8)
axes[1, 0].axhline(0, color='k', linewidth=0.5, linestyle='--')
axes[1, 0].set_title("Features split by filter type")
axes[1, 0].set_xlabel("Index within type")
axes[1, 0].set_ylabel("Value")
axes[1, 0].legend(fontsize=8)

# Top-10 strongest features as bar chart
ax = axes[1, 1]
vals   = features[top_idx]
labels = [f"#{i}\n{['T0','T1','T2','T3'][templates[i][0]]}" for i in top_idx]
colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in vals]
ax.barh(range(10), vals[::-1], color=colors[::-1])
ax.set_yticks(range(10))
ax.set_yticklabels(labels[::-1], fontsize=7)
ax.axvline(0, color='k', linewidth=0.8)
ax.set_title("Top-10 strongest feature responses")
ax.set_xlabel("Value")

plt.tight_layout()
plt.savefig(OUT / "stage4_feature_vector.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved → output/stage4_feature_vector.png")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — SVM Score
# ══════════════════════════════════════════════════════════════════════════════
model_path = ROOT / "output" / "drone_svm.pkl"
if not model_path.exists():
    print(f"\n── Stage 5: SVM Score ── SKIPPED (no model at {model_path})")
    print("  Run 'python src/train_svm.py' first, then re-run this script.")
else:
    print("\n── Stage 5: SVM Score ──")
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    svm    = bundle['svm']
    scaler = bundle['scaler']

    feat_scaled = scaler.transform(features.reshape(1, -1))
    prob        = svm.predict_proba(feat_scaled)[0]   # [prob_bg, prob_drone]
    decision    = svm.decision_function(feat_scaled)[0]
    pred_label  = svm.predict(feat_scaled)[0]

    prob_bg    = prob[0] if -1 in svm.classes_ else prob[0]
    # find which class index is drone (+1)
    drone_idx  = list(svm.classes_).index(1)
    prob_drone = prob[drone_idx]

    print(f"  StandardScaler: mean (first 5) = {scaler.mean_[:5].round(4)}")
    print(f"  StandardScaler: std  (first 5) = {scaler.scale_[:5].round(4)}")
    print(f"  Scaled feature range: [{feat_scaled.min():.3f}, {feat_scaled.max():.3f}]")
    print(f"  Decision function value: {decision:.4f}  (>0 → drone side of boundary)")
    print(f"  prob(background): {prob_bg:.4f}")
    print(f"  prob(drone):      {prob_drone:.4f}  ← this is the detection score")
    print(f"  Threshold=0.65 → {'DETECTED ✓' if prob_drone >= 0.65 else 'not detected'}")
    print(f"  Predicted class: {'+1 (drone)' if pred_label == 1 else '-1 (background)'}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Stage 5 — SVM Classification Output", fontsize=13, fontweight='bold')

    # Original image
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Input patch\n(drone_0000.png)")
    axes[0].axis('off')

    # Before vs after scaling (first 50 features)
    ax = axes[1]
    x = np.arange(50)
    ax.bar(x - 0.2, features[:50],          width=0.4, label='Before scaling', alpha=0.7)
    ax.bar(x + 0.2, feat_scaled[0, :50],    width=0.4, label='After scaling',  alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_title("First 50 features:\nbefore vs after StandardScaler")
    ax.set_xlabel("Feature index")
    ax.legend(fontsize=7)

    # Score gauge
    ax = axes[2]
    bar_colors = ['#e74c3c', '#2ecc71']
    labels_bar = ['Background\n(prob)', 'Drone\n(prob)']
    values_bar = [prob_bg, prob_drone]
    bars = ax.bar(labels_bar, values_bar, color=bar_colors, width=0.5)
    ax.axhline(0.65, color='orange', linewidth=2, linestyle='--', label='Threshold = 0.65')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probability")
    ax.set_title(f"SVM output\ndecision value = {decision:.3f}")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, values_bar):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f"{val:.3f}", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT / "stage5_svm_output.png", dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved → output/stage5_svm_output.png")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY: all stages side-by-side
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Summary figure ──")

fig = plt.figure(figsize=(18, 5))
fig.suptitle("Full Pipeline: one drone image traced through every stage",
             fontsize=13, fontweight='bold')

# 1. Raw image
ax1 = fig.add_subplot(1, 5, 1)
ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
ax1.set_title("①  Raw pixels\n64×64 uint8", fontsize=9)
ax1.axis('off')

# 2. Integral image
ax2 = fig.add_subplot(1, 5, 2)
ax2.imshow(ii, cmap='hot')
ax2.set_title("②  Integral image\n65×65 float64", fontsize=9)
ax2.axis('off')

# 3. Sample Haar response map (Type 0)
rmap0 = response_map(ii, 0, h=16, w=16, stride=4)
vabs  = np.abs(rmap0).max() or 1
ax3 = fig.add_subplot(1, 5, 3)
ax3.imshow(rmap0, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
ax3.set_title("③  Haar responses\n(Type 0, vertical edge)", fontsize=9)
ax3.axis('off')

# 4. Feature vector
ax4 = fig.add_subplot(1, 5, 4)
ax4.plot(features, linewidth=0.3, color='steelblue')
ax4.axhline(0, color='k', linewidth=0.4)
ax4.set_title(f"④  Feature vector\n{len(features)} values, L2-normed", fontsize=9)
ax4.set_xlim(0, len(features))
ax4.tick_params(labelsize=7)

# 5. SVM output (if model exists)
ax5 = fig.add_subplot(1, 5, 5)
if model_path.exists():
    ax5.bar(['BG', 'Drone'], [prob_bg, prob_drone],
            color=['#e74c3c', '#2ecc71'])
    ax5.axhline(0.65, color='orange', linestyle='--', linewidth=2)
    ax5.set_ylim(0, 1.05)
    ax5.set_title(f"⑤  SVM score\nprob(drone)={prob_drone:.3f}", fontsize=9)
    ax5.tick_params(labelsize=8)
else:
    ax5.text(0.5, 0.5, "SVM model\nnot found\n(run train_svm.py)",
             ha='center', va='center', transform=ax5.transAxes, fontsize=9)
    ax5.set_title("⑤  SVM score", fontsize=9)
    ax5.axis('off')

plt.tight_layout()
plt.savefig(OUT / "stage0_summary.png", dpi=130, bbox_inches='tight')
plt.close()
print("  Saved → output/stage0_summary.png")

print("\nDone. Files written to output/:")
for p in sorted(OUT.glob("stage*.png")):
    print(f"  {p.name}")
