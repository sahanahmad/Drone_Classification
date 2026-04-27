"""
visualize_rbf.py
----------------
Deep-dive visualizations for the RBF kernel SVM stage.

Panels produced:
  rbf1_kernel_function.png   — What the RBF kernel formula actually computes
  rbf2_decision_boundary.png — 2D PCA projection of support vectors + boundary
  rbf3_support_vectors.png   — The 138 support vector patches as images
  rbf4_kernel_similarity.png — Kernel similarity from one test image to every SV
  rbf5_decision_value.png    — How the final score is assembled from SVs

Run: python src/visualize_rbf.py
"""

import sys, pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from haar_features import extract_features, get_templates

ROOT  = Path(__file__).parent.parent
OUT   = ROOT / "output"
OUT.mkdir(exist_ok=True)

# ── Load model + one test image ───────────────────────────────────────────────
with open(ROOT / "output" / "drone_svm.pkl", "rb") as f:
    bundle = pickle.load(f)
svm    = bundle["svm"]
scaler = bundle["scaler"]

img = cv2.imread(str(ROOT / "data" / "positives" / "drone_0000.png"),
                 cv2.IMREAD_GRAYSCALE)
templates = get_templates(64, 8)
feat      = extract_features(img, templates)
feat_sc   = scaler.transform(feat.reshape(1, -1))   # shape (1, 4212)

gamma     = svm._gamma          # computed gamma ('scale' mode)
SVs       = svm.support_vectors_   # (138, 4212)
dual_coef = svm.dual_coef_[0]      # (138,)  αᵢ·yᵢ
bias      = svm.intercept_[0]      # b

drone_class_idx = list(svm.classes_).index(1)
n_bg    = svm.n_support_[0]   # 22
n_drone = svm.n_support_[1]   # 116

print(f"gamma={gamma:.6f}  C={svm.C}  bias={bias:.4f}")
print(f"Support vectors: {n_bg} background, {n_drone} drone  (total={len(SVs)})")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — What the RBF kernel formula computes
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("RBF Kernel — What the Formula Actually Computes",
             fontsize=13, fontweight='bold')

# 1a. K(x, sv) = exp(-γ ||x - sv||²) as a function of distance
ax = axes[0]
dist2 = np.linspace(0, 60000, 400)
for g, lbl in [(gamma, f"γ={gamma:.4f} (this model)"),
               (gamma*5,  f"γ×5 (sharper)"),
               (gamma/5,  f"γ/5 (smoother)")]:
    ax.plot(dist2, np.exp(-g * dist2), label=lbl)
ax.set_xlabel("||x − sv||²  (squared distance)")
ax.set_ylabel("K(x, sv)")
ax.set_title("K(x,sv) = exp(−γ‖x−sv‖²)\nvaries with distance")
ax.legend(fontsize=7)
ax.set_ylim(-0.05, 1.05)

# 1b. K(x, sv) for our test image vs every support vector
dist2_to_svs = np.sum((feat_sc - SVs)**2, axis=1)   # (138,)
k_vals       = np.exp(-gamma * dist2_to_svs)         # (138,)

ax = axes[1]
colors = ['#e74c3c']*n_bg + ['#2ecc71']*n_drone
ax.bar(range(len(k_vals)), k_vals, color=colors, edgecolor='none', width=1.0)
ax.axvline(n_bg - 0.5, color='k', linestyle='--', linewidth=1)
bg_patch    = plt.matplotlib.patches.Patch(color='#e74c3c', label=f'BG SVs ({n_bg})')
drone_patch = plt.matplotlib.patches.Patch(color='#2ecc71', label=f'Drone SVs ({n_drone})')
ax.legend(handles=[bg_patch, drone_patch], fontsize=8)
ax.set_xlabel("Support vector index")
ax.set_ylabel("K(test, sv)")
ax.set_title(f"Kernel similarity from test image\nto each of the {len(SVs)} support vectors")

# 1c. Weighted contributions: αᵢ·yᵢ·K(x, svᵢ)
weighted = dual_coef * k_vals
ax = axes[2]
ax.bar(range(len(weighted)), weighted,
       color=['#e74c3c' if v < 0 else '#2ecc71' for v in weighted],
       edgecolor='none', width=1.0)
ax.axhline(0, color='k', linewidth=0.8)
ax.axvline(n_bg - 0.5, color='k', linestyle='--', linewidth=1)
total = weighted.sum() + bias
ax.set_xlabel("Support vector index")
ax.set_ylabel("αᵢ·yᵢ · K(x, svᵢ)")
ax.set_title(f"Weighted contributions\nSum + bias = {total:.4f}  → "
             f"{'drone ✓' if total > 0 else 'background'}")

plt.tight_layout()
plt.savefig(OUT / "rbf1_kernel_function.png", dpi=120, bbox_inches='tight')
plt.close()
print("Saved → rbf1_kernel_function.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — 2D PCA projection: decision boundary + support vectors
# ══════════════════════════════════════════════════════════════════════════════

# Build a small dataset to project: SVs + our test image + some random samples
pos_dir = ROOT / "data" / "positives"
neg_dir = ROOT / "data" / "negatives"

def load_feats(directory, n=80):
    paths = sorted(directory.glob("*.png"))[:n]
    imgs  = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    feats = np.vstack([extract_features(im, templates).reshape(1,-1) for im in imgs])
    return scaler.transform(feats)

print("Loading samples for PCA projection...")
X_pos = load_feats(pos_dir, 80)   # (80, 4212) — drone samples
X_neg = load_feats(neg_dir, 80)   # (80, 4212) — background samples

all_X  = np.vstack([SVs, X_pos, X_neg, feat_sc])
labels = (['sv_bg']  * n_bg +
          ['sv_drone'] * n_drone +
          ['drone'] * len(X_pos) +
          ['bg']    * len(X_neg) +
          ['test'])

pca = PCA(n_components=2, random_state=42)
all_2d = pca.fit_transform(all_X)
print(f"PCA variance explained: {pca.explained_variance_ratio_*100}")

# Decision boundary grid in PCA space
x_min, x_max = all_2d[:,0].min()-1, all_2d[:,0].max()+1
y_min, y_max = all_2d[:,1].min()-1, all_2d[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120),
                     np.linspace(y_min, y_max, 120))
grid_pca = np.c_[xx.ravel(), yy.ravel()]
grid_orig = pca.inverse_transform(grid_pca)
grid_score = svm.decision_function(grid_orig).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle("SVM Decision Boundary (2D PCA Projection of 4,212-D Feature Space)",
             fontsize=12, fontweight='bold')

# Background decision surface
cf = ax.contourf(xx, yy, grid_score, levels=20, cmap='RdYlGn', alpha=0.4)
ax.contour(xx, yy, grid_score, levels=[0], colors='black', linewidths=2)
ax.contour(xx, yy, grid_score, levels=[-1, 1], colors='black',
           linewidths=1, linestyles='--')
plt.colorbar(cf, ax=ax, label='Decision function value')

# Plot points
split = n_bg + n_drone
idx = {
    'sv_bg':    (slice(0, n_bg),               '#c0392b', 200, 'x', f'BG support vectors ({n_bg})'),
    'sv_drone': (slice(n_bg, split),            '#27ae60', 200, 'x', f'Drone support vectors ({n_drone})'),
    'drone':    (slice(split, split+80),        '#2ecc71',  40, 'o', 'Drone samples'),
    'bg':       (slice(split+80, split+160),    '#e74c3c',  40, 'o', 'Background samples'),
}
for key, (sl, color, size, marker, label) in idx.items():
    pts = all_2d[sl]
    ax.scatter(pts[:,0], pts[:,1], c=color, s=size, marker=marker,
               label=label, zorder=3, alpha=0.85,
               linewidths=1.5 if marker=='x' else 0)

# Test image
test_2d = all_2d[-1]
ax.scatter(*test_2d, c='yellow', s=250, marker='*', zorder=5,
           edgecolors='black', linewidths=1, label='Test image (drone_0000)')
ax.annotate(f" decision={total:.2f}", test_2d, fontsize=8,
            xytext=(test_2d[0]+0.3, test_2d[1]+0.3))

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("Green region = drone side  |  Red region = background side\n"
             "Dashed lines = margin  |  Solid line = decision boundary (f=0)")
ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig(OUT / "rbf2_decision_boundary.png", dpi=130, bbox_inches='tight')
plt.close()
print("Saved → rbf2_decision_boundary.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Support vector patches as actual images
# ══════════════════════════════════════════════════════════════════════════════

# Inverse-transform SVs back to pixel space (approximate — not perfect)
SVs_feat_space = scaler.inverse_transform(SVs)   # undo StandardScaler

# Load actual training images to find nearest neighbour to each SV
print("Loading training images to match support vectors...")
pos_paths = sorted((ROOT / "data" / "positives").glob("*.png"))
neg_paths = sorted((ROOT / "data" / "negatives").glob("*.png"))
all_paths = neg_paths + pos_paths   # neg first to match SV ordering

all_imgs   = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in all_paths]
all_feats  = np.vstack([extract_features(im, templates).reshape(1,-1) for im in all_imgs])
all_feats_sc = scaler.transform(all_feats)

# For each SV, find the closest training image by L2 distance in feature space
sv_imgs = []
for sv in SVs:
    dists = np.sum((all_feats_sc - sv)**2, axis=1)
    nn_idx = np.argmin(dists)
    sv_imgs.append(all_imgs[nn_idx])

# Show first 22 bg SVs and first 30 drone SVs
n_show_bg    = min(n_bg, 22)
n_show_drone = min(n_drone, 30)

fig, axes = plt.subplots(5, 11, figsize=(16, 8))
fig.suptitle(f"Support Vector Patches — The {len(SVs)} Training Images That Define the Boundary\n"
             f"Top rows: {n_bg} Background SVs  |  Bottom rows: {n_drone} Drone SVs",
             fontsize=11, fontweight='bold')

for i, ax in enumerate(axes.flat):
    ax.axis('off')

# BG SVs (first 22 = top 2 rows)
for i in range(n_show_bg):
    ax = axes[i // 11][i % 11]
    ax.imshow(sv_imgs[i], cmap='gray', vmin=0, vmax=255)
    ax.set_title(f"bg{i}\nα={dual_coef[i]:.3f}", fontsize=5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#e74c3c'); spine.set_linewidth(2)

# Drone SVs (next rows)
for i in range(n_show_drone):
    sv_i = n_bg + i
    row  = 2 + i // 11
    col  = i % 11
    if row < 5:
        ax = axes[row][col]
        ax.imshow(sv_imgs[sv_i], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"dr{i}\nα={dual_coef[sv_i]:.3f}", fontsize=5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2ecc71'); spine.set_linewidth(2)

# Row labels
axes[0][0].set_ylabel("BG SVs", fontsize=9, color='#e74c3c')
axes[2][0].set_ylabel("Drone SVs", fontsize=9, color='#2ecc71')

plt.tight_layout()
plt.savefig(OUT / "rbf3_support_vectors.png", dpi=120, bbox_inches='tight')
plt.close()
print("Saved → rbf3_support_vectors.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Step-by-step decision value assembly
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("How the Final Decision Value Is Assembled from Support Vectors",
             fontsize=12, fontweight='bold')

# 4a. Cumulative sum of weighted kernel values
sorted_idx  = np.argsort(np.abs(weighted))[::-1]
cumulative  = np.cumsum(weighted[sorted_idx])

ax = axes[0]
ax.plot(cumulative, color='steelblue', linewidth=2)
ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
ax.axhline(-bias, color='orange', linewidth=1.5, linestyle='--',
           label=f'−bias = {-bias:.3f}')
ax.axhline(total, color='green', linewidth=2,
           label=f'Final f = {total:.4f}')
ax.set_xlabel("Support vectors added (sorted by |contribution|)")
ax.set_ylabel("Cumulative Σ αᵢyᵢ K(x,svᵢ)")
ax.set_title("Cumulative decision value\nas SVs are added one by one")
ax.legend(fontsize=8)

# 4b. Dual coefficients (α·y) for all SVs
ax = axes[1]
ax.bar(range(n_bg),   dual_coef[:n_bg],    color='#e74c3c', label='BG SVs')
ax.bar(range(n_bg, len(dual_coef)), dual_coef[n_bg:], color='#2ecc71', label='Drone SVs')
ax.axhline(0, color='k', linewidth=0.8)
ax.set_xlabel("Support vector index")
ax.set_ylabel("αᵢ · yᵢ  (dual coefficient)")
ax.set_title("Dual coefficients αᵢ·yᵢ\n(learned weights, bounded by C=1.0)")
ax.legend(fontsize=8)
ax.axvline(n_bg - 0.5, color='k', linestyle='--', linewidth=1)

# 4c. Final score breakdown as a waterfall-style bar
ax = axes[2]
contrib_bg    = weighted[:n_bg].sum()
contrib_drone = weighted[n_bg:].sum()
final_score   = contrib_bg + contrib_drone + bias

categories = ['BG SVs\ncontribution', 'Drone SVs\ncontribution', 'Bias (b)', 'Final\ndecision f']
values     = [contrib_bg, contrib_drone, bias, final_score]
colors_bar = ['#e74c3c', '#2ecc71', '#f39c12',
              '#27ae60' if final_score > 0 else '#c0392b']
bars = ax.bar(categories, values, color=colors_bar)
ax.axhline(0, color='k', linewidth=1)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + (0.02 if val >= 0 else -0.06),
            f"{val:.3f}", ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel("Value")
ax.set_title(f"Decision value breakdown\nf = Σ(BG) + Σ(Drone) + b = {final_score:.4f}\n"
             f"→ {'drone (f>0) ✓' if final_score > 0 else 'background (f<0)'}")

plt.tight_layout()
plt.savefig(OUT / "rbf4_decision_value.png", dpi=120, bbox_inches='tight')
plt.close()
print("Saved → rbf4_decision_value.png")

print("\nAll RBF visualizations saved to output/:")
for p in sorted(OUT.glob("rbf*.png")):
    print(f"  {p.name}")
