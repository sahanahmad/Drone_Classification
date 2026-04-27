"""
train_svm.py
------------
Trains an SVM classifier on Haar features extracted from drone/background patches.

Follows the FPCV-2-5 SVM pipeline:
  - Feature vectors f for each training window (slide 11)
  - Labels: +1 for drone, -1 for background
  - SVM finds the decision boundary with maximum margin (slides 42–48)
  - The resulting w, b, and support vectors define the classifier (slide 48)

We use sklearn's SVC which implements the same soft-margin SVM described
in [Cortes 1995] as cited in FPCV-2-5.
"""

import numpy as np
import cv2
import os
import pickle
from pathlib import Path

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from haar_features import extract_features_batch, get_templates

WINDOW_SIZE = 64
HAAR_STEP   = 8


# ── Load images from directory ─────────────────────────────────────────────────

def load_images(directory: str, max_count: int = None) -> list:
    """Load all .png/.jpg images from directory as grayscale numpy arrays."""
    paths = sorted(Path(directory).glob("*.png"))
    if max_count:
        paths = paths[:max_count]

    images = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if img.shape != (WINDOW_SIZE, WINDOW_SIZE):
                img = cv2.resize(img, (WINDOW_SIZE, WINDOW_SIZE))
            images.append(img)
    return images


# ── Feature extraction ─────────────────────────────────────────────────────────

def build_feature_matrix(pos_dir: str, neg_dir: str):
    """
    Load positive and negative images, extract Haar features, return X, y.

    X: (N_total, n_features) float64
    y: (N_total,)  +1 for drone, -1 for background
    """
    print("Loading positive (drone) images...")
    pos_imgs = load_images(pos_dir)
    print(f"  {len(pos_imgs)} positives loaded")

    print("Loading negative (background) images...")
    neg_imgs = load_images(neg_dir)
    print(f"  {len(neg_imgs)} negatives loaded")

    print("Extracting Haar features from positives...")
    X_pos = extract_features_batch(pos_imgs, WINDOW_SIZE, HAAR_STEP)

    print("Extracting Haar features from negatives...")
    X_neg = extract_features_batch(neg_imgs, WINDOW_SIZE, HAAR_STEP)

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([
        np.ones(len(X_pos),  dtype=np.int32),
        -np.ones(len(X_neg), dtype=np.int32)
    ])

    print(f"\nTotal samples: {len(y)}  |  Features per sample: {X.shape[1]}")
    return X, y


# ── Train ──────────────────────────────────────────────────────────────────────

def train(pos_dir: str, neg_dir: str, model_path: str):
    """
    Full training pipeline:
      1. Build feature matrix
      2. Scale features (zero-mean, unit-variance)
      3. Train SVM with RBF kernel
      4. Evaluate on held-out test split
      5. Save model + scaler to disk
    """
    X, y = build_feature_matrix(pos_dir, neg_dir)

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Feature scaling — required before SVM (makes margin computation meaningful)
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Train SVM
    # C=1.0: soft-margin slack (allows some misclassification, avoids overfitting)
    # kernel='rbf': handles non-linear boundaries via kernel trick
    # class_weight='balanced': compensates for any class imbalance
    print("Training SVM (this may take 30–60 seconds)...")
    svm = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,      # enables predict_proba for detection scores
        random_state=42
    )
    svm.fit(X_train_sc, y_train)

    # Evaluate
    y_pred = svm.predict(X_test_sc)
    print("\n── Classification Report ──")
    print(classification_report(
        y_test, y_pred,
        target_names=["background (-1)", "drone (+1)"]))

    print("── Confusion Matrix ──")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    n_sv = svm.n_support_
    print(f"\nSupport vectors: {n_sv[0]} background, {n_sv[1]} drone")
    print("(These are the training samples that define the decision boundary —")
    print(" exactly the 'support vectors' described in FPCV-2-5 slide 45)")

    # Save
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({'svm': svm, 'scaler': scaler}, f)
    print(f"\nModel saved to: {model_path}")

    return svm, scaler


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    train(
        pos_dir   = str(base / "data" / "positives"),
        neg_dir   = str(base / "data" / "negatives"),
        model_path= str(base / "output" / "drone_svm.pkl")
    )
