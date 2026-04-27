# Drone Silhouette Detector — Haar Feature + SVM

A classical computer vision pipeline for detecting quadcopter drone silhouettes in images, built entirely from first principles following Prof. Shree K. Nayar's [First Principles of Computer Vision (FPCV-2-5)](https://fpcv.cs.columbia.edu/) lecture series at Columbia University.

> **Key insight:** A face detector and a drone detector share the exact same pipeline structure. The only difference is what images you train on. The mathematics — Haar filters, integral images, and SVMs — are identical.

---

## What This Project Demonstrates

- Haar filter-based feature extraction using integral images
- SVM training with a maximum-margin decision boundary
- Multi-scale sliding window detection
- Non-maximal suppression (NMS) to refine detections
- End-to-end classical CV pipeline applied to drone silhouettes as a defense-relevant target

---

## Project Structure

```
drone_detector/
├── src/
│   ├── data_generator.py   ← creates synthetic training images
│   ├── haar_features.py    ← integral image + Haar filters
│   ├── train_svm.py        ← trains the SVM classifier
│   ├── detector.py         ← sliding window + NMS
│   └── run_pipeline.py     ← runs everything end-to-end
├── data/                   ← created automatically on first run
│   ├── positives/          ← drone silhouette training images
│   ├── negatives/          ← background patch training images
│   └── test/               ← test scene images
└── output/                 ← created automatically on first run
    ├── drone_svm.pkl        ← saved trained model
    ├── detection_result.png ← annotated test image
    └── pipeline_summary.png ← 8-panel visualization
```

The `data/` and `output/` folders do not exist when you first clone the repo. They are created automatically the first time you run `run_pipeline.py`.

---

## The Detection Pipeline

The pipeline has five sequential steps, all orchestrated by `run_pipeline.py`:

| Step | File | What it does |
|------|------|--------------|
| 1 | `data_generator.py` | Synthesizes drone + background training images |
| 2 | `haar_features.py` | Computes integral images and extracts Haar features |
| 3 | `train_svm.py` | Trains SVM classifier on the feature vectors |
| 4 | `detector.py` | Sliding window detection + non-maximal suppression |
| 5 | `run_pipeline.py` | Orchestrates steps 1–4 and generates visualizations |

### Step 1 — Synthetic Data Generation
Generates 64×64 grayscale training images: procedurally drawn quadcopter silhouettes (positive samples) with random rotation, scale, and Gaussian noise, and four types of background texture patches (negative samples).

### Step 2 — Haar Feature Extraction
Builds an integral image from each patch — a lookup table that enables any rectangular region sum in O(1) time — then applies four Haar filter types (horizontal edge, vertical edge, dark bar, checkerboard) at every valid position and scale. Each 64×64 window produces a 4,212-dimensional feature vector.

### Step 3 — SVM Training
The SVM finds the maximum-margin hyperplane separating drone feature vectors (+1) from background vectors (−1) in 4,212-dimensional space. Features are scaled to zero mean and unit variance before training. The trained model is saved to `output/drone_svm.pkl`.

### Step 4 — Sliding Window Detection
A 64×64 window scans the test image at stride=8 across three scales (1.0×, 0.75×, 0.5×). At each position, Haar features are extracted and fed to the SVM. Detections with confidence ≥ 0.60 are kept, then collapsed by NMS (IoU threshold = 0.30).

---

## How to Run

```bash
# Create virtual environment
python3 -m venv drone_env
source drone_env/bin/activate

# Install dependencies
pip install numpy scikit-learn scikit-image matplotlib opencv-python

# Clone the repo and navigate into it
git clone <your-repo-url>
cd drone_detector

# Run the full pipeline
python src/run_pipeline.py
```

> **macOS note:** Use `opencv-python`, not `opencv-python-headless`. The headless version strips the display backend and can cause issues on Mac.

Expected runtime on a Mac Mini M2: approximately 2–4 minutes. SVM training is the slowest step.

---

## Pipeline Visualization

A core part of understanding this project is seeing what happens at each stage. Running `run_pipeline.py` produces an **8-panel summary figure** (`output/pipeline_summary.png`) that visualizes:

1. The synthesized test scene (sky background with embedded drone silhouettes)
2. The integral image of the scene
3. The four Haar filter types applied to a sample patch
4. The Haar response map across the image
5. The SVM decision boundary (projected from high-dimensional space)
6. Raw detections before NMS (many overlapping boxes)
7. Post-NMS detections (one clean box per drone)
8. Sample positive and negative training patches

This visualization was intentionally built to support a **data-first learning approach** — tracing what happens to a single image through every stage of the pipeline. Rather than reading the math abstractly, you can observe the integral image as a cumulative sum table, watch Haar filters respond to drone edges, and see NMS collapse redundant detections into clean bounding boxes.

---

## Why This Works on Synthetic Data — But Not on Real Images

The detector achieves **100% accuracy on synthetic test data**. This is expected and is not a measure of real-world performance. Here is why the gap exists:

### Why it works on synthetic data
The synthetic training images and test scenes are generated by the same `data_generator.py` function. The drone silhouettes in the test scene are drawn using the same procedural shape, the same scale range, and placed on similar background textures as the training positives. The SVM has essentially seen every visual variation it will be tested on. The problem is well-posed and closed.

### Why it fails on real images

**1. Domain gap — the core problem**
The Haar features learned from clean procedural silhouettes do not generalize to real drone images, which contain texture, surface detail, motion blur, compression artifacts, and complex lighting. The SVM decision boundary is tuned to synthetic feature distributions that do not match the real-world distribution.

**2. Background complexity**
Real backgrounds — sky gradients, tree lines, buildings — produce Haar responses that closely resemble drone responses. The synthetic negatives (noise, blobs, gradients) are too simple to teach the classifier what real confusing backgrounds look like.

**3. Viewpoint and appearance variation**
Synthetic drones are always rendered from the same canonical top-down silhouette view. Real drones appear at arbitrary angles, in partial occlusion, at varying lighting conditions, and with different models that have different physical shapes.

**4. Scale mismatch**
The detector scans at three fixed scales. Real drones at long range may be only a few pixels across — far smaller than the 64×64 window can meaningfully process.

**5. No hard negative mining**
Hard negatives are background patches that the detector incorrectly flags as drones — the most confusing cases. This project does not include a hard negative mining step, which is the standard method for reducing false positives in classical detectors.

### The path forward
To improve real-world performance, the next steps are: replace synthetic positives with real labeled drone images (e.g., from Roboflow Universe or MAV-VID), perform hard negative mining, and eventually compare against a modern deep learning approach (YOLOv8 or MobileNet-SSD) on the same dataset.

---

## Connection to FPCV-2-5 Lecture Slides

Every component maps directly to a specific concept in Prof. Nayar's lecture:

| Concept | Slide(s) | Where in code |
|---------|----------|---------------|
| Sliding window detection framework | 10–11 | `detector.py → detect_at_scale()` |
| Haar feature response = white − black | 17 | `haar_features.py → compute_haar_response()` |
| Integral image definition | 21 | `haar_features.py → compute_integral_image()` |
| Rectangle sum using 3 additions | 23 | `haar_features.py → rect_sum()` |
| Integral image raster scan computation | 26 | `haar_features.py → compute_integral_image()` |
| Decision boundary with maximum margin | 42–44 | `train_svm.py → SVC(kernel='rbf')` |
| Support vectors define the boundary | 45 | `svm.support_vectors_` |
| Non-maximal suppression | FPCV-2-1 slide 49 | `detector.py → non_maximal_suppression()` |
| Multi-scale detection | 16 | `detector.py → SCALES = [1.0, 0.75, 0.5]` |

---

## Glossary

| Term | Definition |
|------|------------|
| Integral image | A pre-computed table where each cell stores the sum of all pixel values above-left of that position. Enables any rectangle sum in O(1) time. |
| Haar filter | A rectangular filter whose response is the sum of white-region pixels minus black-region pixels. |
| Feature vector | A 1D array of 4,212 Haar filter responses that describes a single 64×64 image window. |
| SVM | Support Vector Machine. A classifier that finds the decision boundary with the largest margin between two classes. |
| Support vectors | The training samples that lie closest to the decision boundary and define it. |
| Sliding window | Moving a fixed-size detection window across an image in a raster scan, running the classifier at every position. |
| Non-maximal suppression (NMS) | Collapsing multiple overlapping detections of the same object into a single best detection. |
| IoU | Intersection over Union. A measure of how much two bounding boxes overlap. |
| Hard negatives | Background patches the detector incorrectly flags as targets. Adding these to training reduces false positives. |
| Domain gap | The mismatch between the distribution of training data (synthetic) and real-world test data. |

---

## References

- Nayar, S.K. *First Principles of Computer Vision*, Lecture FPCV-2-5. Columbia University. [https://fpcv.cs.columbia.edu/](https://fpcv.cs.columbia.edu/)
- Viola, P. & Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade of Simple Features*. CVPR 2001.

---

*Built as a learning project following Nayar FPCV-2-5 — First Principles of Computer Vision, Columbia University.*