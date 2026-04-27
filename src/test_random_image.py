"""
test_random_image.py
--------------------
Picks a random image from the real test dataset, runs the detector,
and saves the result to output/real_test_result.png.

Run: python src/test_random_image.py
"""

import sys, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from detector import detect

TEST_DIR   = Path(__file__).parent.parent / "drone_dataset" / "drone_yolov8" / "test" / "images"
MODEL_PATH = Path(__file__).parent.parent / "output" / "drone_svm.pkl"
OUT_PATH   = Path(__file__).parent.parent / "output" / "real_test_result.png"

img_paths = list(TEST_DIR.glob("*.jpg")) + list(TEST_DIR.glob("*.png"))
chosen    = random.choice(img_paths)

print(f"Selected: {chosen.name}")
detect(str(chosen), str(MODEL_PATH), str(OUT_PATH))
print(f"Result saved → output/real_test_result.png")
