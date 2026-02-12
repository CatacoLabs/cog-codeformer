#!/usr/bin/env python3
"""Apply randomized degradation pipeline to clean portrait images for benchmarking."""

import cv2
import numpy as np
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).parent
NUM_IMAGES = 100


def degrade_image(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    h, w = img.shape[:2]

    # 1. Gaussian Blur — kernel size from [3, 11], odd only
    k = rng.choice([3, 5, 7, 9, 11])
    img = cv2.GaussianBlur(img, (k, k), 0)

    # 2. Downscale + Upscale — factor from [2, 4]
    scale = rng.choice([2, 3, 4])
    small = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 3. Gaussian Noise — sigma from [5, 30]
    sigma = rng.uniform(5, 30)
    noise = rng.randn(*img.shape) * sigma
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    # 4. JPEG Compression — quality from [15, 60]
    quality = rng.randint(15, 61)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return img


def main():
    rng = np.random.RandomState(SEED)
    for i in range(1, NUM_IMAGES + 1):
        path = DATA_DIR / f"{i}.jpg"
        img = cv2.imread(str(path))
        if img is None:
            print(f"SKIP {path} (not found)")
            continue
        img = degrade_image(img, rng)
        cv2.imwrite(str(path), img)
        print(f"Degraded {path.name}")


if __name__ == "__main__":
    main()
