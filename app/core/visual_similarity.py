from __future__ import annotations

from typing import Optional
import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def load_grayscale_image(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def resize_to_common(a: np.ndarray, b: np.ndarray, size: tuple[int, int] = (224, 224)) -> tuple[np.ndarray, np.ndarray]:
    a_resized = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    b_resized = cv2.resize(b, size, interpolation=cv2.INTER_AREA)
    return a_resized, b_resized


def structural_similarity_from_paths(path_a: str, path_b: str) -> float:
    img_a = load_grayscale_image(path_a)
    img_b = load_grayscale_image(path_b)

    if img_a is None or img_b is None:
        return 0.0

    img_a, img_b = resize_to_common(img_a, img_b)

    img_a = cv2.GaussianBlur(img_a, (3, 3), 0)
    img_b = cv2.GaussianBlur(img_b, (3, 3), 0)

    score = ssim(img_a, img_b)
    return max(0.0, min(1.0, float(score)))