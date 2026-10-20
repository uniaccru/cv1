"""Adaptive median filter (variable kernel size) using OpenCV primitives.

Min/max come from cv2.erode / cv2.dilate, median from cv2.medianBlur.
Both use replicate borders, matching the native implementation.
"""
import cv2
import numpy as np


def adaptive_median_cv(img: np.ndarray, max_size: int = 7) -> np.ndarray:
    """img: uint8 grayscale image, max_size: odd, >= 3."""
    img = img.astype(np.uint8)
    out = img.copy()
    pending = np.ones(img.shape, dtype=bool)  # pixels not yet resolved
    zmed = img

    for size in range(3, max_size + 1, 2):
        k = np.ones((size, size), np.uint8)
        zmin = cv2.erode(img, k, borderType=cv2.BORDER_REPLICATE)
        zmax = cv2.dilate(img, k, borderType=cv2.BORDER_REPLICATE)
        zmed = cv2.medianBlur(img, size)

        # Stage A: is the median an impulse?
        stage_a = (zmin < zmed) & (zmed < zmax)
        # Stage B: is the center pixel an impulse?
        center_ok = (zmin < img) & (img < zmax)

        done = pending & stage_a
        out[done] = np.where(center_ok, img, zmed)[done]
        pending &= ~stage_a

    # Pixels still unresolved at max window size get the median of that window
    out[pending] = zmed[pending]
    return out
