import cv2
import numpy as np


def adaptive_median_cv(img: np.ndarray, max_size: int = 7) -> np.ndarray:
    """img: uint8 grayscale image, max_size: odd, >= 3."""
    img = img.astype(np.uint8)
    out = img.copy()
    pending = np.ones(img.shape, dtype=bool)
    zmed = img

    for size in range(3, max_size + 1, 2):
        k = np.ones((size, size), np.uint8)
        zmin = cv2.erode(img, k, borderType=cv2.BORDER_REPLICATE)
        zmax = cv2.dilate(img, k, borderType=cv2.BORDER_REPLICATE)
        zmed = cv2.medianBlur(img, size)

        stage_a = (zmin < zmed) & (zmed < zmax)

        center_ok = (zmin < img) & (img < zmax)

        done = pending & stage_a
        out[done] = np.where(center_ok, img, zmed)[done]
        pending &= ~stage_a

    out[pending] = zmed[pending]
    return out
