import time, json
import numpy as np
import cv2
from benchmark import salt_pepper, psnr, best_time
from adaptive_median_cv import adaptive_median_cv
from adaptive_median_native import adaptive_median_native

img = cv2.imread("images/source_photo.jpeg", cv2.IMREAD_GRAYSCALE)
print("shape", img.shape)
res = {"shape": list(img.shape), "quality": {}, "speed": {}}
for d in [0.05, 0.1, 0.2, 0.3]:
    noisy = salt_pepper(img, d, seed=5)
    cvout = adaptive_median_cv(noisy)
    med3 = cv2.medianBlur(noisy, 3)
    res["quality"][d] = {
        "noisy": round(psnr(img, noisy), 2),
        "fixed_median3": round(psnr(img, med3), 2),
        "adaptive": round(psnr(img, cvout), 2),
    }
    if d == 0.2:
        nat = np.array(adaptive_median_native(noisy.tolist()), dtype=np.uint8)
        res["quality"]["native_match_p0.2"] = int(np.count_nonzero(nat != cvout))
        cv2.imwrite("real_noisy_p20.png", noisy)
        cv2.imwrite("real_adaptive_p20.png", cvout)
        cv2.imwrite("real_median3_p20.png", med3)
        t_cv, _ = best_time(adaptive_median_cv, noisy, 5)
        t = time.perf_counter(); adaptive_median_native(noisy.tolist()); t_nat = time.perf_counter() - t
        res["speed"] = {"opencv_s": round(t_cv, 5), "native_s": round(t_nat, 3), "ratio": round(t_nat / t_cv, 1)}
print(json.dumps(res, indent=2))
