"""Generate test image, add salt-and-pepper noise, filter with both variants,
check consistency, measure PSNR and speed."""
import time
import json
import numpy as np
import cv2
from adaptive_median_cv import adaptive_median_cv
from adaptive_median_native import adaptive_median_native


def make_clean(size, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    img = 110 + 60 * np.sin(x / size * 3 * np.pi) * np.cos(y / size * 2 * np.pi)
    cv2.circle(img, (size // 3, size // 3), size // 6, 220, -1)
    cv2.rectangle(img, (size // 2, size // 2), (size - 8, size - 8), 40, -1)
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255)
    return img.astype(np.uint8)


def salt_pepper(img, density, seed=1):
    rng = np.random.default_rng(seed)
    noisy = img.copy()
    r = rng.random(img.shape)
    noisy[r < density / 2] = 0
    noisy[(r >= density / 2) & (r < density)] = 255
    return noisy


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def best_time(fn, arg, repeat):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        res = fn(arg)
        best = min(best, time.perf_counter() - t)
    return best, res


if __name__ == "__main__":
    results = {}

    clean = make_clean(256)
    noisy = salt_pepper(clean, 0.2)
    out_cv = adaptive_median_cv(noisy)
    out_nat = np.array(adaptive_median_native(noisy.tolist()), dtype=np.uint8)
    med3 = cv2.medianBlur(noisy, 3)
    results["quality_256_p20"] = {
        "psnr_noisy": round(psnr(clean, noisy), 2),
        "psnr_fixed_median3": round(psnr(clean, med3), 2),
        "psnr_adaptive_cv": round(psnr(clean, out_cv), 2),
        "psnr_adaptive_native": round(psnr(clean, out_nat), 2),
        "pixels_differ_cv_vs_native": int(np.count_nonzero(out_cv != out_nat)),
    }
    cv2.imwrite("noisy.png", noisy)
    cv2.imwrite("adaptive_cv.png", out_cv)
    cv2.imwrite("adaptive_native.png", out_nat)

    results["quality_vs_density_256"] = {}
    for d in [0.05, 0.1, 0.2, 0.3, 0.4]:
        n = salt_pepper(clean, d, seed=2)
        results["quality_vs_density_256"][d] = {
            "noisy": round(psnr(clean, n), 2),
            "fixed_median3": round(psnr(clean, cv2.medianBlur(n, 3)), 2),
            "adaptive": round(psnr(clean, adaptive_median_cv(n)), 2),
        }

    results["speed"] = {}
    for s in [64, 128, 256, 512]:
        img = salt_pepper(make_clean(s), 0.2, seed=3)
        lst = img.tolist()
        t_cv, _ = best_time(adaptive_median_cv, img, repeat=5)
        if s <= 256:
            t_nat, _ = best_time(adaptive_median_native, lst, repeat=1)
        else:
            t_nat = None 
        if s == 512:
            t = time.perf_counter()
            adaptive_median_native(lst)
            t_nat = time.perf_counter() - t
        results["speed"][s] = {
            "opencv_s": round(t_cv, 5),
            "native_s": round(t_nat, 3),
            "ratio_native_over_cv": round(t_nat / t_cv, 1),
        }

    print(json.dumps(results, indent=2, ensure_ascii=False))
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
