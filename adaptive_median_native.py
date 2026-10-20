def _pad_replicate(img, r):
    h, w = len(img), len(img[0])
    padded = []
    for y in range(h + 2 * r):
        sy = min(max(y - r, 0), h - 1)
        src = img[sy]
        row = [src[min(max(x - r, 0), w - 1)] for x in range(w + 2 * r)]
        padded.append(row)
    return padded


def adaptive_median_native(img, max_size=7):
    h, w = len(img), len(img[0])
    r_max = max_size // 2
    pad = _pad_replicate(img, r_max)
    out = [[0] * w for _ in range(h)]

    for y in range(h):
        cy = y + r_max
        for x in range(w):
            cx = x + r_max
            z = pad[cy][cx]
            size = 3
            while True:
                r = size // 2
                win = []
                for j in range(cy - r, cy + r + 1):
                    win.extend(pad[j][cx - r: cx + r + 1])
                win.sort()
                zmin = win[0]
                zmax = win[-1]
                zmed = win[len(win) // 2]

                if zmin < zmed < zmax:
                    out[y][x] = z if zmin < z < zmax else zmed
                    break

                if size >= max_size:
                    out[y][x] = zmed 
                    break
                size += 2
    return out
