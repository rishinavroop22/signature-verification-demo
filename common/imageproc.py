"""One cleaning routine shared by real, generated and uploaded signatures.

Using the same routine everywhere matters: otherwise a detector can learn the scanner
background or the file source instead of the handwriting.
"""
import cv2
import numpy as np

from common.paths import IMG_H, IMG_W

PAD = 8


def ink_map(gray: np.ndarray) -> np.ndarray:
    """uint8 gray (dark ink on light paper) -> float ink strength, 0 = paper, 1 = full ink."""
    g = gray.astype(np.uint8)
    k = int(np.clip(min(g.shape) // 7, 31, 101)) | 1
    # Paper estimate: closing removes thin dark strokes; median keeps hard band edges.
    closed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    paper = cv2.medianBlur(closed, k)
    d = paper.astype(np.float32) - g.astype(np.float32)
    d[d < 0] = 0
    strong = d[d > 12]
    if strong.size < 20:
        return np.zeros_like(d)
    top = np.percentile(strong, 90)              # per-image ink level, removes pen/scan darkness
    d = np.clip(d / max(float(top), 1.0), 0, 1)
    # Hysteresis: faint strokes survive when they touch a strong stroke; paper texture does not.
    weak = (d > 0.12).astype(np.uint8)
    n, lab = cv2.connectedComponents(weak, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[d > 0.38])] = True
    keep[0] = False
    return np.clip((d - 0.06) / 0.50, 0, 1) * keep[lab]


def drop_specks(ink: np.ndarray, min_area: int = 8) -> np.ndarray:
    mask = (ink > 0.3).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = st[i, cv2.CC_STAT_AREA] >= min_area
    m = cv2.dilate(keep[lab].astype(np.uint8), np.ones((3, 3), np.uint8))
    return ink * m


def fit_canvas(ink: np.ndarray) -> np.ndarray:
    """Crop to ink and centre on the IMG_H x IMG_W canvas, keeping the aspect ratio."""
    out = np.zeros((IMG_H, IMG_W), np.float32)
    ys, xs = np.where(ink > 0.3)
    if len(ys) < 20:
        return out
    c = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = c.shape
    s = min((IMG_H - 2 * PAD) / h, (IMG_W - 2 * PAD) / w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    c = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y, x = (IMG_H - nh) // 2, (IMG_W - nw) // 2
    out[y:y + nh, x:x + nw] = np.clip(c, 0, 1)
    return out


def canonical_ink(gray: np.ndarray) -> np.ndarray:
    """Clean any signature image -> float ink canvas (H, W) in [0, 1]."""
    return fit_canvas(drop_specks(ink_map(gray)))


def to_uint8(ink: np.ndarray) -> np.ndarray:
    """Ink canvas -> dark-on-white uint8 image."""
    return np.round(255 * (1 - np.clip(ink, 0, 1))).astype(np.uint8)


def standardise_generated(ink: np.ndarray) -> np.ndarray:
    """Generated ink canvas -> same crop/centre/speck rules as real images."""
    return fit_canvas(drop_specks(np.clip((ink - 0.06) / 0.94, 0, 1)))
