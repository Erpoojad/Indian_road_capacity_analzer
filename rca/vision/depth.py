"""
rca.vision.depth
================
Optional MiDaS pothole depth INDICATOR (ported from the earlier version's
core.py, same method and bands). MiDaS_small runs with ONNX Runtime
(exported from the official v2.1 small weights; output matches PyTorch to
5e-7 relative), so no PyTorch is needed on the server. The session is
created for one request and released, to keep the free server's memory low.

What MiDaS gives: RELATIVE inverse depth (bigger = nearer the camera), known
only up to an unknown scale and shift. It can say "this patch lies further
from the camera than the road around it", never "this pothole is 40 mm deep".
The output is an INDICATOR, kept separate from the IRC:82 size class
(width-based), and never in mm.

Method, per pothole box:
 1. Take a ring of road around the box (other potholes masked out).
 2. The road surface is roughly a plane, and in inverse depth a plane is
    linear in image x, y. Fit d = a + b x + c y to the ring (least squares,
    refit once without outliers). This removes the near-to-far gradient.
 3. Residual inside the box: r = plane - d  (> 0 = further than the road
    surface there = a depression).
 4. Noise of the road surface: sigma = 1.4826 x MAD of ring residuals.
 5. Score = 75th percentile of r inside / sigma. Area % = share of box
    pixels with r > 2 sigma.
Classes (UNCALIBRATED; calibrate with ruler depths from site):
not_distinct < 2 <= shallow < 5 <= deep. "not_distinct" means the photo
cannot separate it from flat road: it may be shallow, or a patch/stain.
"""
from __future__ import annotations

import gc
import os
from typing import List, Sequence

import cv2
import numpy as np

DEPTH_BANDS = [(2.0, "not_distinct"), (5.0, "shallow"), (float("inf"), "deep")]
DEPTH_RING = 0.6          # ring width as a share of the box size
DEPTH_NOTE = ("MiDaS relative depth indicator: how clearly the pothole lies below the "
              "surrounding road surface in the photo. Unitless and uncalibrated; not a depth in mm. "
              "IRC:82 severity still needs a depth measured on site.")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _multiple(x: float, m: int = 32, max_val: int = 256) -> int:
    """MiDaS Resize.constrain_to_multiple_of."""
    y = int(np.round(x / m) * m)
    if y > max_val:
        y = int(np.floor(x / m) * m)
    if y < m:
        y = int(np.ceil(x / m) * m)
    return y


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """MiDaS small_transform: keep aspect, 'upper_bound' 256, multiple of 32,
    bicubic, ImageNet normalisation, NCHW float32."""
    h, w = img_bgr.shape[:2]
    s = min(256 / h, 256 / w)                      # upper_bound: fit inside 256 x 256
    nh, nw = _multiple(h * s), _multiple(w * s)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def depth_map(img_bgr: np.ndarray, model_path: str) -> np.ndarray:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    try:
        d = sess.run(None, {sess.get_inputs()[0].name: preprocess(img_bgr)})[0][0]
    finally:
        del sess
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
    return cv2.resize(d, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)


def depth_indicator(dmap: np.ndarray, box_xyxy, other_boxes=()) -> dict:
    """Plane-corrected depression score for one box on a MiDaS inverse-depth map."""
    h, w = dmap.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    empty = {"depth_class": "not_distinct", "depth_score": 0.0, "depression_area_pct": 0.0,
             "reliable": False}
    if x2 - x1 < 4 or y2 - y1 < 4:
        return empty
    mx = max(4, int((x2 - x1) * DEPTH_RING))
    my = max(4, int((y2 - y1) * DEPTH_RING))
    rx1, ry1, rx2, ry2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
    yy, xx = np.mgrid[ry1:ry2, rx1:rx2]
    patch = dmap[ry1:ry2, rx1:rx2].astype(np.float64)
    inside = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
    ring = ~inside
    for ob in other_boxes:
        ox1, oy1, ox2, oy2 = ob
        ring &= ~((xx >= ox1) & (xx < ox2) & (yy >= oy1) & (yy < oy2))
    if ring.sum() < 30:
        return empty
    A = np.column_stack([np.ones(ring.sum()), xx[ring], yy[ring]])
    d = patch[ring]
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    res = d - A @ coef
    mad = np.median(np.abs(res - np.median(res)))
    keep = np.abs(res) <= 3 * 1.4826 * mad + 1e-12
    if keep.sum() >= 30:
        coef, *_ = np.linalg.lstsq(A[keep], d[keep], rcond=None)
        res = d[keep] - A[keep] @ coef
    sigma = 1.4826 * np.median(np.abs(res - np.median(res)))
    sigma = max(sigma, 1e-3 * (np.ptp(patch) + 1e-12))
    plane_in = coef[0] + coef[1] * xx[inside] + coef[2] * yy[inside]
    r = plane_in - patch[inside]
    score = float(np.percentile(r, 75) / sigma)
    area = float((r > 2 * sigma).mean() * 100)
    cls = next(c for upper, c in DEPTH_BANDS if score < upper)
    return {"depth_class": cls, "depth_score": round(max(score, 0.0), 2),
            "depression_area_pct": round(area, 1),
            "reliable": bool(ring.sum() >= 200 and (x2 - x1) * (y2 - y1) >= 400)}


def estimate(img_bgr: np.ndarray, boxes: Sequence[Sequence[float]], model_path: str) -> List[dict]:
    if not boxes:
        return []
    dm = depth_map(img_bgr, model_path)
    return [depth_indicator(dm, b, [o for o in boxes if o is not b]) for b in boxes]
