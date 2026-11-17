"""
rca.vision.lanes
================
Lane-marking detection and condition assessment by image processing.
(No lane-marking dataset is available to the project yet, so no trained
model is used; the method is deterministic and explainable.)

Pipeline
1. Road region: the carriageway trapezoid marked by the engineer, or an
   automatic region: below the horizon, above the car bonnet/dashboard
   (found as the dark band at the bottom of a dash-cam photo).
2. Paint candidates: markings are narrow strips brighter than the asphalt on
   BOTH sides. A symmetric local filter (Nieto et al. 2010 style) compares each
   pixel with the road a stripe-width to its left and right; the width follows
   perspective. Shadow edges (bright on one side only) cancel out. Then only
   blobs shaped like strips (long, thin) are kept: gravel and sun patches go.
   Colour: white = low saturation; yellow = hue 28-68 deg with saturation.
3. Segments: probabilistic Hough transform; segments flatter than 20 deg
   are not longitudinal markings (in a forward view) and are discarded.
4. Lines: segments are grouped into marking lines by where they cut two
   reference rows (perspective lines converge, so two cuts identify a line).
5. Per line: colour (white/yellow), type (solid / broken from the share of
   its length that is painted), contrast (paint lightness / adjacent asphalt),
   and a condition score 0-100.

Condition score (CHOICE, uncalibrated): 50 % contrast + 50 % continuity.
  contrast term  = clip((ratio - 1.05) / 0.4, 0, 1)   (1.05 = invisible, >= 1.45 = clear)
  continuity term= painted share / expected share (solid 1.0, broken 0.4), max 1
Road score: mean of detected lines; no line -> 0 ("absent / worn out").
IRC:35 grades markings by retro-reflectivity, which a photo cannot measure;
calibrate this score against such readings before quoting it as a rating.
Indo-HCM 2017 has no lane-marking factor, so the score never changes capacity.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


def bonnet_row(img_bgr: np.ndarray) -> int:
    """Top of the car bonnet / dashboard in a dash-cam photo: the topmost strong
    brightness step across the lower image (row average over the central 80 %).
    Returns the image height when there is no such step (not a dash-cam photo)."""
    h, w = img_bgr.shape[:2]
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)[:, int(0.1 * w):int(0.9 * w)]
    m = cv2.GaussianBlur(g, (0, 0), 2).mean(axis=1)
    k = max(3, int(0.012 * h))
    d = np.zeros(h)
    d[k:h - k] = m[2 * k:] - m[:h - 2 * k]
    lo, hi = int(0.55 * h), int(0.97 * h)
    a = np.abs(d[lo:hi])
    if a.size == 0 or a.max() < 40:
        return h
    return lo + int(np.where(a >= max(40.0, 0.5 * a.max()))[0][0])


def road_mask(img_bgr: np.ndarray, trapezoid: Optional[Sequence[Point]]) -> Tuple[np.ndarray, bool]:
    h, w = img_bgr.shape[:2]
    m = np.zeros((h, w), np.uint8)
    if trapezoid and len(trapezoid) == 4:
        cv2.fillPoly(m, [np.array(trapezoid, np.int32)], 255)
        return m, True
    bottom = bonnet_row(img_bgr) - int(0.01 * h)       # keep clear of wiper / bonnet edge
    top = int(h * 0.45)
    if bottom - top < h * 0.08:
        bottom = h - 1
    poly = [(0, bottom), (w - 1, bottom), (w * 0.62, top), (w * 0.38, top)]
    cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
    return m, False


def analyse(img_bgr: np.ndarray, trapezoid: Optional[Sequence[Point]] = None,
            exclude_boxes: Sequence[Sequence[float]] = ()) -> dict:
    h, w = img_bgr.shape[:2]
    mask, marked = road_mask(img_bgr, trapezoid)
    for b in exclude_boxes:                         # vehicles, vendors... are not paint
        x0, y0, x1, y1 = (int(v) for v in b)
        mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0
    roi = mask > 0
    if roi.sum() < 2000:
        return {"available": False, "reason": "Road region too small to assess markings."}

    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    ys = np.where(roi.any(axis=1))[0]
    y_top, y_bot = int(ys.min()), int(ys.max())
    span = max(y_bot - y_top, 1)
    R = ridge_response(L, y_top, y_bot, w)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S = hsv[:, :, 0].astype(np.int32), hsv[:, :, 1].astype(np.int32)
    white = S < 60
    yellow = (H >= 14) & (H <= 34) & (S >= 25)         # OpenCV hue = degrees / 2 (yellow 28-68 deg)
    paint = (R >= np.maximum(24.0, 0.2 * L)) & (white | yellow) & roi
    paint_u8 = strip_filter(paint, max(10, span // 14), w)
    min_len = max(12, span // 12)
    raw = cv2.HoughLinesP(paint_u8, 1, np.pi / 180, threshold=15, minLineLength=min_len,
                          maxLineGap=max(4, span // 25))
    segs = []
    if raw is not None:
        for x0, y0, x1, y1 in np.asarray(raw).reshape(-1, 4):
            ang = abs(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
            if min(ang, 180 - ang) >= 20:
                if y0 > y1:
                    x0, y0, x1, y1 = x1, y1, x0, y0
                segs.append((float(x0), float(y0), float(x1), float(y1)))

    out_lines = []
    for g in _group(segs, y_top, y_bot, w):
        info = _describe(g, paint_u8, L, yellow, roi, y_top, y_bot)
        if info:
            info["id"] = len(out_lines) + 1
            out_lines.append(info)
    idx = float(np.mean([ln["score"] for ln in out_lines])) if out_lines else 0.0
    return {"available": True, "visibility_index": round(idx, 1), "condition": grade(idx),
            "n_lines": len(out_lines), "lines": out_lines,
            "region": "marked carriageway" if marked else "automatic (below horizon, above bonnet)",
            "paint_pixels_pct": round(100 * float((paint_u8 > 0).sum()) / float(roi.sum()), 2),
            "status": ("Image-processing indicator, uncalibrated. Does not change capacity "
                       "(Indo-HCM 2017 has no lane-marking factor)."
                       + ("" if marked else " Road edges not marked: kerbs at the road edge may be "
                          "counted as lines; mark the carriageway for an exact assessment."))}


def ridge_response(L: np.ndarray, y_top: int, y_bot: int, w: int) -> np.ndarray:
    """Symmetric local filter for bright strips (Nieto et al. 2010 style):
        r(x) = 2 L(x) - L(x - t) - L(x + t) - |L(x - t) - L(x + t)|
    t follows perspective: ~2 px at the far end, ~1.8 % of the width near the
    camera. A shadow edge is bright on one side only, so the |.| term cancels it."""
    Lb = cv2.GaussianBlur(L, (3, 3), 0)
    R = np.zeros_like(L)
    for y in range(y_top, y_bot + 1):
        t = int(round(2 + (y - y_top) / max(y_bot - y_top, 1) * 0.018 * w))
        row = Lb[y]
        lft = np.empty_like(row); rgt = np.empty_like(row)
        lft[t:] = row[:-t]; lft[:t] = row[0]
        rgt[:-t] = row[t:]; rgt[-t:] = row[-1]
        R[y] = 2 * row - lft - rgt - np.abs(lft - rgt)
    return R


def strip_filter(paint: np.ndarray, min_len: int, w: int) -> np.ndarray:
    """Keep only connected blobs shaped like painted strips: long (>= min_len px),
    thin (length / width >= 4) and narrower than 3 % of the photo width.
    Gravel speckle and sunlit patches between shadows are removed."""
    m = paint.astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 12:
            continue
        pts = np.column_stack(np.where(lab == i))[:, ::-1].astype(np.float32)
        _, (a, b), _ = cv2.minAreaRect(pts)
        long_, short = max(a, b), max(min(a, b), 1.0)
        keep[i] = long_ >= min_len and long_ / short >= 4.0 and short <= 0.03 * w
    return (keep[lab] * 255).astype(np.uint8)


def _x_at(seg, y):
    x0, y0, x1, y1 = seg
    return x0 + (x1 - x0) * (y - y0) / (y1 - y0)


def _group(segs, y_top, y_bot, w):
    """Group segments whose extensions cut two reference rows at nearly the same x."""
    ya, yb = y_top + 0.35 * (y_bot - y_top), y_bot
    tol = max(10.0, 0.035 * w)
    groups: List[dict] = []
    for s in sorted(segs, key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1])):
        xa, xb = _x_at(s, ya), _x_at(s, yb)
        for g in groups:
            if abs(g["xa"] - xa) < tol and abs(g["xb"] - xb) < tol * 1.6:
                g["segs"].append(s)
                break
        else:
            groups.append({"xa": xa, "xb": xb, "segs": [s]})
    return groups


def _describe(g, paint_u8, L, yellow, roi, y_top, y_bot):
    segs = g["segs"]
    ext = max(s[3] for s in segs) - min(s[1] for s in segs)
    if ext < 0.2 * (y_bot - y_top):
        return None                                 # too short along the road: wiper, debris, arrow
    pts_y = np.array([v for s in segs for v in (s[1], s[3])])
    pts_x = np.array([v for s in segs for v in (s[0], s[2])])
    a, b = np.polyfit(pts_y, pts_x, 1)              # x = a*y + b
    y0 = float(pts_y.min())
    h, w = L.shape
    ys = np.arange(int(max(y_top, min(y0, y_bot - 1))), int(y_bot))
    xs = np.round(a * ys + b).astype(int)
    ok = (xs >= 2) & (xs < w - 2)
    ys, xs = ys[ok], xs[ok]
    if len(ys) < 10:
        return None
    valid = roi[ys, xs]
    if valid.sum() < 10:
        return None
    ys, xs = ys[valid], xs[valid]
    half = np.maximum(2, ((ys - y_top) / max(y_bot - y_top, 1) * 6).astype(int) + 2)
    hit = np.array([paint_u8[y, max(0, x - r):x + r + 1].any() for y, x, r in zip(ys, xs, half)])
    painted = float(hit.mean())
    runs = int(np.sum(np.diff(hit.astype(int)) == 1) + (1 if hit[0] else 0))
    kind = "solid" if painted >= 0.7 or runs <= 1 else "broken"
    expected = 1.0 if kind == "solid" else 0.4
    pm = np.zeros_like(paint_u8)
    for s in segs:
        cv2.line(pm, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), 255, 3)
    on = (pm > 0) & (paint_u8 > 0)
    ring = ((cv2.dilate(pm, np.ones((15, 15), np.uint8)) > 0)
            & ~(cv2.dilate(pm, np.ones((5, 5), np.uint8)) > 0) & roi)
    if on.sum() < 15 or ring.sum() < 15:
        return None
    contrast = float(L[on].mean() / max(np.median(L[ring]), 1.0))
    if contrast < 1.05:
        return None                                 # not brighter than the asphalt: no paint
    colour = "yellow" if float(yellow[on].mean()) > 0.5 else "white"
    c_term = float(np.clip((contrast - 1.05) / 0.4, 0, 1))
    k_term = float(np.clip(painted / expected, 0, 1))
    score = 100 * (0.5 * c_term + 0.5 * k_term)
    yt = float(max(y0, y_top))
    return {"type": kind, "colour": colour, "contrast_ratio": round(contrast, 2),
            "painted_share": round(painted, 2), "score": round(score, 1), "condition": grade(score),
            "p_top": [round(float(a * yt + b), 1), round(yt, 1)],
            "p_bottom": [round(float(a * y_bot + b), 1), float(y_bot)],
            "segments": [[int(v) for v in s] for s in segs]}


def grade(idx: float) -> str:
    return "good" if idx >= 65 else "fair" if idx >= 40 else "poor" if idx >= 15 else "absent / worn out"


def draw(img_bgr: np.ndarray, res: dict) -> np.ndarray:
    out = img_bgr.copy()
    lw = max(2, img_bgr.shape[1] // 350)
    for ln in res.get("lines", []):
        col = (0, 215, 255) if ln["colour"] == "yellow" else (255, 255, 255)
        (x0, y0), (x1, y1) = ln["p_top"], ln["p_bottom"]
        cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 0), lw + 4, cv2.LINE_AA)
        cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)), col, lw, cv2.LINE_AA)
        for s in ln["segments"]:
            cv2.line(out, (s[0], s[1]), (s[2], s[3]), (255, 140, 0), lw + 1, cv2.LINE_AA)
        label = f"L{ln['id']} {ln['type']} {ln['colour']} {ln['score']:.0f}"
        fs = max(0.45, img_bgr.shape[1] / 1800)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        tx = int(np.clip(x1 - tw // 2, 2, out.shape[1] - tw - 4))
        ty = int(y1) - 10 - (ln["id"] - 1) % 3 * (th + 10)
        cv2.rectangle(out, (tx - 3, ty - th - 4), (tx + tw + 3, ty + 4), (40, 40, 40), -1)
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
    return out
