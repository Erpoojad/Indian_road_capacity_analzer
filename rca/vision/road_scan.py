"""
rca.vision.road_scan
====================
Photo -> detections -> metres across the road -> capacity inputs.

Detection rules (same as the earlier version, core.py):
* Per-class confidence thresholds (CONF_THRESHOLDS), overridden by
  models/class_thresholds.json when present (per-class best-F1 thresholds
  written by training/evaluate.py).
* Zoomed second look for small potholes (RCA_TILED, default on): when the
  photo is at least 1280 px on its long side, the detector also runs on four
  overlapping 2x2 tiles. Only pothole boxes are taken from tiles; boxes cut by
  a tile edge are dropped; where a pothole is seen twice the more confident
  view is kept, and a box lying mostly inside another is not split off.
* One object never counted under two obstruction labels (IoU >= 0.60 between
  different classes: keep the more confident).
* Vehicle veto: vendor/cart boxes overlapping a 'vehicle' box (IoU >= 0.40)
  are dropped (auto-rickshaws are often misread as carts/vendors).
* Optional MiDaS depth indicator for potholes (RCA_ENABLE_DEPTH or per request).

Perspective: the engineer may mark the carriageway as a trapezoid (near-left,
near-right, far-right, far-left). Road edges are interpolated row by row and
each object is measured at its bottom edge. Without a trapezoid the photo
width is taken as the carriageway width (the earlier version's assumption).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from rca.hcm import extensions as ext
from rca.vision import lanes
from rca.vision.onnx_yolo import OnnxYolo

CLASSES = ["barricade", "pothole", "illegal_parking", "street_vendor", "cart", "garbage", "vehicle"]
OBSTRUCTION_CLASSES = ["barricade", "pothole", "illegal_parking", "street_vendor", "cart", "garbage"]
CONF_THRESHOLDS: Dict[str, float] = {"barricade": 0.45, "pothole": 0.45, "illegal_parking": 0.50,
                                     "street_vendor": 0.65, "cart": 0.65, "garbage": 0.45,
                                     "vehicle": 0.40}
DEFAULT_CONF = 0.45
THRESHOLDS_FILE = "class_thresholds.json"
CROSS_CLASS_IOU = 0.60
VETO_IOU = 0.40
TILE_CLASSES = ("pothole",)
TILE_GRID, TILE_OVERLAP, TILE_MIN_EDGE_PX, TILE_EDGE_PX = 2, 0.25, 640, 3
COLORS = {"pothole": (0, 0, 255), "street_vendor": (255, 0, 200), "cart": (200, 0, 255),
          "garbage": (0, 160, 255), "barricade": (0, 255, 255), "illegal_parking": (255, 120, 0),
          "vehicle": (160, 160, 160)}


def _env_on(key: str, default: str) -> bool:
    return os.environ.get(key, default).strip().lower() in ("1", "true", "yes", "on")


def load_class_thresholds(model_path: str) -> Dict[str, float]:
    out = dict(CONF_THRESHOLDS)
    f = Path(model_path).with_name(THRESHOLDS_FILE)
    if f.exists():
        try:
            out.update({k: float(v) for k, v in json.loads(f.read_text()).items() if 0 < float(v) < 1})
        except (ValueError, TypeError, AttributeError):
            pass
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def containment(small, big) -> float:
    iw = max(0.0, min(small[2], big[2]) - max(small[0], big[0]))
    ih = max(0.0, min(small[3], big[3]) - max(small[1], big[1]))
    a = (small[2] - small[0]) * (small[3] - small[1])
    return iw * ih / a if a > 0 else 0.0


def tile_windows(w: int, h: int, grid: int = TILE_GRID, overlap: float = TILE_OVERLAP):
    tw = int(round(w / (grid - (grid - 1) * overlap)))
    th = int(round(h / (grid - (grid - 1) * overlap)))
    xs = [min(int(round(i * tw * (1 - overlap))), w - tw) for i in range(grid)]
    ys = [min(int(round(j * th * (1 - overlap))), h - th) for j in range(grid)]
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def merge_tiled(full: List[dict], tiles, image_wh, classes=TILE_CLASSES, iou_threshold: float = 0.5):
    W, H = image_wh
    cand = []
    for (x0, y0, x1, y1), dets in tiles:
        for d in dets:
            if d["name"] not in classes:
                continue
            bx1, by1, bx2, by2 = d["box"]
            cut = ((bx1 <= TILE_EDGE_PX and x0 > 0) or (by1 <= TILE_EDGE_PX and y0 > 0) or
                   (bx2 >= (x1 - x0) - TILE_EDGE_PX and x1 < W) or (by2 >= (y1 - y0) - TILE_EDGE_PX and y1 < H))
            if not cut:
                cand.append({**d, "box": [bx1 + x0, by1 + y0, bx2 + x0, by2 + y0], "from_tile": True})
    pool = sorted([d for d in full if d["name"] in classes] + cand, key=lambda d: -d["conf"])
    kept = [d for d in full if d["name"] not in classes]
    chosen: List[dict] = []
    for d in pool:
        if any(k["name"] == d["name"] and (iou(k["box"], d["box"]) >= iou_threshold
               or containment(d["box"], k["box"]) > 0.7 or containment(k["box"], d["box"]) > 0.7)
               for k in chosen):
            continue
        chosen.append(d)
    return kept + chosen


def suppress_cross_class(dets: List[dict]) -> Tuple[List[dict], int]:
    cand = sorted((d for d in dets if d["name"] in OBSTRUCTION_CLASSES), key=lambda d: -d["conf"])
    kept, dropped = [], 0
    for d in cand:
        if any(k["name"] != d["name"] and iou(k["box"], d["box"]) >= CROSS_CLASS_IOU for k in kept):
            dropped += 1
            continue
        kept.append(d)
    return kept + [d for d in dets if d["name"] not in OBSTRUCTION_CLASSES], dropped


def vehicle_veto(dets: List[dict]) -> Tuple[List[dict], int]:
    vb = [d["box"] for d in dets if d["name"] == "vehicle"]
    kept, vetoed = [], 0
    for d in dets:
        if d["name"] in ("street_vendor", "cart") and vb and max(iou(d["box"], v) for v in vb) >= VETO_IOU:
            vetoed += 1
            continue
        kept.append(d)
    return kept, vetoed


class Edges:
    """Left/right carriageway edge (px) at any image row."""

    def __init__(self, w: int, h: int, trapezoid: Optional[Sequence[Tuple[float, float]]]):
        if trapezoid and len(trapezoid) == 4:
            (nlx, nly), (nrx, nry), (frx, fry), (flx, fly) = trapezoid
            self.near_y, self.far_y = (nly + nry) / 2, (fly + fry) / 2
            self.near, self.far, self.marked = (nlx, nrx), (flx, frx), True
        else:
            self.near_y, self.far_y, self.near, self.far, self.marked = h, 0, (0, w), (0, w), False

    def at(self, y: float) -> Tuple[float, float]:
        if self.near_y == self.far_y:
            return self.near
        t = float(np.clip((y - self.near_y) / (self.far_y - self.near_y), -0.2, 1.0))
        return (self.near[0] + t * (self.far[0] - self.near[0]),
                self.near[1] + t * (self.far[1] - self.near[1]))


def to_metres(box, edges: Edges, width_m: float) -> Tuple[float, float]:
    x0, _, x1, y1 = box
    l_px, r_px = edges.at(y1)
    span = max(r_px - l_px, 1.0)
    l, r = (x0 - l_px) / span * width_m, (x1 - l_px) / span * width_m
    return max(0.0, min(width_m, l)), max(0.0, min(width_m, r))


class RoadScanner:
    def __init__(self, model_path: str, depth_model: Optional[str] = None):
        self.model = OnnxYolo(model_path)
        missing = set(CLASSES) - set(self.model.names.values())
        if missing:
            raise ValueError(f"Defect model lacks classes {sorted(missing)}; retrain with training/class_map.yaml")
        self.thresholds = load_class_thresholds(model_path)
        self.tiled = _env_on("RCA_TILED", "1")
        self.depth_model = depth_model if depth_model and Path(depth_model).exists() else None
        self.depth_default = _env_on("RCA_ENABLE_DEPTH", "0")

    def detect(self, img: np.ndarray) -> Tuple[List[dict], dict]:
        floor = min(0.25, min(self.thresholds.values()))
        full = self.model(img, conf=floor)
        h, w = img.shape[:2]
        tiled = self.tiled and max(h, w) >= 2 * TILE_MIN_EDGE_PX
        if tiled:
            wins = tile_windows(w, h)
            full = merge_tiled(full, [(win, self.model(img[win[1]:win[3], win[0]:win[2]], conf=floor))
                                      for win in wins], (w, h))
        raw = [d for d in full if d["conf"] >= self.thresholds.get(d["name"], DEFAULT_CONF)]
        raw, merged = suppress_cross_class(raw)
        kept, vetoed = vehicle_veto(raw)
        return kept, {"tiled": tiled, "from_tiles": sum(1 for d in kept if d.get("from_tile")),
                      "cross_class_merged": merged, "vehicle_vetoed": vetoed,
                      "thresholds": self.thresholds}

    def scan(self, img_bgr: np.ndarray, width_m: float,
             trapezoid: Optional[Sequence[Tuple[float, float]]] = None,
             depth: Optional[bool] = None) -> dict:
        h, w = img_bgr.shape[:2]
        dets, info = self.detect(img_bgr)
        edges = Edges(w, h, trapezoid)
        objects, intervals, parking = [], [], 0
        for d in dets:
            l, r = (round(v, 2) for v in to_metres(d["box"], edges, width_m))   # one set of numbers everywhere
            on_road = r - l > 0.01
            o = {"class": d["name"], "conf": round(d["conf"], 3), "box": [round(v, 1) for v in d["box"]],
                 "left_m": round(l, 2), "right_m": round(r, 2), "width_m": round(r - l, 2),
                 "on_carriageway": on_road, "acts_through": None,
                 "found_by": "zoomed tile" if d.get("from_tile") else "full image"}
            if d["name"] in ext.WIDTH_CLASSES and on_road:
                o["acts_through"] = "F_W (effective width)"
                intervals.append((l, r))
            elif d["name"] in ext.PARKING_CLASSES and on_road:
                o["acts_through"] = "F_OP (Indo-HCM Tables 5.9/5.10)"
                parking += 1
            elif d["name"] == "vehicle":
                o["acts_through"] = "moving traffic (vendor/cart veto only)"
            if d["name"] == "pothole":
                o["irc82_size"] = ext.pothole_size_class(r - l)
            objects.append(o)
        notes = [] if edges.marked else [
            "Road edges not marked: the full photo width was taken as the carriageway width. "
            "Mark the road edges for a perspective-corrected measurement."]
        use_depth = self.depth_default if depth is None else depth
        p_idx = [i for i, o in enumerate(objects) if o["class"] == "pothole"]
        if use_depth and p_idx:
            if self.depth_model:
                from rca.vision import depth as dep
                try:
                    res = dep.estimate(img_bgr, [objects[i]["box"] for i in p_idx], self.depth_model)
                    for i, r in zip(p_idx, res):
                        objects[i]["depth_indicator"] = r
                    notes.append(dep.DEPTH_NOTE)
                except Exception as e:  # noqa: BLE001
                    notes.append(f"Depth indicator failed ({e}); width-only pothole class used.")
            else:
                notes.append("Depth indicator requested but models/midas_small.onnx is missing.")
        exclude = [d["box"] for d in dets if d["name"] in ("vehicle", "illegal_parking", "street_vendor", "cart")]
        lane = lanes.analyse(img_bgr, trapezoid, exclude)
        counts: Dict[str, int] = {}
        for o in objects:
            counts[o["class"]] = counts.get(o["class"], 0) + 1
        return {"objects": objects, "counts": counts, "intervals_m": intervals,
                "parking_detected": parking > 0, "lane_markings": lane, "detector": info,
                "perspective_corrected": edges.marked, "image_wh": [w, h], "notes": notes}

    @staticmethod
    def annotate(img_bgr: np.ndarray, scan: dict, trapezoid=None) -> np.ndarray:
        out = lanes.draw(img_bgr, scan.get("lane_markings", {}))
        if trapezoid and len(trapezoid) == 4:
            cv2.polylines(out, [np.array(trapezoid, np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
        s = max(0.5, min(out.shape[:2]) / 900)
        for o in scan["objects"]:
            x0, y0, x1, y1 = (int(v) for v in o["box"])
            c = COLORS.get(o["class"], (255, 255, 255))
            cv2.rectangle(out, (x0, y0), (x1, y1), c, max(2, int(3 * s)))
            label = f"{o['class']} {o['conf']:.2f}"
            if o["acts_through"] and o["class"] != "vehicle":
                label += f" | {o['width_m']:.2f} m"
            if o.get("found_by") == "zoomed tile":
                label += " (tile)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, 1)
            cv2.rectangle(out, (x0, max(0, y0 - th - 8)), (x0 + tw + 6, y0), c, -1)
            cv2.putText(out, label, (x0 + 3, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (0, 0, 0), 1, cv2.LINE_AA)
        return out
