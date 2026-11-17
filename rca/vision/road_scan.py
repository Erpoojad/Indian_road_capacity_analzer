"""
rca.vision.road_scan
====================
Photo -> detections -> metres across the road -> capacity inputs.

Perspective: the user may mark the carriageway as a trapezoid (near-left,
near-right, far-right, far-left). The left and right road edges are then
interpolated row by row, so an object's lateral position and width are
measured at its own distance from the camera (its bottom edge, where it
touches the road). Without a trapezoid the full photo width is taken as the
carriageway width (only correct for a photo taken square to the road).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from rca.hcm import extensions as ext
from rca.vision import lanes
from rca.vision.onnx_yolo import OnnxYolo

CLASSES = ["barricade", "pothole", "illegal_parking", "street_vendor", "cart", "garbage", "vehicle"]
CONF = {"pothole": 0.25, "barricade": 0.35, "illegal_parking": 0.35, "street_vendor": 0.35,
        "cart": 0.35, "garbage": 0.30, "vehicle": 0.35}
CROSS_CLASS_IOU = 0.6
COLORS = {"pothole": (0, 0, 255), "street_vendor": (255, 0, 200), "cart": (200, 0, 255),
          "garbage": (0, 160, 255), "barricade": (0, 255, 255), "illegal_parking": (255, 120, 0),
          "vehicle": (160, 160, 160)}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def clean(dets: List[dict]) -> List[dict]:
    """Per-class thresholds; one object never counted under two obstruction
    labels; vendor/cart boxes that are really moving vehicles are dropped."""
    dets = sorted((d for d in dets if d["conf"] >= CONF.get(d["name"], 0.35)), key=lambda d: -d["conf"])
    vehicles = [d for d in dets if d["name"] == "vehicle"]
    kept: List[dict] = []
    for d in dets:
        if d["name"] == "vehicle":
            continue
        if d["name"] in ("street_vendor", "cart") and any(
                iou(d["box"], v["box"]) > CROSS_CLASS_IOU and v["conf"] > d["conf"] for v in vehicles):
            continue
        if any(iou(d["box"], k["box"]) > CROSS_CLASS_IOU for k in kept):
            continue
        kept.append(d)
    return kept + vehicles


class Edges:
    """Left/right carriageway edge (px) at any image row."""

    def __init__(self, w: int, h: int, trapezoid: Optional[Sequence[Tuple[float, float]]]):
        if trapezoid and len(trapezoid) == 4:
            (nlx, nly), (nrx, nry), (frx, fry), (flx, fly) = trapezoid
            self.near_y, self.far_y = (nly + nry) / 2, (fly + fry) / 2
            self.near = (nlx, nrx)
            self.far = (flx, frx)
            self.marked = True
        else:
            self.near_y, self.far_y, self.near, self.far, self.marked = h, 0, (0, w), (0, w), False

    def at(self, y: float) -> Tuple[float, float]:
        if self.near_y == self.far_y:
            return self.near
        t = (y - self.near_y) / (self.far_y - self.near_y)
        t = float(np.clip(t, -0.2, 1.0))
        return (self.near[0] + t * (self.far[0] - self.near[0]),
                self.near[1] + t * (self.far[1] - self.near[1]))


def to_metres(box, edges: Edges, width_m: float) -> Tuple[float, float]:
    x0, _, x1, y1 = box
    l_px, r_px = edges.at(y1)
    span = max(r_px - l_px, 1.0)
    l = (x0 - l_px) / span * width_m
    r = (x1 - l_px) / span * width_m
    return max(0.0, min(width_m, l)), max(0.0, min(width_m, r))


class RoadScanner:
    def __init__(self, model_path: str):
        self.model = OnnxYolo(model_path)
        missing = set(CLASSES) - set(self.model.names.values())
        if missing:
            raise ValueError(f"Defect model lacks classes {sorted(missing)}; retrain with training/class_map.yaml")

    def scan(self, img_bgr: np.ndarray, width_m: float,
             trapezoid: Optional[Sequence[Tuple[float, float]]] = None) -> dict:
        h, w = img_bgr.shape[:2]
        dets = clean(self.model(img_bgr, conf=min(CONF.values())))
        edges = Edges(w, h, trapezoid)
        objects, intervals, parking = [], [], 0
        for d in dets:
            l, r = to_metres(d["box"], edges, width_m)
            on_road = r - l > 0.01
            o = {"class": d["name"], "conf": round(d["conf"], 3), "box": [round(v, 1) for v in d["box"]],
                 "left_m": round(l, 2), "right_m": round(r, 2), "width_m": round(r - l, 2),
                 "on_carriageway": on_road, "acts_through": None}
            if d["name"] in ext.WIDTH_CLASSES and on_road:
                o["acts_through"] = "F_W (effective width)"
                intervals.append((l, r))
            elif d["name"] in ext.PARKING_CLASSES and on_road:
                o["acts_through"] = "F_OP (Indo-HCM Tables 5.9/5.10)"
                parking += 1
            elif d["name"] == "vehicle":
                o["acts_through"] = "moving traffic (ignored)"
            if d["name"] == "pothole":
                o["irc82_size"] = ext.pothole_size_class(r - l)
            objects.append(o)
        exclude = [d["box"] for d in dets if d["name"] in ("vehicle", "illegal_parking", "street_vendor", "cart")]
        lane = lanes.analyse(img_bgr, trapezoid, exclude)
        counts: Dict[str, int] = {}
        for o in objects:
            counts[o["class"]] = counts.get(o["class"], 0) + 1
        return {"objects": objects, "counts": counts, "intervals_m": intervals,
                "parking_detected": parking > 0, "lane_markings": lane,
                "perspective_corrected": edges.marked, "image_wh": [w, h],
                "notes": [] if edges.marked else
                ["Road edges not marked: the full photo width was taken as the carriageway width. "
                 "Mark the road edges for a perspective-corrected measurement."]}

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
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, 1)
            cv2.rectangle(out, (x0, max(0, y0 - th - 8)), (x0 + tw + 6, y0), c, -1)
            cv2.putText(out, label, (x0 + 3, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (0, 0, 0), 1, cv2.LINE_AA)
        return out
