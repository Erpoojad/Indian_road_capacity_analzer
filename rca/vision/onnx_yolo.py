"""
rca.vision.onnx_yolo
====================
YOLOv8 inference with ONNX Runtime (no PyTorch: the server fits in Render's
512 MB free plan). Pre/post-processing follows Ultralytics: square letterbox
with grey 114 padding, RGB/255, class-aware NMS, boxes mapped back to the
original image.

Export after retraining:
    yolo export model=best.pt format=onnx imgsz=640 opset=12 simplify=True
"""
from __future__ import annotations

import ast
import os
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np


def letterbox(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ov = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ov <= iou]
    return keep


class OnnxYolo:
    def __init__(self, path: str, threads: Optional[int] = None):
        import onnxruntime as ort
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or max(1, min(4, os.cpu_count() or 1))
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_cpu_mem_arena = False      # lower peak RAM (Render free plan: 512 MB)
        so.enable_mem_pattern = False
        self.session = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.imgsz = int(shape[2]) if isinstance(shape[2], int) else 640
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names: Dict[int, str] = ast.literal_eval(meta["names"]) if "names" in meta else {}

    def __call__(self, img_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.7,
                 classes: Optional[Sequence[int]] = None, max_det: int = 300) -> List[dict]:
        h0, w0 = img_bgr.shape[:2]
        lb, r, (px, py) = letterbox(img_bgr, self.imgsz)
        x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        pred = self.session.run(None, {self.input_name: np.ascontiguousarray(x)})[0][0]  # (4+nc, N)
        pred = pred.T
        cls_scores = pred[:, 4:]
        cls_id = cls_scores.argmax(1)
        score = cls_scores[np.arange(len(cls_id)), cls_id]
        m = score >= conf
        if classes is not None:
            m &= np.isin(cls_id, list(classes))
        pred, cls_id, score = pred[m], cls_id[m], score[m]
        if not len(pred):
            return []
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        keep = nms(boxes + (cls_id[:, None] * 7680.0), score, iou)[:max_det]
        boxes, cls_id, score = boxes[keep], cls_id[keep], score[keep]
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - px) / r).clip(0, w0)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - py) / r).clip(0, h0)
        return [{"cls": int(c), "name": self.names.get(int(c), str(int(c))), "conf": float(s),
                 "box": [float(v) for v in b]} for b, c, s in zip(boxes, cls_id, score)]
