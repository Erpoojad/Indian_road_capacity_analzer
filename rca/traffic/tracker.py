"""
rca.traffic.tracker
===================
Light ByteTrack-style multi-object tracker (no PyTorch, no extra packages
beyond SciPy).

* Constant-velocity prediction of each track's box.
* Two-stage association (ByteTrack): first high-confidence detections, then
  the remaining low-confidence ones, matched by IoU with the Hungarian
  method; low-confidence boxes never start a track (cuts false tracks).
* A track lost for more than `max_lost` processed frames is closed.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


class Track:
    _next = 1

    def __init__(self, box, cls: str, conf: float, frame: int):
        self.id = Track._next
        Track._next += 1
        self.box = np.asarray(box, float)
        self.vel = np.zeros(4)
        self.cls_votes: Dict[str, float] = {cls: conf}
        self.lost = 0
        self.hits = 1
        self.first_frame = self.last_frame = frame
        self.history: List[tuple] = [(frame, self.anchor())]

    def anchor(self):
        """Bottom-centre: the point where the vehicle touches the road."""
        return (float((self.box[0] + self.box[2]) / 2), float(self.box[3]))

    def predict(self):
        return self.box + self.vel

    def update(self, box, cls, conf, frame):
        box = np.asarray(box, float)
        gap = max(frame - self.last_frame, 1)
        self.vel = 0.6 * self.vel + 0.4 * (box - self.box) / gap
        self.box = box
        self.cls_votes[cls] = self.cls_votes.get(cls, 0.0) + conf
        self.lost = 0
        self.hits += 1
        self.last_frame = frame
        self.history.append((frame, self.anchor()))

    @property
    def cls(self) -> str:
        return max(self.cls_votes, key=self.cls_votes.get)


class ByteTrackLite:
    def __init__(self, high: float = 0.45, low: float = 0.15, match_iou: float = 0.25, max_lost: int = 12):
        self.high, self.low, self.match_iou, self.max_lost = high, low, match_iou, max_lost
        self.tracks: List[Track] = []
        self.closed: List[Track] = []

    def _match(self, tracks, dets, frame):
        if not tracks or not dets:
            return list(range(len(tracks))), list(range(len(dets)))
        pred = np.array([t.predict() for t in tracks])
        boxes = np.array([d["box"] for d in dets])
        m = iou_matrix(pred, boxes)
        r, c = linear_sum_assignment(-m)
        un_t, un_d = set(range(len(tracks))), set(range(len(dets)))
        for i, j in zip(r, c):
            if m[i, j] >= self.match_iou:
                tracks[i].update(dets[j]["box"], dets[j]["name"], dets[j]["conf"], frame)
                un_t.discard(i); un_d.discard(j)
        return sorted(un_t), sorted(un_d)

    def step(self, dets: List[dict], frame: int) -> List[Track]:
        high = [d for d in dets if d["conf"] >= self.high]
        low = [d for d in dets if self.low <= d["conf"] < self.high]
        un_t, un_d = self._match(self.tracks, high, frame)
        rest = [self.tracks[i] for i in un_t]
        un_t2, _ = self._match(rest, low, frame)
        for i in un_t2:
            t = rest[i]
            t.lost += 1
            t.box = t.predict()
        for j in un_d:
            d = high[j]
            self.tracks.append(Track(d["box"], d["name"], d["conf"], frame))
        alive = []
        for t in self.tracks:
            (alive if t.lost <= self.max_lost else self.closed).append(t)
        self.tracks = alive
        return [t for t in self.tracks if t.lost == 0]

    def finish(self) -> List[Track]:
        self.closed += self.tracks
        self.tracks = []
        return self.closed
