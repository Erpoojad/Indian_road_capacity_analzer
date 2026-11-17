"""
rca.traffic.counter
===================
Video -> classified vehicle counts -> flow (veh/h, PCU/h), speeds, density.

Survey layout (Indo-HCM Annex 5C): a count line A and, for speeds, a second
line B a known distance downstream (trap length, 50-75 m recommended). Each
tracked vehicle is counted ONCE per line, when the bottom-centre of its box
(the point touching the road) crosses the line. Crossing times are
interpolated between processed frames.

Classes: the vehicle model is COCO-pretrained YOLOv8n. Mapping to Indo-HCM
vehicle types:
    car -> car, motorcycle -> two_wheeler, bus -> bus, bicycle -> bicycle,
    truck -> lcv or tat by size (CHOICE: median box area >= 2.5 x the median
             car box area -> tat, else lcv; tat if no cars were seen)
LIMITATION: COCO has no auto-rickshaw class. Autos are usually detected as
car or motorcycle. Retrain with an Indian dataset (e.g. IDD) to separate them;
until then the auto share must be corrected manually in the dashboard.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from rca.hcm import indo_hcm as hcm
from rca.traffic.tracker import ByteTrackLite, Track
from rca.vision.onnx_yolo import OnnxYolo

COCO_TO_HCM = {"car": "car", "motorcycle": "two_wheeler", "bus": "bus", "bicycle": "bicycle", "truck": "truck"}
TRUCK_TAT_AREA_RATIO = 2.5
TARGET_FPS = 10.0
MIN_HITS = 3
Line = Sequence[Sequence[float]]


def _side(p, a, b) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossing(p, q, a, b) -> Optional[float]:
    """Fraction along p->q where it crosses segment a-b, or None."""
    s1, s2 = _side(p, a, b), _side(q, a, b)
    if s1 == 0 and s2 == 0 or (s1 > 0) == (s2 > 0) and s1 != 0 and s2 != 0:
        return None
    t = s1 / (s1 - s2) if s1 != s2 else 0.0
    x = (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
    ab = (b[0] - a[0], b[1] - a[1])
    u = ((x[0] - a[0]) * ab[0] + (x[1] - a[1]) * ab[1]) / (ab[0] ** 2 + ab[1] ** 2 + 1e-9)
    return t if -0.02 <= u <= 1.02 else None


def first_crossing(track: Track, line: Line, fps: float) -> Optional[Tuple[float, str]]:
    a, b = line
    h = track.history
    for (f0, p), (f1, q) in zip(h, h[1:]):
        t = crossing(p, q, a, b)
        if t is not None:
            when = (f0 + t * (f1 - f0)) / fps
            return when, ("down" if q[1] >= p[1] else "up")
    return None


class VehicleCounter:
    def __init__(self, model_path: str):
        self.model = OnnxYolo(model_path)
        wanted = set(COCO_TO_HCM)
        self.class_ids = [i for i, n in self.model.names.items() if n in wanted]

    def run(self, video_path: str, line_a: Line, line_b: Optional[Line] = None,
            trap_length_m: Optional[float] = None, road_type: str = "divided_4lane",
            max_seconds: Optional[float] = None, annotate_path: Optional[str] = None,
            progress: Optional[Callable[[float], None]] = None) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open the video.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stride = max(1, int(round(fps / TARGET_FPS)))
        limit = int(max_seconds * fps) if max_seconds else None
        tracker = ByteTrackLite(max_lost=int(round(1.2 * fps / stride)))
        writer = _Writer(annotate_path, w, h, fps / stride) if annotate_path else None
        frame_idx, last = 0, -1
        counted_a: Dict[int, str] = {}
        while True:
            ok = cap.grab()
            if not ok or (limit and frame_idx >= limit):
                break
            if frame_idx % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                dets = self.model(frame, conf=0.15, iou=0.6, classes=self.class_ids)
                active = tracker.step(dets, frame_idx)
                last = frame_idx
                if writer:
                    for t in active:
                        if t.id not in counted_a and t.hits >= MIN_HITS and first_crossing(t, line_a, fps):
                            counted_a[t.id] = t.cls
                    writer.write(_draw(frame, active, line_a, line_b, counted_a))
                if progress and n_total:
                    progress(min(frame_idx / (limit or n_total), 1.0))
            frame_idx += 1
        cap.release()
        if writer:
            writer.close()
        tracks = [t for t in tracker.finish() if t.hits >= MIN_HITS]
        duration_s = (last + 1) / fps if last >= 0 else 0.0
        return summarise(tracks, fps, duration_s, line_a, line_b, trap_length_m, road_type,
                         meta={"fps": fps, "stride": stride, "processed_fps": fps / stride,
                               "frames": frame_idx, "size": [w, h]})


def classify(tracks: List[Track]) -> Dict[int, str]:
    area = lambda t: float(np.median([(t.box[2] - t.box[0]) * (t.box[3] - t.box[1])]))
    cars = [area(t) for t in tracks if t.cls == "car"]
    car_area = median(cars) if len(cars) >= 3 else None
    out = {}
    for t in tracks:
        c = COCO_TO_HCM.get(t.cls, t.cls)
        if c == "truck":
            c = "tat" if car_area is None or area(t) >= TRUCK_TAT_AREA_RATIO * car_area else "lcv"
        out[t.id] = c
    return out


def summarise(tracks: List[Track], fps: float, duration_s: float, line_a: Line, line_b: Optional[Line],
              trap_m: Optional[float], road_type: str, meta: dict) -> dict:
    cls_of = classify(tracks)
    events = []
    for t in tracks:
        ca = first_crossing(t, line_a, fps)
        if not ca:
            continue
        ev = {"id": t.id, "class": cls_of[t.id], "t_a": round(ca[0], 3), "direction": ca[1]}
        if line_b is not None:
            cb = first_crossing(t, line_b, fps)
            if cb:
                ev["t_b"] = round(cb[0], 3)
                if trap_m:
                    dt = abs(cb[0] - ca[0])
                    v = trap_m / dt * 3.6 if dt > 0 else None
                    if v and 3 <= v <= 150:
                        ev["speed_kmh"] = round(v, 1)
                        ev["travel_time_s"] = round(dt, 3)
        events.append(ev)
    events.sort(key=lambda e: e["t_a"])
    notes = []
    if duration_s < 300:
        notes.append(f"Video covers {duration_s:.0f} s. Indo-HCM counts in 5-minute periods; the hourly "
                     f"flow is extrapolated from a shorter period and is less reliable.")
    notes.append("COCO model has no auto-rickshaw class; autos appear as car or two-wheeler. "
                 "Correct the auto count before using the PCU flow.")
    per_dir = {}
    for d in sorted({e["direction"] for e in events}):
        evs = [e for e in events if e["direction"] == d]
        per_dir[d] = direction_stats(evs, duration_s, road_type)
    return {"duration_s": round(duration_s, 2), "total_vehicles": len(events), "events": events,
            "directions": per_dir, "road_type": road_type, "meta": meta, "notes": notes,
            "speed_available": bool(line_b is not None and trap_m)}


def direction_stats(evs: List[dict], duration_s: float, road_type: str) -> dict:
    counts: Dict[str, int] = {}
    for e in evs:
        counts[e["class"]] = counts.get(e["class"], 0) + 1
    scale = 3600.0 / duration_s if duration_s > 0 else 0.0
    veh_h = {k: v * scale for k, v in counts.items()}
    pcu = hcm.pcu_values(road_type)
    known = {k: v for k, v in veh_h.items() if k in pcu}
    flow = hcm.flow_pcu(known, road_type) if known else {"flow_pcu_h": 0.0, "pcu_used": {}}
    out = {"counts": counts, "flow_veh_h": round(sum(veh_h.values()), 1),
           "flow_pcu_h": round(flow["flow_pcu_h"], 1), "pcu_used": flow["pcu_used"],
           "pcu_table": "Indo-HCM Table 5.3" if road_type not in hcm.PCU_TABLE_2_4 else "Indo-HCM Table 2.4"}
    if duration_s >= 600:
        bins: Dict[int, float] = {}
        for e in evs:
            b = int(e["t_a"] // 300)
            bins[b] = bins.get(b, 0.0) + pcu.get(e["class"], 1.0)
        full = [v * 12 for k, v in bins.items() if (k + 1) * 300 <= duration_s]
        if full:
            out["peak_5min_flow_pcu_h"] = round(max(full), 1)
    speeds = [e["speed_kmh"] for e in evs if "speed_kmh" in e]
    if speeds:
        tts = [e["travel_time_s"] for e in evs if "travel_time_s" in e]
        trap = speeds[0] * tts[0] / 3.6
        sms = hcm.space_mean_speed(trap, tts)
        out.update(n_speeds=len(speeds), space_mean_speed_kmh=round(sms, 1),
                   time_mean_speed_kmh=round(sum(speeds) / len(speeds), 1),
                   density_pcu_km=round(out["flow_pcu_h"] / sms, 1) if sms > 0 else None)
        cars = [e for e in evs if e["class"] == "car" and "speed_kmh" in e]
        car_times = [e["t_a"] for e in evs]  # headway to the vehicle ahead, any class
        heads = []
        for e in cars:
            i = car_times.index(e["t_a"])
            heads.append(e["t_a"] - car_times[i - 1] if i > 0 else None)
        if cars:
            op = hcm.operating_speed([e["speed_kmh"] for e in cars], heads)
            out["operating_speed"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in op.items()}
    return out


class _Writer:
    """H.264 MP4 through ffmpeg (browser-playable); silently disabled without ffmpeg."""

    def __init__(self, path, w, h, fps):
        self.p = None
        if shutil.which("ffmpeg"):
            self.p = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
                stdin=subprocess.PIPE)

    def write(self, frame):
        if self.p:
            self.p.stdin.write(frame.tobytes())

    def close(self):
        if self.p:
            self.p.stdin.close()
            self.p.wait()


def _draw(frame, tracks, line_a, line_b, counted):
    out = frame.copy()
    for ln, col, name in ((line_a, (0, 255, 0), "A"), (line_b, (0, 200, 255), "B")):
        if ln is not None:
            a, b = (tuple(int(v) for v in p) for p in ln)
            cv2.line(out, a, b, col, 2, cv2.LINE_AA)
            cv2.putText(out, name, (a[0] + 4, a[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    for t in tracks:
        x0, y0, x1, y1 = (int(v) for v in t.box)
        col = (0, 220, 0) if t.id in counted else (255, 160, 0)
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
        cv2.putText(out, f"#{t.id} {COCO_TO_HCM.get(t.cls, t.cls)}", (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (190, 28), (0, 0, 0), -1)
    cv2.putText(out, f"Counted at A: {len(counted)}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out
