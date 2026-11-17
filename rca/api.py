"""
rca.api
=======
FastAPI backend. Run:  uvicorn rca.api:app --host 0.0.0.0 --port 8000

Human-in-the-loop: the detector proposes obstructions, the engineer accepts
or rejects each one in the dashboard, and capacity is recomputed only from
the accepted ones. A traffic volume is used only when the engineer enters it
or confirms the video-measured value.

State is kept on disk (DATA_DIR), so a restart does not lose jobs.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rca.hcm import capacity as capmod
from rca.hcm import indo_hcm as hcm
from rca.twin import ctm

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("RCA_MODELS_DIR", ROOT / "models"))
DATA = Path(os.environ.get("RCA_DATA_DIR", ROOT / "data"))
DATA.mkdir(parents=True, exist_ok=True)
MAX_IMAGE_MB = 15
MAX_VIDEO_MB = int(os.environ.get("RCA_MAX_VIDEO_MB", "200"))
MAX_VIDEO_S = float(os.environ.get("RCA_MAX_VIDEO_SECONDS", "900"))
ANNOTATE_VIDEO = os.environ.get("RCA_ANNOTATE_VIDEO", "1") == "1"
ID_RE = re.compile(r"^[0-9a-f]{32}$")
def _metrics():
    p = MODELS / "defects_metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


MODEL_METRICS = _metrics()

app = FastAPI(title="Indian Road Capacity Analyzer", version="2.0.0")
_lock = threading.Lock()
_video_slot = threading.Semaphore(1)       # one video at a time on a small server
_scanner = None
_counter = None


def scanner():
    global _scanner
    with _lock:
        if _scanner is None:
            from rca.vision.road_scan import RoadScanner
            _scanner = RoadScanner(str(MODELS / "defects.onnx"))
    return _scanner


def counter():
    global _counter
    with _lock:
        if _counter is None:
            from rca.traffic.counter import VehicleCounter
            _counter = VehicleCounter(str(MODELS / "vehicles.onnx"))
    return _counter


def _dir(kind: str, rid: str) -> Path:
    if not ID_RE.match(rid):
        raise HTTPException(400, "Bad id.")
    p = DATA / kind / rid
    if not p.exists():
        raise HTTPException(404, "Not found.")
    return p


def _save(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def _load(path: Path):
    return json.loads(path.read_text())


async def _read_upload(f: UploadFile, max_mb: int, dest: Path) -> int:
    size = 0
    with open(dest, "wb") as out:
        while chunk := await f.read(1 << 20):
            size += len(chunk)
            if size > max_mb * (1 << 20):
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File larger than {max_mb} MB.")
            out.write(chunk)
    return size


def _parse_json(s: Optional[str], default):
    if not s:
        return default
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON: {e}")


# ------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"ok": True, "models": {m: (MODELS / m).exists() for m in ("defects.onnx", "vehicles.onnx")}}


@app.get("/api/options")
def options():
    return {"road_types": capmod.road_type_options(), "factors": capmod.factor_options(),
            "model": MODEL_METRICS, "los_tables": hcm.LOS_TABLES}


class CapacityRequest(BaseModel):
    road: dict
    intervals_m: List[List[float]] = Field(default_factory=list)
    parking_detected: bool = False


@app.post("/api/capacity")
def capacity(req: CapacityRequest):
    try:
        inp = capmod.RoadInputs.from_dict(req.road)
        return capmod.evaluate(inp, [tuple(i[:2]) for i in req.intervals_m], req.parking_detected)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/scan")
async def scan(image: UploadFile = File(...), road: str = Form("{}"), trapezoid: str = Form("")):
    road_d = _parse_json(road, {})
    trap = _parse_json(trapezoid, None)
    try:
        inp = capmod.RoadInputs.from_dict(road_d)
    except ValueError as e:
        raise HTTPException(400, str(e))
    rid = uuid.uuid4().hex
    d = DATA / "scans" / rid
    d.mkdir(parents=True)
    src = d / "input"
    await _read_upload(image, MAX_IMAGE_MB, src)
    img = cv2.imdecode(np.fromfile(str(src), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Not a readable image (use JPG or PNG).")
    max_side = 1920
    if max(img.shape[:2]) > max_side:
        r = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
        if trap:
            trap = [[x * r, y * r] for x, y in trap]
    cv2.imwrite(str(d / "original.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    src.unlink(missing_ok=True)
    width = inp.width_m or hcm.ROAD_TYPES[inp.road_type]["paved_width_m"]
    res = scanner().scan(img, width, trap)
    cv2.imwrite(str(d / "annotated.jpg"), scanner().annotate(img, res, trap), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cap = capmod.evaluate(inp, res["intervals_m"], res["parking_detected"])
    record = {"id": rid, "road": road_d, "trapezoid": trap, "scan": res, "capacity": cap,
              "created": time.time()}
    _save(d / "record.json", record)
    return {**record, "annotated_url": f"/files/scans/{rid}/annotated.jpg",
            "original_url": f"/files/scans/{rid}/original.jpg"}


@app.post("/api/video")
async def video(bg: BackgroundTasks, video: UploadFile = File(...), line_a: str = Form(...),
                line_b: str = Form(""), trap_length_m: str = Form(""), road_type: str = Form("divided_4lane")):
    la = _parse_json(line_a, None)
    lb = _parse_json(line_b, None)
    if not la or len(la) != 2:
        raise HTTPException(400, "line_a must be [[x1,y1],[x2,y2]] in video pixels.")
    if road_type not in hcm.ROAD_TYPES:
        raise HTTPException(400, "Unknown road type.")
    trap = float(trap_length_m) if trap_length_m else None
    if lb and not trap:
        raise HTTPException(400, "Enter the distance between line A and line B (trap length, m).")
    jid = uuid.uuid4().hex
    d = DATA / "jobs" / jid
    d.mkdir(parents=True)
    ext = Path(video.filename or "v.mp4").suffix.lower()[:5] or ".mp4"
    await _read_upload(video, MAX_VIDEO_MB, d / f"input{ext}")
    _save(d / "job.json", {"id": jid, "status": "queued", "progress": 0.0})
    bg.add_task(_run_video, jid, str(d / f"input{ext}"), la, lb, trap, road_type)
    return {"job_id": jid}


def _run_video(jid, path, la, lb, trap, road_type):
    d = DATA / "jobs" / jid
    with _video_slot:
        state = {"id": jid, "status": "running", "progress": 0.0}
        _save(d / "job.json", state)
        last = [0.0]

        def prog(p):
            if p - last[0] >= 0.02:
                last[0] = p
                _save(d / "job.json", {**state, "progress": round(p, 3)})
        try:
            ann = str(d / "annotated.mp4") if ANNOTATE_VIDEO else None
            res = counter().run(path, la, lb, trap, road_type, max_seconds=MAX_VIDEO_S,
                                annotate_path=ann, progress=prog)
            if ann and Path(ann).exists():
                res["annotated_url"] = f"/files/jobs/{jid}/annotated.mp4"
            _save(d / "job.json", {"id": jid, "status": "done", "progress": 1.0, "result": res})
        except Exception as e:  # noqa: BLE001 - report any failure to the user
            _save(d / "job.json", {"id": jid, "status": "error", "error": str(e)})
        finally:
            Path(path).unlink(missing_ok=True)


@app.get("/api/jobs/{jid}")
def job(jid: str):
    return _load(_dir("jobs", jid) / "job.json")


@app.post("/api/video/frame")
async def first_frame(video: UploadFile = File(...)):
    """First frame of a video, for drawing the count lines."""
    tmp = DATA / f"frame_{uuid.uuid4().hex}"
    await _read_upload(video, MAX_VIDEO_MB, tmp)
    try:
        cap = cv2.VideoCapture(str(tmp))
        ok, frame = cap.read()
        info = {"fps": cap.get(cv2.CAP_PROP_FPS), "frames": cap.get(cv2.CAP_PROP_FRAME_COUNT),
                "width": frame.shape[1] if ok else None, "height": frame.shape[0] if ok else None}
        cap.release()
    finally:
        tmp.unlink(missing_ok=True)
    if not ok:
        raise HTTPException(400, "Could not read the video.")
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    import base64
    return {**info, "image": "data:image/jpeg;base64," + base64.b64encode(buf).decode()}


class TwinRequest(BaseModel):
    capacity_clear_pcu_h: float = Field(gt=0)
    capacity_defect_pcu_h: float = Field(gt=0)
    vf_kmh: float = Field(gt=5, le=120)
    demand_pcu_h: float = Field(ge=0)
    length_m: float = Field(1000, ge=100, le=5000)
    obstruction_at: float = Field(0.6, ge=0.05, le=0.95)
    obstruction_length_m: float = Field(30, ge=5, le=500)
    duration_s: float = Field(1800, ge=300, le=7200)
    demand_profile: str = "constant"


@app.post("/api/twin")
def twin(req: TwinRequest):
    return ctm.compare(req.capacity_clear_pcu_h, req.capacity_defect_pcu_h, req.vf_kmh, req.demand_pcu_h,
                       length_m=req.length_m, obstruction_at=req.obstruction_at,
                       obstruction_length_m=req.obstruction_length_m, duration_s=req.duration_s,
                       demand_profile=req.demand_profile)


class ReportRequest(BaseModel):
    scan_id: Optional[str] = None
    road: dict
    accepted: Optional[List[int]] = None        # indices of scan objects the engineer accepted
    job_id: Optional[str] = None
    twin: Optional[TwinRequest] = None


@app.post("/api/report")
def report(req: ReportRequest):
    from rca import report as rep
    scan, img = None, None
    intervals, parking = [], False
    if req.scan_id:
        d = _dir("scans", req.scan_id)
        rec = _load(d / "record.json")
        scan = rec["scan"]
        objs = scan["objects"]
        keep = set(range(len(objs))) if req.accepted is None else set(req.accepted)
        scan = {**scan, "objects": [o for i, o in enumerate(objs) if i in keep]}
        intervals = [(o["left_m"], o["right_m"]) for o in scan["objects"]
                     if o["acts_through"] and o["acts_through"].startswith("F_W")]
        parking = any(o["acts_through"] and o["acts_through"].startswith("F_OP") for o in scan["objects"])
        from rca.vision.road_scan import RoadScanner
        orig = cv2.imread(str(d / "original.jpg"))
        img = str(d / "report_annotated.jpg")
        cv2.imwrite(img, RoadScanner.annotate(orig, scan, rec.get("trapezoid")), [cv2.IMWRITE_JPEG_QUALITY, 88])
    try:
        cap = capmod.evaluate(capmod.RoadInputs.from_dict(req.road), intervals, parking)
    except ValueError as e:
        raise HTTPException(400, str(e))
    traffic = None
    if req.job_id:
        j = _load(_dir("jobs", req.job_id) / "job.json")
        traffic = j.get("result")
    tw = twin(req.twin) if req.twin else None
    rid = uuid.uuid4().hex
    out = DATA / "reports"
    out.mkdir(exist_ok=True)
    path = out / f"{rid}.pdf"
    rep.build(str(path), cap, scan, img, traffic, tw, road_name=req.road.get("road_name", ""), metrics=MODEL_METRICS)
    return FileResponse(path, media_type="application/pdf", filename="road_capacity_report.pdf")


@app.get("/files/{kind}/{rid}/{name}")
def files(kind: str, rid: str, name: str):
    if kind not in ("scans", "jobs") or name not in ("annotated.jpg", "original.jpg", "annotated.mp4"):
        raise HTTPException(404)
    p = _dir(kind, rid) / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "web" / "static"), html=True), name="static")
