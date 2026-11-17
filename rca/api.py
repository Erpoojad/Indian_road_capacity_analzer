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
            _scanner = RoadScanner(str(MODELS / "defects.onnx"), str(MODELS / "midas_small.onnx"))
    return _scanner


def counter():
    """Loaded for one video job and released after it (memory on the free plan)."""
    from rca.traffic.counter import VehicleCounter
    return VehicleCounter(str(MODELS / "vehicles.onnx"))


def _trim_memory() -> None:
    """Give freed heap back to the OS after heavy work (glibc keeps it otherwise),
    so the free 512 MB plan is not held at its high-water mark."""
    import ctypes
    import gc
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


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
    return {"ok": True, "models": {m: (MODELS / m).exists()
                                   for m in ("defects.onnx", "vehicles.onnx", "midas_small.onnx")}}


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


def _scan_image(img: np.ndarray, road_d: dict, trap, depth: Optional[bool], image_name: str,
                chainage: Optional[float] = None) -> dict:
    try:
        inp = capmod.RoadInputs.from_dict(road_d)
    except ValueError as e:
        raise HTTPException(400, str(e))
    rid = uuid.uuid4().hex
    d = DATA / "scans" / rid
    d.mkdir(parents=True)
    max_side = 1600                      # earlier version's limit; tiles start at 1280 px
    if max(img.shape[:2]) > max_side:
        r = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
        if trap:
            trap = [[x * r, y * r] for x, y in trap]
    cv2.imwrite(str(d / "original.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    width = inp.width_m or hcm.ROAD_TYPES[inp.road_type]["paved_width_m"]
    res = scanner().scan(img, width, trap, depth=depth)
    cv2.imwrite(str(d / "annotated.jpg"), scanner().annotate(img, res, trap), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cap = capmod.evaluate(inp, res["intervals_m"], res["parking_detected"])
    record = {"id": rid, "road": road_d, "trapezoid": trap, "scan": res, "capacity": cap,
              "image_name": image_name, "chainage_m": chainage, "created": time.time()}
    _save(d / "record.json", record)
    leg = _legacy_from_record(record, None)
    _trim_memory()
    return {**record, "legacy": _legacy_summary(leg),
            "annotated_url": f"/files/scans/{rid}/annotated.jpg",
            "original_url": f"/files/scans/{rid}/original.jpg"}


def _legacy_summary(leg: dict) -> dict:
    return {k: leg.get(k) for k in ("per_defect", "overall_guidance", "capacity_loss_pct",
                                    "indo_hcm_loss_pct", "indicative_vehicles", "dsv_los_c_ideal_pcu_h",
                                    "dsv_los_c_reduced_pcu_h", "digital_twin")}


def _legacy_from_record(rec: dict, accepted: Optional[List[int]], road: Optional[dict] = None) -> dict:
    """Earlier version's result dict for a scan, using the engineer's accepted
    detections and the current road inputs."""
    from rca.legacy import result as lr
    road = road if road is not None else rec["road"]
    scan = rec["scan"]
    objs = scan["objects"]
    keep = set(range(len(objs))) if accepted is None else set(accepted)
    scan = {**scan, "objects": [o for i, o in enumerate(objs) if i in keep]}
    iv = [(o["left_m"], o["right_m"]) for o in scan["objects"] if (o["acts_through"] or "").startswith("F_W")]
    parking = any((o["acts_through"] or "").startswith("F_OP") for o in scan["objects"])
    try:
        cap = capmod.evaluate(capmod.RoadInputs.from_dict(road), iv, parking)
    except ValueError as e:
        raise HTTPException(400, str(e))
    leg = lr.build(scan, cap, {**road}, rec.get("image_name") or "photo", rec.get("chainage_m"))
    leg["_scan"], leg["_cap"] = scan, cap
    return lr.attach_twin(leg)


@app.post("/api/scan")
async def scan(image: UploadFile = File(...), road: str = Form("{}"), trapezoid: str = Form(""),
               depth: str = Form("")):
    road_d = _parse_json(road, {})
    trap = _parse_json(trapezoid, None)
    tmp = DATA / f"upload_{uuid.uuid4().hex}"
    await _read_upload(image, MAX_IMAGE_MB, tmp)
    img = cv2.imdecode(np.fromfile(str(tmp), np.uint8), cv2.IMREAD_COLOR)
    tmp.unlink(missing_ok=True)
    if img is None:
        raise HTTPException(400, "Not a readable image (use JPG or PNG).")
    return _scan_image(img, road_d, trap, (depth == "1") if depth else None, image.filename or "photo")


@app.post("/api/corridor")
async def corridor(images: List[UploadFile] = File(...), road: str = Form("{}"),
                   spacing_m: float = Form(100.0), start_chainage_m: float = Form(0.0), depth: str = Form("")):
    """Several photos along one road (earlier version's batch mode). Photo i is
    at chainage start + i x spacing. The corridor is governed by its weakest
    section (bottleneck rule)."""
    from rca.legacy import result as lr
    from rca.legacy.roadrunner_export import corridor_capacity_summary
    road_d = _parse_json(road, {})
    if not 1 <= len(images) <= 30:
        raise HTTPException(400, "Upload 1 to 30 photos.")
    cid = uuid.uuid4().hex
    cdir = DATA / "corridors" / cid
    cdir.mkdir(parents=True)
    sections, errors = [], []
    for i, f in enumerate(images):
        tmp = cdir / f"in_{i}"
        await _read_upload(f, MAX_IMAGE_MB, tmp)
        img = cv2.imdecode(np.fromfile(str(tmp), np.uint8), cv2.IMREAD_COLOR)
        tmp.unlink(missing_ok=True)
        if img is None:
            errors.append({"image": f.filename, "error": "not a readable image"})
            continue
        ch = start_chainage_m + i * spacing_m
        rec = _scan_image(img, {**road_d, "chainage_m": ch}, None, (depth == "1") if depth else None,
                          f.filename or f"photo_{i + 1}", ch)
        sections.append({"scan_id": rec["id"], "image": rec["image_name"], "chainage_m": ch,
                         "annotated_url": rec["annotated_url"],
                         "capacity_hcm_pcu_h": rec["capacity"]["capacity_hcm_pcu_h"],
                         "reduced_capacity_pcu_h": rec["capacity"]["capacity_defect_pcu_h"],
                         "capacity_loss_pct": rec["legacy"]["capacity_loss_pct"],
                         "defects_found": list(rec["legacy"]["per_defect"].keys()),
                         "lane_score": (rec["scan"]["lane_markings"] or {}).get("visibility_index")})
    legs = [_legacy_from_record(_load(_dir("scans", s["scan_id"]) / "record.json"), None) for s in sections]
    summary = {**lr.corridor_summary(legs), **({"corridor": corridor_capacity_summary(legs)} if legs else {})}
    out = {"id": cid, "road": road_d, "spacing_m": spacing_m, "sections": sections, "errors": errors,
           "summary": summary}
    _save(cdir / "corridor.json", out)
    return out


EXPORT_TYPES = {"defects.xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "roadrunner.xodr": "application/xml", "roadrunner-ideal.xodr": "application/xml",
                "roadrunner-capacity.json": "application/json", "department-report.pdf": "application/pdf",
                "bundle.zip": "application/zip"}
BUNDLE_STATIC = ["run_twin_safely.m", "run_ideal_digital_twin.bat", "run_nonideal_digital_twin.bat",
                 "run_2d_digital_twin.bat", "auto_import_roadrunner.m", "add_vehicles_ideal.m",
                 "add_vehicles_nonideal.m", "build_traffic.m", "add_defects_to_scene.m", "twin_config.m",
                 "twin_assets.m", "roadtwin.m", "create_digital_twin_model.m", "run_digital_twin.m",
                 "DIGITAL_TWIN_README.md"]
BUNDLE_HOWTO = ("1. Extract this zip into an empty folder. Keep every file together.\r\n"
                "2. Double-click run_2d_digital_twin.bat first: it needs only MATLAB\r\n"
                "   and shows the ideal vs current road animation.\r\n"
                "3. For RoadRunner: double-click run_ideal_digital_twin.bat, then\r\n"
                "   run_nonideal_digital_twin.bat. The first run asks for your RoadRunner\r\n"
                "   project folder and remembers it (roadrunner_project.txt).\r\n"
                "4. If MATLAB is not found, set MATLAB_EXE to your matlab.exe path.\r\n"
                "5. If anything fails, a popup appears and the details are in twin_log.txt.\r\n")
MATLAB_DIR = ROOT / "matlab_twin"


class ExportRequest(BaseModel):
    scan_id: Optional[str] = None
    corridor_id: Optional[str] = None
    road: Optional[dict] = None
    accepted: Optional[List[int]] = None
    site_label: Optional[str] = None


def _export_bytes(kind: str, results: List[dict], stem: str, site_label: str) -> bytes:
    import io
    import tempfile
    import zipfile
    from rca.legacy import roadrunner_export as rr
    from rca.legacy import result as lr
    corridor = len(results) > 1
    worst = max(results, key=lambda r: r["capacity_loss_pct"])
    if kind == "roadrunner.xodr":
        return (rr.build_corridor_xodr(results) if corridor else rr.build_single_road_xodr(results[0])).encode()
    if kind == "roadrunner-ideal.xodr":
        return (rr.build_ideal_corridor_xodr(results) if corridor else rr.build_ideal_road_xodr(results[0])).encode()
    if kind == "roadrunner-capacity.json":
        data = rr.corridor_capacity_summary(results) if corridor else lr.capacity_sidecar(results[0])
        return json.dumps(data, indent=2, default=str).encode()
    with tempfile.TemporaryDirectory() as td:
        if kind == "defects.xlsx":
            from rca.legacy.excel_export import export_defects_excel
            from openpyxl import load_workbook
            path = f"{td}/x.xlsx"
            export_defects_excel(worst if not corridor else results[0], path)
            if corridor:                      # one Defects sheet per section
                wb = load_workbook(path)
                for r in results[1:]:
                    p2 = f"{td}/y.xlsx"
                    export_defects_excel(r, p2)
                    src = load_workbook(p2)["Defects"]
                    ws = wb.create_sheet(f"Defects ch {r['road_config'].get('chainage_m', '')}"[:31])
                    for row in src.iter_rows(values_only=True):
                        ws.append(row)
                wb["Defects"].title = f"Defects ch {results[0]['road_config'].get('chainage_m', '')}"[:31]
                wb.save(path)
            return open(path, "rb").read()
        if kind == "department-report.pdf":
            from rca.legacy.department_extensions import generate_department_report_pdf
            path = f"{td}/d.pdf"
            kw = {}
            if corridor:
                kw = {"appendix_title": "Corridor sections (main tables describe the bottleneck section)",
                      "appendix_headers": ["Chainage (m)", "Photo", "Capacity incl. extensions (PCU/h)", "Loss (%)"],
                      "appendix_rows": [{"Chainage (m)": r["road_config"].get("chainage_m"), "Photo": r["image"],
                                         "Capacity incl. extensions (PCU/h)": r["reduced_capacity_pcu_h"],
                                         "Loss (%)": r["capacity_loss_pct"]} for r in results]}
            generate_department_report_pdf(worst, path, site_label=site_label, **kw)
            return open(path, "rb").read()
        if kind == "bundle.zip":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name in ("roadrunner.xodr", "roadrunner-ideal.xodr", "roadrunner-capacity.json"):
                    zf.writestr(f"{stem}_{name.replace('-', '_')}",
                                _export_bytes(name, results, stem, site_label))
                for name in BUNDLE_STATIC:
                    if (MATLAB_DIR / name).is_file():
                        zf.write(MATLAB_DIR / name, arcname=name)
                zf.writestr("HOW_TO_USE.txt", BUNDLE_HOWTO)
            return buf.getvalue()
    raise HTTPException(404, "Unknown export.")


@app.post("/api/legacy")
def legacy(req: ExportRequest):
    """Per-defect factors, IRC actions and overall guidance for the accepted
    detections and current road inputs (recomputed when the engineer edits)."""
    if not req.scan_id:
        raise HTTPException(400, "scan_id needed.")
    leg = _legacy_from_record(_load(_dir("scans", req.scan_id) / "record.json"), req.accepted, req.road)
    return _legacy_summary(leg)


@app.post("/api/export/{kind}")
def export(kind: str, req: ExportRequest):
    """Earlier version's per-analysis files: defect Excel log, RoadRunner
    .xodr (with and without defects), capacity JSON, department-routed PDF,
    and the MATLAB/RoadRunner digital-twin bundle. Works for one scan or a corridor."""
    from fastapi.responses import Response
    if kind not in EXPORT_TYPES:
        raise HTTPException(404, "Unknown export.")
    if req.corridor_id:
        c = _load(_dir("corridors", req.corridor_id) / "corridor.json")
        results = [_legacy_from_record(_load(_dir("scans", s["scan_id"]) / "record.json"), None, req.road or None)
                   for s in c["sections"]]
        if not results:
            raise HTTPException(400, "Corridor has no analysed photos.")
        stem, label = "corridor", req.site_label or (req.road or c["road"]).get("road_name") or "Corridor"
    elif req.scan_id:
        rec = _load(_dir("scans", req.scan_id) / "record.json")
        results = [_legacy_from_record(rec, req.accepted, req.road)]
        stem = Path(rec.get("image_name") or "photo").stem[:40] or "photo"
        label = req.site_label or (req.road or rec["road"]).get("road_name") or rec.get("image_name") or "photo"
    else:
        raise HTTPException(400, "Give scan_id or corridor_id.")
    for r in results:
        r.pop("_scan", None), r.pop("_cap", None)
    data = _export_bytes(kind, results, re.sub(r"[^A-Za-z0-9_-]+", "_", stem), label)
    name = f"{re.sub(r'[^A-Za-z0-9_-]+', '_', stem)}_{kind}"
    return Response(data, media_type=EXPORT_TYPES[kind],
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.post("/api/video")
async def video(bg: BackgroundTasks, video: UploadFile = File(...), line_a: str = Form(...),
                line_b: str = Form(""), trap_length_m: str = Form(""), road_type: str = Form("divided_4lane"),
                speed_limit_kmh: str = Form("")):
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
    limit = float(speed_limit_kmh) if speed_limit_kmh else None
    if limit is not None and not lb:
        raise HTTPException(400, "The speed check needs line B and the trap length.")
    _save(d / "job.json", {"id": jid, "status": "queued", "progress": 0.0})
    bg.add_task(_run_video, jid, str(d / f"input{ext}"), la, lb, trap, road_type, limit)
    return {"job_id": jid}


def _run_video(jid, path, la, lb, trap, road_type, limit=None):
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
            vc = counter()
            res = vc.run(path, la, lb, trap, road_type, max_seconds=MAX_VIDEO_S,
                                annotate_path=ann, progress=prog, speed_limit_kmh=limit,
                                evidence_dir=str(d / "evidence") if limit else None)
            res["trap_length_m"] = trap
            for ev in (res.get("speed_check") or {}).get("over_limit", []):
                if ev.get("evidence_image"):
                    ev["evidence_image"] = f"/files/jobs/{jid}/evidence/{Path(ev['evidence_image']).name}"
            if ann and Path(ann).exists():
                res["annotated_url"] = f"/files/jobs/{jid}/annotated.mp4"
            _save(d / "job.json", {"id": jid, "status": "done", "progress": 1.0, "result": res})
        except Exception as e:  # noqa: BLE001 - report any failure to the user
            _save(d / "job.json", {"id": jid, "status": "error", "error": str(e)})
        finally:
            Path(path).unlink(missing_ok=True)
            vc = None
            _trim_memory()


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
    scan, img, leg = None, None, None
    if req.scan_id:
        d = _dir("scans", req.scan_id)
        rec = _load(d / "record.json")
        leg = _legacy_from_record(rec, req.accepted, req.road)
        scan, cap = leg.pop("_scan"), leg.pop("_cap")
        from rca.vision.road_scan import RoadScanner
        orig = cv2.imread(str(d / "original.jpg"))
        img = str(d / "report_annotated.jpg")
        cv2.imwrite(img, RoadScanner.annotate(orig, scan, rec.get("trapezoid")), [cv2.IMWRITE_JPEG_QUALITY, 88])
    else:
        try:
            cap = capmod.evaluate(capmod.RoadInputs.from_dict(req.road))
        except ValueError as e:
            raise HTTPException(400, str(e))
    traffic = None
    if req.job_id:
        traffic = _load(_dir("jobs", req.job_id) / "job.json").get("result")
    tw = twin(req.twin) if req.twin else None
    out = DATA / "reports"
    out.mkdir(exist_ok=True)
    path = out / f"{uuid.uuid4().hex}.pdf"
    rep.build(str(path), cap, scan, img, traffic, tw, road_name=req.road.get("road_name", ""),
              metrics=MODEL_METRICS, legacy=leg)
    return FileResponse(path, media_type="application/pdf", filename="road_capacity_report.pdf")


@app.get("/files/{kind}/{rid}/{name}")
def files(kind: str, rid: str, name: str):
    if kind not in ("scans", "jobs") or name not in ("annotated.jpg", "original.jpg", "annotated.mp4"):
        raise HTTPException(404)
    p = _dir(kind, rid) / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/files/jobs/{rid}/evidence/{name}")
def evidence(rid: str, name: str):
    if not re.fullmatch(r"vehicle_\d+\.jpg", name):
        raise HTTPException(404)
    p = _dir("jobs", rid) / "evidence" / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


# ------------------------------------------------------------------
# Field validation (Indo-HCM Annex 5C-5E), earlier version's module
# ------------------------------------------------------------------
@app.post("/api/field/analyse")
async def field_analyse(file: UploadFile = File(...), trap_m: float = Form(...),
                        road_type: str = Form(...), pcu_method: str = Form("table")):
    """vehicles.csv: vehicle_type,t_entry_s,t_exit_s (docs/field_templates/)."""
    from rca import field_validation as fv
    if road_type not in hcm.ROAD_TYPES or pcu_method not in ("table", "se"):
        raise HTTPException(400, "Unknown road_type or pcu_method.")
    tmp = DATA / f"field_{uuid.uuid4().hex}.csv"
    await _read_upload(file, 20, tmp)
    try:
        return capmod._json_safe(fv.analyse_site(str(tmp), trap_m, road_type, pcu_method))
    except (ValueError, KeyError, ZeroDivisionError) as e:
        raise HTTPException(400, f"Could not analyse survey file: {e}")
    finally:
        tmp.unlink(missing_ok=True)


class FieldFromVideo(BaseModel):
    job_id: str
    road_type: str
    pcu_method: str = "table"


@app.post("/api/field/from-video")
def field_from_video(req: FieldFromVideo):
    """Methods A and B on the automatic video survey (vehicles that crossed
    both trap lines), instead of a hand-made vehicles.csv."""
    from rca import field_validation as fv
    j = _load(_dir("jobs", req.job_id) / "job.json")
    res = j.get("result")
    if not res:
        raise HTTPException(400, "That video has not finished counting.")
    if not res.get("trap_length_m"):
        raise HTTPException(400, "This video was counted without line B / trap length; speeds are needed.")
    if req.road_type not in hcm.ROAD_TYPES or req.pcu_method not in ("table", "se"):
        raise HTTPException(400, "Unknown road_type or pcu_method.")
    try:
        out = fv.analyse_vehicles(fv.vehicles_from_video(res["events"]), res["trap_length_m"],
                                  req.road_type, req.pcu_method)
    except ValueError as e:
        raise HTTPException(400, str(e))
    out["source"] = "automatic video count (COCO vehicle classes; autos not separated)"
    return capmod._json_safe(out)


class ComparePairs(BaseModel):
    predicted: List[float]
    observed: List[float]


@app.post("/api/field/compare")
def field_compare(body: ComparePairs):
    """Dashboard-predicted vs field values (capacity, or capacity loss %)."""
    from rca import field_validation as fv
    if len(body.predicted) != len(body.observed):
        raise HTTPException(400, "predicted and observed must have the same length.")
    try:
        return fv.error_metrics(body.predicted, body.observed)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/field/template/{name}")
def field_template(name: str):
    if name not in ("vehicles_template.csv", "validation_template.csv"):
        raise HTTPException(404)
    return FileResponse(ROOT / "docs" / "field_templates" / name, media_type="text/csv", filename=name)


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "web" / "static"), html=True), name="static")
