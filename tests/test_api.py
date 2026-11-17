"""API smoke tests with the real ONNX models."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "defects.onnx").exists(), reason="models missing")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    os.environ["RCA_DATA_DIR"] = str(tmp_path_factory.mktemp("data"))
    import importlib
    import rca.api
    importlib.reload(rca.api)
    return TestClient(rca.api.app)


def test_options_and_health(client):
    assert client.get("/api/health").json()["ok"]
    o = client.get("/api/options").json()
    assert len(o["road_types"]) == 7 and "bus_stop" in o["factors"]


def test_scan_capacity_report_roundtrip(client):
    img = np.full((480, 640, 3), 90, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    r = client.post("/api/scan", files={"image": ("r.jpg", buf.tobytes(), "image/jpeg")},
                    data={"road": json.dumps({"road_type": "divided_4lane", "width_m": 7})})
    assert r.status_code == 200, r.text
    rec = r.json()
    assert rec["capacity"]["capacity_hcm_pcu_h"] == 2700 and rec["capacity"]["vc_defect"] is None
    rep = client.post("/api/report", json={"road": {"road_type": "divided_4lane"}, "scan_id": rec["id"], "accepted": []})
    assert rep.status_code == 200 and rep.content[:4] == b"%PDF"


def test_rejects_bad_ids_and_inputs(client):
    assert client.get("/api/jobs/..%2F..%2Fetc").status_code in (400, 404)
    assert client.get("/files/scans/abc/annotated.jpg").status_code in (400, 404)
    assert client.post("/api/capacity", json={"road": {"road_type": "x"}}).status_code == 400
    r = client.post("/api/scan", files={"image": ("r.jpg", b"not an image", "image/jpeg")}, data={"road": "{}"})
    assert r.status_code == 400


def test_twin_endpoint(client):
    r = client.post("/api/twin", json={"capacity_clear_pcu_h": 2700, "capacity_defect_pcu_h": 2000,
                                       "vf_kmh": 50, "demand_pcu_h": 2300})
    assert r.status_code == 200 and r.json()["summary"]["breakdown"] is True


def _jpg(h=480, w=640):
    img = np.full((h, w, 3), 90, np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_all_exports_for_a_scan(client):
    road = {"road_type": "divided_4lane", "width_m": 7, "volume_pcu_h": 1800, "operating_speed_kmh": 55}
    sample = (ROOT / "rca" / "web" / "static" / "samples" / "potholes.jpg").read_bytes()
    rec = client.post("/api/scan", files={"image": ("p.jpg", sample, "image/jpeg")},
                      data={"road": json.dumps(road), "depth": "1"}).json()
    pd = rec["legacy"]["per_defect"]
    assert pd["pothole"]["severity"] == "URGENT" and "rectification" in pd["pothole"]
    assert any("depth_indicator" in o for o in rec["scan"]["objects"])
    sigs = {"defects.xlsx": b"PK", "roadrunner.xodr": b"<?xml", "roadrunner-ideal.xodr": b"<?xml",
            "roadrunner-capacity.json": b"{", "department-report.pdf": b"%PDF", "bundle.zip": b"PK"}
    for kind, sig in sigs.items():
        r = client.post(f"/api/export/{kind}", json={"scan_id": rec["id"], "road": road})
        assert r.status_code == 200, (kind, r.text[:200])
        assert r.content[:len(sig)] == sig, kind
    import io, zipfile
    names = zipfile.ZipFile(io.BytesIO(client.post("/api/export/bundle.zip", json={"scan_id": rec["id"]}).content)).namelist()
    assert "run_2d_digital_twin.bat" in names and any(n.endswith("roadrunner_capacity.json") for n in names)
    cap = json.loads(client.post("/api/export/roadrunner-capacity.json", json={"scan_id": rec["id"], "road": road}).content)
    assert cap["twin"] is not None                               # volume + speed entered -> twin numbers


def test_corridor_bottleneck_and_exports(client):
    files = [("images", ("a.jpg", _jpg(), "image/jpeg")),
             ("images", ("b.jpg", (ROOT / "rca/web/static/samples/potholes.jpg").read_bytes(), "image/jpeg"))]
    c = client.post("/api/corridor", files=files, data={"road": json.dumps({"road_type": "divided_4lane"}),
                                                        "spacing_m": "50"}).json()
    assert [s["chainage_m"] for s in c["sections"]] == [0, 50]
    assert c["summary"]["bottleneck_capacity_pcu_h"] == pytest.approx(min(s["reduced_capacity_pcu_h"] for s in c["sections"]), abs=0.1)
    for kind in ("roadrunner.xodr", "department-report.pdf", "defects.xlsx", "roadrunner-capacity.json"):
        assert client.post(f"/api/export/{kind}", json={"corridor_id": c["id"]}).status_code == 200


def test_field_endpoints(client):
    rows = "vehicle_type,t_entry_s,t_exit_s\n" + "".join(f"car,{10 + 12 * i},{13 + 12 * i}\n" for i in range(120))
    r = client.post("/api/field/analyse", files={"file": ("v.csv", rows.encode(), "text/csv")},
                    data={"trap_m": "60", "road_type": "divided_4lane"}).json()
    assert r["method_A"]["operating_speed_kmh"] == pytest.approx(72.0)
    m = client.post("/api/field/compare", json={"predicted": [110, 90], "observed": [100, 100]}).json()
    assert m["mape_pct"] == pytest.approx(10)
    assert client.get("/api/field/template/vehicles_template.csv").status_code == 200
