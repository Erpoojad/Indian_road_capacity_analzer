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
