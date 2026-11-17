"""Tests carried over from the earlier version for the restored features:
detection rules (tiles, thresholds file, cross-class, veto), MiDaS depth
indicator, field validation (methods A/B/C, error metrics), IRC actions,
parking assumption, legacy exports and speed checks."""
import random

import numpy as np
import pytest

from rca import field_validation as fv
from rca.hcm import actions
from rca.hcm import capacity as cap
from rca.hcm import indo_hcm as hcm
from rca.vision import depth as dep
from rca.vision import road_scan as rs


# ---------------- detection rules ----------------
def test_cross_class_duplicate_keeps_more_confident():
    dets = [{"name": "barricade", "conf": 0.8, "box": (0, 0, 100, 100)},
            {"name": "illegal_parking", "conf": 0.6, "box": (5, 5, 100, 100)},
            {"name": "pothole", "conf": 0.7, "box": (200, 200, 260, 240)},
            {"name": "vehicle", "conf": 0.9, "box": (0, 0, 100, 100)}]
    kept, dropped = rs.suppress_cross_class(dets)
    assert dropped == 1 and sorted(d["name"] for d in kept) == ["barricade", "pothole", "vehicle"]


def test_vehicle_veto_drops_cart_on_vehicle():
    dets = [{"name": "cart", "conf": 0.9, "box": (0, 0, 100, 100)},
            {"name": "vehicle", "conf": 0.5, "box": (10, 10, 110, 110)},
            {"name": "garbage", "conf": 0.6, "box": (0, 0, 100, 100)}]
    kept, vetoed = rs.vehicle_veto(dets)
    assert vetoed == 1 and sorted(d["name"] for d in kept) == ["garbage", "vehicle"]


def test_class_thresholds_file_overrides_defaults(tmp_path):
    (tmp_path / "class_thresholds.json").write_text('{"pothole": 0.3, "cart": 5}')
    t = rs.load_class_thresholds(str(tmp_path / "defects.onnx"))
    assert t["pothole"] == 0.3 and t["cart"] == rs.CONF_THRESHOLDS["cart"]


def test_tile_windows_cover_image_with_overlap():
    wins = rs.tile_windows(1600, 1200)
    assert len(wins) == 4
    assert min(w[0] for w in wins) == 0 and max(w[2] for w in wins) == 1600
    assert min(w[1] for w in wins) == 0 and max(w[3] for w in wins) == 1200
    assert wins[1][0] < wins[0][2]


def test_merge_tiled_rules():
    W, H = 1600, 1200
    wins = rs.tile_windows(W, H)
    full = [{"name": "pothole", "conf": 0.6, "box": [100, 900, 400, 1000]},
            {"name": "barricade", "conf": 0.9, "box": [800, 300, 1000, 700]}]
    t0 = wins[0]
    tile_dets = [{"name": "pothole", "conf": 0.5, "box": [200, 200, 240, 220]},
                 {"name": "pothole", "conf": 0.5, "box": [t0[2] - t0[0] - 40, 100, t0[2] - t0[0] - 1, 130]},
                 {"name": "barricade", "conf": 0.9, "box": [10, 10, 100, 200]}]
    t2 = wins[2]
    tile2 = [{"name": "pothole", "conf": 0.7, "box": [150 - t2[0], 920 - t2[1], 380 - t2[0], 990 - t2[1]]}]
    out = rs.merge_tiled(full, [(t0, tile_dets), (t2, tile2)], (W, H))
    pot = [d for d in out if d["name"] == "pothole"]
    assert len(pot) == 2
    assert [200, 200, 240, 220] in [list(d["box"]) for d in pot]
    big = [d for d in pot if d["box"][0] < 200][0]
    assert big["conf"] == 0.7 and big.get("from_tile")
    assert [d["name"] for d in out].count("barricade") == 1


# ---------------- MiDaS depth indicator ----------------
def _road(h=200, w=300, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    return 5.0 + 0.04 * yy + 0.005 * xx + rng.normal(0, 0.02, (h, w))


def test_depth_indicator_flat_road_not_flagged_despite_gradient():
    assert dep.depth_indicator(_road(), (120, 80, 180, 120))["depth_class"] == "not_distinct"


def test_depth_indicator_grades_depressions():
    shallow, deep = _road(), _road()
    shallow[85:115, 125:175] -= 0.07
    deep[85:115, 125:175] -= 0.40
    assert dep.depth_indicator(shallow, (120, 80, 180, 120))["depth_class"] == "shallow"
    d = dep.depth_indicator(deep, (120, 80, 180, 120))
    assert d["depth_class"] == "deep" and d["depression_area_pct"] > 50


def test_midas_preprocess_matches_official_transform_size():
    assert dep.preprocess(np.zeros((300, 533, 3), np.uint8)).shape == (1, 3, 160, 256)
    assert dep.preprocess(np.zeros((720, 720, 3), np.uint8)).shape == (1, 3, 256, 256)


# ---------------- field validation ----------------
def test_operating_speed_filters_free_cars():
    speeds = [40 + i * 0.2 for i in range(150)]
    heads = [10 if i % 2 else 3 for i in range(150)]
    r = hcm.operating_speed(speeds, heads)
    assert r["n_free_cars"] == 75 and r["warning"]


def test_greenshields_recovers_known_capacity():
    vf, kj = 60.0, 200.0
    rnd = random.Random(1)
    periods = []
    for k in [10, 20, 35, 50, 65, 80, 90, 100, 110]:
        v = vf * (1 - k / kj) + rnd.uniform(-0.5, 0.5)
        periods.append({"density_pcu_km": k, "sms_kmh": v, "flow_pcu_h": k * v})
    assert fv.greenshields_fit(periods)["capacity_pcu_h"] == pytest.approx(3000, rel=0.03)


def test_error_metrics():
    m = fv.error_metrics([110, 90], [100, 100])
    assert m["mape_pct"] == pytest.approx(10) and m["bias"] == pytest.approx(0) and m["n"] == 2


def test_first_vehicle_headway_not_counted_as_free():
    vs = [{"type": "car", "t_in": 0.0, "t_out": 5.0}, {"type": "car", "t_in": 2.0, "t_out": 7.0}]
    fv.add_speeds_and_headways(vs, 60)
    assert vs[0]["headway_s"] is None
    assert fv.capacity_from_operating_speed(vs, "divided_4lane")["capacity_pcu_h"] is None


def test_partial_last_period_dropped():
    vs = [{"type": "car", "t_in": float(t), "t_out": t + 4.0} for t in range(0, 420, 2)]
    fv.add_speeds_and_headways(vs, 60)
    periods = fv.five_minute_periods(vs, "divided_4lane", 60)
    assert len(periods) == 1 and periods[0]["flow_pcu_h"] == pytest.approx(1800)


def test_queue_discharge_method_c():
    r = fv.capacity_from_queue_discharge([{"flow_pcu_h": 2000}, {"flow_pcu_h": 2200}])
    assert r["capacity_pcu_h"] == 2100 and r["periods"] == 2


def test_field_validation_from_video_events():
    events = [{"class": "car", "t_a": 10.0 + 12 * i, "t_b": 13.0 + 12 * i} for i in range(120)]
    events.append({"class": "car", "t_a": 5.0})                       # never reached line B
    rows = fv.vehicles_from_video(events)
    assert len(rows) == 120
    r = fv.analyse_vehicles(rows, 60, "divided_4lane")
    assert r["method_A"]["operating_speed_kmh"] == pytest.approx(72.0)  # 60 m / 3 s
    assert r["method_A"]["capacity_pcu_h"] == pytest.approx((20.11 * 72 + 210.8) * 2)


# ---------------- IRC actions, parking rule ----------------
def test_irc_actions_and_guidance():
    assert actions.get_irc_action("pothole", 0, "large")["severity"] == "URGENT"
    assert actions.get_irc_action("pothole", 0, "small")["severity"] == "ROUTINE"
    assert actions.get_irc_action("garbage", 3)["severity"] == "MONITOR"
    assert actions.get_irc_action("garbage", 20)["severity"] == "URGENT"
    assert actions.get_overall_guidance(30)["band"] == "Significant"


def test_detected_parking_assumes_mildest_class():
    r = cap.evaluate(cap.RoadInputs(road_type="divided_4lane"), [], parking_detected=True)
    assert r["parking_class_used"] == "len_lt25" and r["capacity_hcm_pcu_h"] == pytest.approx(2700 * 0.70)


def test_legacy_exports_run():
    from rca.legacy import department_extensions as de, excel_export as xe, result as lr, roadrunner_export as rr
    scan = {"objects": [{"class": "pothole", "conf": 0.8, "box": [0, 0, 10, 10], "left_m": 1.0, "right_m": 1.6,
                         "width_m": 0.6, "on_carriageway": True, "irc82_size": "large"}]}
    road = {"road_type": "divided_4lane", "width_m": 7.0}
    c = cap.evaluate(cap.RoadInputs.from_dict(road), [(1.0, 1.6)])
    res = lr.build(scan, c, road, "x.jpg")
    assert res["per_defect"]["pothole"]["severity"] == "URGENT"
    assert "<OpenDRIVE>" in rr.build_single_road_xodr(res)
    import tempfile, os
    d = tempfile.mkdtemp()
    xe.export_defects_excel(res, os.path.join(d, "a.xlsx"))
    de.generate_department_report_pdf(res, os.path.join(d, "a.pdf"))
    assert os.path.getsize(os.path.join(d, "a.pdf")) > 1000
