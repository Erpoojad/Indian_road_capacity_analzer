"""Vehicle counting (synthetic vehicles with known speeds), tracker, lane
markings, perspective mapping and the digital twin."""
import cv2
import numpy as np
import pytest

from rca.traffic import counter as ctr
from rca.traffic.tracker import ByteTrackLite, Track
from rca.twin import ctm
from rca.vision import lanes
from rca.vision.road_scan import Edges, to_metres

FPS = 25.0
LINE_A = [[0, 300], [1000, 300]]
LINE_B = [[0, 500], [1000, 500]]


def synthetic_tracks(n=5, px_per_frame=10.0, gap_frames=40):
    """n cars moving straight down at constant speed, one every gap_frames."""
    tr = ByteTrackLite()
    frames = n * gap_frames + 120
    for f in range(frames):
        dets = []
        for i in range(n):
            y = (f - i * gap_frames) * px_per_frame
            if 0 < y < 800:
                x = 100 + 150 * (i % 4)
                dets.append({"box": [x, y - 60, x + 80, y], "name": "car", "conf": 0.9})
        tr.step(dets, f)
    return [t for t in tr.finish() if t.hits >= ctr.MIN_HITS]


def test_each_vehicle_counted_once_with_exact_speed():
    tracks = synthetic_tracks()
    res = ctr.summarise(tracks, FPS, duration_s=12.0, line_a=LINE_A, line_b=LINE_B, trap_m=20.0,
                        road_type="divided_4lane", meta={})
    assert res["total_vehicles"] == 5
    d = res["directions"]["down"]
    assert d["counts"] == {"car": 5}
    # 200 px between lines at 10 px/frame = 20 frames = 0.8 s for 20 m -> 90 km/h
    assert d["space_mean_speed_kmh"] == pytest.approx(90.0, abs=0.5)
    assert d["flow_pcu_h"] == pytest.approx(5 * 3600 / 12.0)
    assert d["density_pcu_km"] == pytest.approx(d["flow_pcu_h"] / 90.0, rel=0.01)


def test_crossing_geometry():
    assert ctr.crossing((5, 0), (5, 10), (0, 5), (10, 5)) == pytest.approx(0.5)
    assert ctr.crossing((5, 0), (5, 4), (0, 5), (10, 5)) is None
    assert ctr.crossing((50, 0), (50, 10), (0, 5), (10, 5)) is None   # outside the segment


def test_truck_split_by_size():
    def mk(cls, size):
        t = Track([0, 0, size, size], cls, 0.9, 0)
        return t
    tracks = [mk("car", 50) for _ in range(3)] + [mk("truck", 60), mk("truck", 120)]
    cl = ctr.classify(tracks)
    assert sorted(cl[t.id] for t in tracks[3:]) == ["lcv", "tat"]


def test_tracker_keeps_identity_through_short_occlusion():
    tr = ByteTrackLite(max_lost=5)
    for f in range(30):
        dets = [] if 10 <= f < 13 else [{"box": [100, 10 * f, 180, 10 * f + 60], "name": "car", "conf": 0.9}]
        tr.step(dets, f)
    assert len(tr.finish()) == 1


def test_perspective_mapping():
    # road 1000 px wide at the bottom (y=1000), 200 px wide at the top (y=0)
    e = Edges(1200, 1000, [(100, 1000), (1100, 1000), (700, 0), (500, 0)])
    l, r = to_metres([500, 0, 600, 0], e, 7.0)       # at the far edge: 100 px of 200 px
    assert (l, r) == pytest.approx((0.0, 3.5))
    l, r = to_metres([100, 900, 600, 1000], e, 7.0)  # near: 500 px of 1000 px
    assert (l, r) == pytest.approx((0.0, 3.5))


DATA = __import__("pathlib").Path(__file__).parent / "data"


def test_lane_synthetic_solid_and_broken_lines_and_shadows():
    rng = np.random.default_rng(0)
    road = np.clip(rng.normal(80, 12, (400, 600, 3)), 0, 255).astype(np.uint8)   # textured asphalt
    road[:, 400:] = (road[:, 400:] * 0.45).astype(np.uint8)                     # hard shadow edge
    img = road.copy()
    cv2.line(img, (120, 399), (270, 200), (235, 235, 235), 7)                   # solid white
    xb = lambda y: int(480 - (399 - y) * 0.4)
    for y in range(399, 200, -50):                                              # broken yellow
        cv2.line(img, (xb(y), y), (xb(y - 25), y - 25), (40, 200, 230), 6)
    trap = [(0, 399), (599, 399), (420, 200), (180, 200)]
    r = lanes.analyse(img, trap)
    kinds = sorted((ln["colour"], ln["type"]) for ln in r["lines"])
    assert kinds == [("white", "solid"), ("yellow", "broken")]
    assert lanes.analyse(road, trap)["n_lines"] == 0                           # texture + shadow: no paint


def test_lane_real_photos():
    good = lanes.analyse(cv2.imread(str(DATA / "lane_yellow_white.jpg")),
                         [(300, 545), (1030, 545), (560, 405), (455, 405)])
    colours = {ln["colour"] for ln in good["lines"]}
    assert colours == {"white", "yellow"} and good["condition"] in ("fair", "good")
    white = [ln for ln in good["lines"] if ln["colour"] == "white"][0]
    assert white["type"] == "solid" and 700 < white["p_bottom"][0] < 780
    for f in ("lane_none_shadow.jpg", "lane_none_wide.jpg"):                    # unmarked roads
        assert lanes.analyse(cv2.imread(str(DATA / f)))["n_lines"] == 0


def test_bonnet_found_in_dashcam_photo():
    img = cv2.imread(str(DATA / "lane_none_shadow.jpg"))
    assert 500 < lanes.bonnet_row(img) < 570
    assert lanes.bonnet_row(np.full((300, 400, 3), 90, np.uint8)) == 300


def test_twin_bottleneck_throughput_equals_reduced_capacity():
    r = ctm.compare(2700, 2025, 50, 2400, duration_s=1800)
    assert r["obstructed"]["throughput_pcu_h_last5min"] == pytest.approx(2025, rel=0.01)
    assert r["clear"]["throughput_pcu_h_last5min"] == pytest.approx(2400, rel=0.01)
    assert r["summary"]["breakdown"] and r["obstructed"]["max_queue_m"] > 0


def test_twin_no_queue_below_capacity_and_greenshields_speed():
    r = ctm.simulate(2700, 50, 1350, duration_s=1800)
    assert r["max_queue_m"] == 0
    # Greenshields at q/C = 0.5: v = vf/2 (1 + sqrt(0.5)) = 42.7 km/h -> 84.4 s per km
    assert r["mean_travel_time_s"] == pytest.approx(1000 / (42.68 / 3.6), rel=0.03)
