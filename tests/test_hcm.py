"""Indo-HCM 2017 Chapter 5 tables and worked examples (Section 5.5), and the
project extensions. Run: python -m pytest -q"""
import pytest

from rca.hcm import capacity as cap
from rca.hcm import extensions as ext
from rca.hcm import indo_hcm as hcm

EX551 = {"two_wheeler": 1441, "auto": 379, "car": 1985, "mini_bus": 44, "bus": 56,
         "lcv": 15, "tat": 42, "bicycle": 21, "cycle_rickshaw": 21}
EX552 = {"two_wheeler": 2640, "auto": 858, "car": 2239, "mini_bus": 11, "bus": 81,
         "lcv": 120, "tat": 10, "mat": 12, "tractor_trailer": 16}
EX554 = {"two_wheeler": 369, "auto": 168, "car": 236, "bus": 58, "mini_bus": 20, "lcv": 78,
         "tat": 25, "mat": 75, "tractor_trailer": 2, "bicycle": 25, "cycle_rickshaw": 12}


def test_table_5_4_values():
    caps = {k: v["capacity_pcu_h"] for k, v in hcm.ROAD_TYPES.items() if v["chapter"] == 5}
    assert caps == {"undivided_2lane": 2400, "divided_4lane": 2700, "divided_6lane": 4200,
                    "divided_8lane": 6800, "divided_10lane": 10000}
    for rt in hcm.ROAD_TYPES.values():
        assert rt["dsv_pcu_h"] == pytest.approx(rt["dsv_vc_ratio"] * rt["capacity_pcu_h"])


@pytest.mark.parametrize("v,expected", [(55, 1317), (60, 1417), (65, 1518), (70, 1619), (75, 1719)])
def test_table_5_5_from_eq_5_4(v, expected):
    assert round(hcm.lane_capacity_from_operating_speed(v)) == pytest.approx(expected, abs=1)


def test_example_5_5_1():
    assert hcm.flow_pcu(EX551, "divided_6lane")["flow_pcu_h"] == pytest.approx(3205.6, abs=0.5)
    shares, n = hcm.se_shares(EX551)
    se = hcm.stream_equivalency_factor(True, shares, n)
    assert se == pytest.approx(0.755, abs=0.002) and se * n == pytest.approx(3023, abs=6)
    r = cap.evaluate(cap.RoadInputs(road_type="divided_6lane", bus_stop="kerbside_high", volume_pcu_h=3023))
    assert r["capacity_hcm_pcu_h"] == pytest.approx(2520) and r["los_hcm"] == "F"
    r = cap.evaluate(cap.RoadInputs(road_type="divided_6lane", volume_pcu_h=3023))
    assert r["los_hcm"] == "C"                                           # V/C 0.72


def test_example_5_5_2_corrected_arithmetic():
    q = hcm.flow_pcu(EX552, "divided_8lane")["flow_pcu_h"]
    assert q == pytest.approx(4582, abs=2)
    assert hcm.los_from_vc(q / 6800, divided=True) == "C"


def test_example_5_5_3_and_5_5_4():
    assert hcm.ROAD_TYPES["divided_4lane"]["dsv_pcu_h"] < 2638 <= hcm.ROAD_TYPES["divided_6lane"]["dsv_pcu_h"]
    q = hcm.flow_pcu(EX554, "divided_4lane")["flow_pcu_h"]
    assert round(hcm.ROAD_TYPES["divided_4lane"]["dsv_pcu_h"] - q) == 436


def test_los_band_edges():
    assert hcm.los_from_vc(0.15, divided=True) == "B"     # A is strict '< 0.15'
    assert hcm.los_from_vc(0.75, divided=True) == "C"
    assert hcm.los_from_vc(1.0, divided=False) == "E"
    assert hcm.los_from_vc(1.01, divided=False) == "F"


def test_width_factor_rules():
    fw = ext.width_factor([(0.0, 1.0), (0.7, 1.2)], 7.0, 2700, True)   # merged kerb obstruction
    assert fw["usable_width_m"] == pytest.approx(5.3) and fw["F_W"] == pytest.approx(5.3 / 6.5)
    assert ext.width_factor([(3.0, 3.4)], 7.0, 2700, True)["F_W"] == pytest.approx(5.6 / 6.5)
    assert ext.width_factor([], 7.0, 2700, True)["F_W"] == 1.0
    ts = ext.width_factor([(0.0, 1.0)], 7.0, 2700, True, method="table_slope")
    assert ts["F_W"] == pytest.approx((2700 - 1.0 * 1500 / 3.5) / 2700)


def test_parking_never_in_width_and_volume_never_assumed():
    r = cap.evaluate(cap.RoadInputs(road_type="divided_4lane"), [], parking_detected=True)
    assert r["factors"]["F_W"] == 1.0 and r["vc_defect"] is None and r["los_defect"] is None
    assert any("parking" in n.lower() for n in r["notes"])


def test_roughness_needs_measured_speed():
    assert ext.roughness_factor(5.0, None, "divided_4lane")["F_R"] == 1.0
    assert ext.roughness_factor(2.5, 60, "divided_4lane")["F_R"] == 1.0
    assert ext.roughness_factor(5.0, 60, "divided_4lane")["F_R"] < 1.0


def test_full_blockage_is_json_safe():
    r = cap.evaluate(cap.RoadInputs(road_type="divided_4lane", volume_pcu_h=1000), [(0, 7.0)])
    assert r["capacity_defect_pcu_h"] == 0 and r["vc_defect"] is None and r["los_defect"] == "F"


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        cap.RoadInputs.from_dict({"road_type": "nonsense"})
    with pytest.raises(ValueError):
        cap.RoadInputs.from_dict({"bus_stop": "lots"})
    with pytest.raises(ValueError):
        cap.RoadInputs.from_dict({"width_m": -3})
