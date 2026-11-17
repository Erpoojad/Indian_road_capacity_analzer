"""
rca.legacy.result
=================
Builds the earlier version's analysis-result dictionary from the new
pipeline (scan + capacity.evaluate), so the earlier export modules —
RoadRunner .xodr, Excel defect log, department PDF — run unchanged with the
same logic. Keys mirror road_analyzer/core.py analyse_image().
"""
from __future__ import annotations

from typing import Dict, List, Optional

from rca.hcm import actions
from rca.hcm import extensions as ext
from rca.hcm import indo_hcm as hcm


def build(scan: Optional[dict], cap: dict, road: dict, image_name: str = "",
          chainage_m: Optional[float] = None) -> dict:
    rt = hcm.ROAD_TYPES[cap["road_type"]]
    width = cap["width_m"]
    objects = [o for o in (scan or {}).get("objects", []) if o["class"] in actions.OBSTRUCTION_CLASSES]
    per_defect = actions.per_defect(objects, width, cap["c0_pcu_h"], rt["divided"],
                                    road.get("width_method") or "proportional", cap["parking_class_used"])
    c0, c_hcm, c_ext = cap["c0_pcu_h"], cap["capacity_hcm_pcu_h"], cap["capacity_defect_pcu_h"]
    f = cap["factors"]
    hcm_f = {k: round(f[k], 3) for k in ("F_OP", "F_BS", "F_AC")}
    ext_f = {k: round(f[k], 3) for k in ("F_W", "F_R")}
    fstr = lambda fs: " x ".join(f"{k} {v:.3f}" for k, v in fs.items())
    los = None
    if cap.get("volume_pcu_h") is not None:
        los = {"observed_volume_pcu_h": cap["volume_pcu_h"], "vc_base": cap.get("vc_base"),
               "los_base": cap.get("los_base"), "vc_hcm": cap["vc_hcm"], "los_hcm": cap["los_hcm"],
               "vc_ext": cap["vc_defect"], "los_ext": cap["los_defect"], "table": cap["los_table"]}
    avg_pcu = ext.avg_pcu_per_vehicle(cap["road_type"])
    loss_pct = round(cap["loss_vs_base_pct"], 1)
    wf = cap["width_factor"]
    rr = [{"defect_type": o["class"], "left_m": o["left_m"], "right_m": o["right_m"],
           "width_m": o["width_m"], "conf": o["conf"], "bbox_px": o["box"]} for o in objects]
    cfg = {**road, "total_width_m": width, "num_lanes": rt["lanes"], "image": image_name}
    if chainage_m is not None:
        cfg["chainage_m"] = chainage_m
    return {
        "image": image_name or "photo",
        "road_config": cfg,
        "capacity_basis": {
            "standard": hcm.SOURCE, "road_type": cap["road_type"], "road_type_label": rt["label"],
            "direction_basis": rt["basis"], "divided": rt["divided"], "indo_hcm_chapter": rt["chapter"],
            "dsv_vc_ratio": rt["dsv_vc_ratio"], "base_capacity_method": cap["c0_method"],
            "base_capacity_detail": cap["steps"][0]["detail"],
            "operating_speed_kmh": road.get("operating_speed_kmh"),
            "operating_speed_source": "measured operating speed (base section)"
            if road.get("operating_speed_kmh") else "not measured",
            "width_method": wf["method"], "warnings": cap["notes"],
        },
        "capacity_calculation": {
            "base_capacity_pcu_h": round(c0, 1), "indo_hcm_factors": hcm_f, "extension_factors": ext_f,
            "factors": {**hcm_f, **ext_f}, "parking_class_used": cap["parking_class_used"],
            "roughness": cap["roughness"],
            "worst_pothole_size": (per_defect.get("pothole", {}).get("size_summary") or {}).get("worst_size_class"),
            "total_width_m": round(width, 2), "obstruction_width_m": round(wf["obstruction_width_m"], 2),
            "blocked_incl_shy_m": round(wf["lost_width_m"], 2), "effective_width_m": round(wf["usable_width_m"], 2),
            "formula_indo_hcm": f"C_HCM = {c0:.0f} x {fstr(hcm_f)} = {c_hcm:.0f} PCU/h (Eq. 5.5)",
            "formula": f"C = {c_hcm:.0f} x {fstr(ext_f)} = {c_ext:.0f} PCU/h (with extensions)",
        },
        "base_capacity_pcu_h": round(c0, 1),
        "indo_hcm_capacity_pcu_h": round(c_hcm, 1),
        "reduced_capacity_pcu_h": round(c_ext, 1),
        "capacity_loss_pcu_h": round(c0 - c_ext, 1),
        "capacity_loss_pct": loss_pct,
        "indo_hcm_loss_pct": round(cap["indo_hcm_loss_pct"], 1),
        "dsv_los_c_ideal_pcu_h": round(c0 * rt["dsv_vc_ratio"], 0),
        "dsv_los_c_reduced_pcu_h": round(c_ext * rt["dsv_vc_ratio"], 0),
        "indicative_vehicles": {
            "avg_pcu_per_vehicle": round(avg_pcu, 3),
            "base_capacity_veh_h": round(c0 / avg_pcu, 0), "reduced_capacity_veh_h": round(c_ext / avg_pcu, 0),
            "note": "ASSUMED traffic mix with Indo-HCM Table 5.3 PCUs. Replace with a classified count."},
        "level_of_service": los,
        "operating_speed_kmh": road.get("operating_speed_kmh"),
        "overall_guidance": actions.get_overall_guidance(loss_pct),
        "effective_width_m": round(wf["usable_width_m"], 2),
        "vehicle_veto_suppressed": ((scan or {}).get("detector") or {}).get("vehicle_vetoed", 0),
        "per_defect": per_defect,
        "roadrunner_obstacles": rr,
        "lane_markings": (scan or {}).get("lane_markings"),
    } | _twin_fields()


def _twin_fields() -> dict:
    return {}


def attach_twin(result: dict) -> dict:
    """Earlier version's macroscopic twin (Greenshields + deterministic queue +
    LWR shockwave); its numbers feed the MATLAB / RoadRunner bundle. None until
    a volume and a measured operating speed are entered."""
    from rca.legacy.digital_twin_engine import generate_twin_data
    try:
        result["digital_twin"] = generate_twin_data(result)
    except Exception:  # noqa: BLE001
        result["digital_twin"] = None
    return result


def capacity_sidecar(result: dict) -> dict:
    """roadrunner_capacity.json: the numbers the MATLAB / RoadRunner scripts read."""
    from rca.legacy.roadrunner_export import get_defect_positions
    calc = result.get("capacity_calculation") or {}
    twin = result.get("digital_twin")
    return {
        "units": "PCU/h unless stated",
        "base_capacity_pcu_h": result.get("base_capacity_pcu_h"),
        "indo_hcm_capacity_pcu_h": result.get("indo_hcm_capacity_pcu_h"),
        "reduced_capacity_pcu_h": result.get("reduced_capacity_pcu_h"),
        "capacity_loss_pct": result.get("capacity_loss_pct"),
        "operating_speed_kmh": result.get("operating_speed_kmh"),
        "avg_pcu_per_vehicle": (result.get("indicative_vehicles") or {}).get("avg_pcu_per_vehicle"),
        "factors": calc.get("factors"),
        "total_width_m": calc.get("total_width_m"),
        "blocked_incl_shy_m": calc.get("blocked_incl_shy_m"),
        "num_lanes": (result.get("road_config") or {}).get("num_lanes"),
        "divided": (result.get("capacity_basis") or {}).get("divided"),
        "defects_found": list((result.get("per_defect") or {}).keys()),
        "defect_positions": get_defect_positions(result),
        "twin": None if twin is None else {k: twin[k] for k in ("demand_pcu_h", "demand_source", "ideal", "defect")},
    }


def corridor_summary(results: List[dict]) -> dict:
    """Earlier version's _summarise: a stretch is limited by its weakest
    section, so the bottleneck (minimum) capacity governs; the median loss
    describes how widespread the loss is."""
    if not results:
        return {"worst_capacity_loss_pct": None, "worst_image_or_frame": None,
                "median_capacity_loss_pct": None, "bottleneck_capacity_pcu_h": None}
    losses = sorted(r["capacity_loss_pct"] for r in results)
    worst = max(results, key=lambda r: r["capacity_loss_pct"])
    n = len(losses)
    median = losses[n // 2] if n % 2 else (losses[n // 2 - 1] + losses[n // 2]) / 2
    return {"worst_capacity_loss_pct": worst["capacity_loss_pct"], "worst_image_or_frame": worst["image"],
            "median_capacity_loss_pct": round(median, 1),
            "bottleneck_capacity_pcu_h": worst["reduced_capacity_pcu_h"]}
