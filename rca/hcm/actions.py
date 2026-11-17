"""
rca.hcm.actions
===============
Maintenance / enforcement actions per detected defect and the overall
capacity-loss guidance (ported unchanged from the earlier version's core.py),
plus the per-defect breakdown: the factor each class would produce on its own.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from rca.hcm import extensions as ext
from rca.hcm import indo_hcm as hcm

OBSTRUCTION_CLASSES = ["barricade", "pothole", "illegal_parking", "street_vendor", "cart", "garbage"]

# ================================================================
# Maintenance / enforcement actions per defect class.
# Pothole tiers key off IRC:82-2015 size class; the other classes key off
# their own capacity loss using project thresholds (5% / 15%), which are
# NOT taken from any code and are labelled as such in the output.
# ================================================================
LOSS_TIERS = [(0, 5, "MONITOR"), (5, 15, "ROUTINE"), (15, 100.0001, "URGENT")]

IRC_ACTION_RULES: Dict[str, dict] = {
    "pothole": {
        "code_ref": "IRC:82-2015 cl. 7.5.3 (potholes), cl. 5.3 (prompt rectification)",
        "by_size": {
            "small":  ("ROUTINE", "Fill with premix (cold or hot mix) per IRC:82 cl. 7.5.3.5 at the "
                                  "next maintenance round; log in the condition survey."),
            "medium": ("URGENT",  "Cut to regular shape, clean, tack coat and fill with premix per "
                                  "IRC:82 cl. 7.5.3.5. Distress affecting riding quality should be "
                                  "rectified straightaway (cl. 5.3)."),
            "large":  ("URGENT",  "Rectify straightaway (IRC:82 cl. 5.3). Check whether the cavity "
                                  "extends into the base course; if so repair the base before "
                                  "surfacing. Barricade and sign the location per IRC:SP:55."),
        },
    },
    "illegal_parking": {
        "code_ref": "Motor Vehicles Act 1988 s.122; IRC:67 (signs), IRC:35 (markings)",
        "tiers": {
            "MONITOR": "Repaint No-Parking markings and check signage.",
            "ROUTINE": "Peak-hour enforcement by traffic wardens; install No-Parking signs.",
            "URGENT":  "Towing enforcement under MV Act s.122; consider bollards on the stretch.",
        },
    },
    "street_vendor": {
        "code_ref": "Street Vendors (Protection of Livelihood and Regulation of Street Vending) Act, 2014",
        "tiers": {
            "MONITOR": "Record vendor occupancy for Town Vending Committee review.",
            "ROUTINE": "Refer to the Town Vending Committee for relocation to a vending zone.",
            "URGENT":  "Relocation drive coordinated with the Town Vending Committee.",
        },
    },
    "cart": {
        "code_ref": "Street Vendors Act, 2014; municipal bye-laws on loading/unloading",
        "tiers": {
            "MONITOR": "Monitor; no action from this survey alone.",
            "ROUTINE": "Restrict carts to designated zones / off-peak hours.",
            "URGENT":  "Remove from carriageway; designate an off-street loading bay.",
        },
    },
    "garbage": {
        "code_ref": "Solid Waste Management Rules, 2016",
        "tiers": {
            "MONITOR": "Schedule clearance in the next collection round.",
            "ROUTINE": "Request priority clearance from the SWM department.",
            "URGENT":  "Immediate clearance; provide a collection point to stop recurrence.",
        },
    },
    "barricade": {
        "code_ref": "IRC:SP:55-2014 (traffic management in work zones)",
        "tiers": {
            "MONITOR": "Verify the work zone is permitted and signed per IRC:SP:55.",
            "ROUTINE": "Reduce barricaded width to the minimum needed; add diversion signs.",
            "URGENT":  "Coordinate with the executing agency to remove/relocate the barricade.",
        },
    },
}

# Overall capacity-loss bands: project reporting thresholds, not from a code.
OVERALL_CAPACITY_LOSS_GUIDANCE = [
    (0,   10,       "Minor",       "Low capacity loss. Continue routine monitoring."),
    (10,  25,       "Moderate",    "Noticeable loss. Schedule the per-defect actions."),
    (25,  50,       "Significant", "Significant loss. Prioritise urgent-tier defects; re-survey."),
    (50,  75,       "Severe",      "Over half the capacity lost. Escalate to traffic/municipal authority."),
    (75,  100.0001, "Critical",    "Road operating at a fraction of capacity. Multi-agency action needed."),
]


def tier_from_loss(loss_pct: float) -> str:
    for lo, hi, tier in LOSS_TIERS:
        if lo <= loss_pct < hi:
            return tier
    return LOSS_TIERS[-1][2]


def get_overall_guidance(capacity_loss_pct: float) -> dict:
    pct = max(0.0, float(capacity_loss_pct))
    for lo, hi, band, action in OVERALL_CAPACITY_LOSS_GUIDANCE:
        if lo <= pct < hi:
            return {"band": band, "action": action, "basis": "project reporting thresholds"}
    _, _, band, action = OVERALL_CAPACITY_LOSS_GUIDANCE[-1]
    return {"band": band, "action": action, "basis": "project reporting thresholds"}


def get_irc_action(defect_name: str, loss_pct: float, size_class: Optional[str] = None) -> dict:
    rule = IRC_ACTION_RULES.get(defect_name)
    if rule is None:
        return {"code_ref": "N/A", "severity": "INVESTIGATE",
                "action": "No standard action mapped — flag for manual inspection.",
                "severity_basis": "unmapped class"}
    if "by_size" in rule:
        sev, action = rule["by_size"].get(size_class or "small", rule["by_size"]["small"])
        return {"code_ref": rule["code_ref"], "severity": sev, "action": action,
                "severity_basis": f"IRC:82 size class ({size_class or 'small'}, width-only proxy)"}
    tier = tier_from_loss(loss_pct)
    return {"code_ref": rule["code_ref"], "severity": tier, "action": rule["tiers"][tier],
            "severity_basis": "capacity-loss project thresholds (5% / 15%)"}




def per_defect(objects: List[dict], width_m: float, c0: float, divided: bool, width_method: str,
               parking_class_used: str) -> Dict[str, dict]:
    """Per class: count, the factor that class ALONE would produce, its capacity
    loss, and the IRC/municipal action (earlier version's per_defect table)."""
    rank = {"small": 0, "medium": 1, "large": 2}
    groups: Dict[str, List[dict]] = {}
    for o in objects:
        if o["class"] in OBSTRUCTION_CLASSES and o.get("on_carriageway", True):
            groups.setdefault(o["class"], []).append(o)
    worst = None
    for o in groups.get("pothole", []):
        sc = o.get("irc82_size") or ext.pothole_size_class(o["width_m"])
        if worst is None or rank[sc] > rank[worst]:
            worst = sc
    out: Dict[str, dict] = {}
    for name, objs in groups.items():
        iv = [(o["left_m"], o["right_m"]) for o in objs]
        if name in ext.WIDTH_CLASSES:
            fw = ext.width_factor(iv, width_m, c0, divided, width_method)
            factor, blocked = fw["F_W"], fw["lost_width_m"]
            mech, basis = "F_W usable width (project extension)", "extension"
        elif name == "illegal_parking":
            pc = parking_class_used if parking_class_used != "none" else "len_lt25"
            factor = hcm.PARKING_FACTORS[pc]["factor"]
            blocked = min(ext.union_length(iv), width_m)
            mech, basis = f"F_OP Indo-HCM Table {hcm.PARKING_FACTORS[pc]['table']}", "Indo-HCM"
        else:
            factor, blocked, mech, basis = 1.0, 0.0, "none", "-"
        loss = (1 - factor) * 100
        irc = get_irc_action(name, loss, worst if name == "pothole" else None)
        rec = {"count": len(objs), "blocked_m": round(blocked, 2), "factor": round(factor, 3),
               "mechanism": mech, "factor_basis": basis,
               "capacity_loss_pcu": round(c0 * (1 - factor), 1), "capacity_loss_pct": round(loss, 1),
               "severity": irc["severity"], "severity_basis": irc["severity_basis"],
               "code_ref": irc["code_ref"], "action": irc["action"],
               "detections": [{k: o.get(k) for k in ("conf", "width_m", "left_m", "right_m", "box",
                                                     "irc82_size", "found_by", "depth_indicator")
                               if o.get(k) is not None} for o in objs]}
        if name == "pothole":
            cls = [o.get("irc82_size") for o in objs]
            rec["size_summary"] = {
                "worst_size_class": worst,
                "counts": {k: cls.count(k) for k in ("small", "medium", "large")},
                "classification": "IRC:82-2015 cl. 7.5.3.4 width thresholds (depth not measured); "
                                  "used for repair priority only",
                "found_in_zoomed_tiles": sum(1 for o in objs if o.get("found_by") == "zoomed tile")}
            from rca.legacy.pothole_rectification import recommend_pothole_rectification
            r = recommend_pothole_rectification(worst or "small", len(objs))
            rec["rectification"] = r.__dict__
            di = [o["depth_indicator"] for o in objs if o.get("depth_indicator")]
            if di:
                from rca.vision.depth import DEPTH_BANDS, DEPTH_NOTE
                rec["depth_summary"] = {"counts": {c: sum(1 for x in di if x["depth_class"] == c)
                                                   for _, c in DEPTH_BANDS}, "note": DEPTH_NOTE}
        out[name] = rec
    return out
