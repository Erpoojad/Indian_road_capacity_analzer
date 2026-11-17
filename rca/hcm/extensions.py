"""
rca.hcm.extensions
==================
PROJECT EXTENSIONS to Indo-HCM 2017 Chapter 5. These are NOT Indo-HCM
factors. They are reported separately and must be validated with field data.

Indo-HCM's urban method (Eq. 5.5, C = C0 x F_OP x F_BS x F_AC) covers only
on-street parking, bus stops and access points. It has no factor for potholes,
street vendors, carts, garbage or barricades, and no roughness adjustment on
urban roads. This module adds:

    C_defect = C_IndoHCM x F_W x F_R

F_W  effective-width factor: physical obstructions on the carriageway.
F_R  roughness factor, applied only when IRI AND operating speed are measured.

Rule: every detected object feeds exactly ONE factor. Parked vehicles act
through Indo-HCM's own F_OP (Tables 5.9/5.10), which already includes the width
they occupy, so they are never put into F_W.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from rca.hcm import indo_hcm as hcm

# Obstructions that act through usable width. In lane-less mixed traffic,
# vehicles steer around a pothole, so its footprint plus shy distance is lost.
WIDTH_CLASSES = ("pothole", "street_vendor", "cart", "garbage", "barricade")
PARKING_CLASSES = ("illegal_parking",)

# Drivers keep the same clearance from an obstruction as from a kerb
# (Indo-HCM 5.1.2 kerb shyness 0.25 m). Adopted assumption.
OBSTRUCTION_SHY_M = hcm.KERB_SHYNESS_M

# Capacity per metre of width implied by Table 5.4 between four-lane (7.5 m
# reference width, 2700 PCU/h/dir) and six-lane (11.0 m, 4200 PCU/h/dir)
# divided rows. 8/10-lane rows carry the 1-3 km signal-spacing condition.
TABLE_5_4_SLOPE_PCU_PER_M = (4200 - 2700) / (11.0 - 7.5)   # 428.6

Interval = Tuple[float, float]


def merge(intervals: Sequence[Interval]) -> List[Interval]:
    out: List[Interval] = []
    for l, r in sorted((min(a, b), max(a, b)) for a, b in intervals):
        if out and l <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], r))
        else:
            out.append((l, r))
    return out


def union_length(intervals: Sequence[Interval]) -> float:
    return sum(r - l for l, r in merge(intervals))


def effective_width(intervals: Sequence[Interval], width_m: float,
                    shy_m: float = OBSTRUCTION_SHY_M) -> float:
    """Width usable by moving traffic. A clear road already loses a 0.25 m shy
    band at each kerb (5.1.2). An obstruction removes its own width plus a shy
    band on each side; overlaps (with each other or a kerb band) count once."""
    w = float(width_m)
    blocked = [(0.0, min(shy_m, w)), (max(0.0, w - shy_m), w)]
    blocked += [(max(0.0, l - shy_m), min(w, r + shy_m)) for l, r in intervals]
    return max(w - union_length(blocked), 0.0)


def width_factor(intervals: Sequence[Interval], width_m: float, c0: float,
                 divided: bool, method: str = "proportional") -> dict:
    """F_W for obstructions occupying [left, right] metres across the road.
      proportional : F_W = usable effective width / clear effective width
      table_slope  : F_W = (C0 - lost width x 428.6) / C0  (divided roads)"""
    intervals = [(max(0.0, l), min(width_m, r)) for l, r in intervals if r > 0 and l < width_m]
    clear = effective_width([], width_m)
    usable = effective_width(intervals, width_m)
    lost = max(clear - usable, 0.0)
    notes = []
    if method == "table_slope" and divided and c0 > 0:
        f, used = max(0.0, (c0 - lost * TABLE_5_4_SLOPE_PCU_PER_M) / c0), "table_slope"
    else:
        if method == "table_slope":
            notes.append("Table-slope method needs divided rows of Table 5.4; proportional used.")
        f, used = (usable / clear if clear > 0 else 1.0), "proportional"
    if intervals and usable < hcm.LANE_WIDTH_M:
        notes.append(f"Usable width {usable:.2f} m is below one 3.5 m lane: the section is near "
                     f"blockage and F_W is outside any calibrated range.")
    return {"F_W": f, "method": used, "obstruction_width_m": union_length(intervals),
            "lost_width_m": lost, "usable_width_m": usable, "clear_effective_width_m": clear,
            "notes": notes}


# ------------------------------------------------------------------
# Roughness. Chapter 5 has no roughness model. Chapter 2 Eq. 2.15
# (V = 104 - 6.8 IRI) is borrowed in RELATIVE form, referenced to the
# base-section limit IRI 3 (Table 5B.1), and converted to capacity with
# Eq. 5.4. Chapter 2 roads use their own Eq. 2.19 / 2.20.
# ------------------------------------------------------------------
BASE_IRI_LIMIT = 3.0
CH2_ROUGHNESS = {"single_lane": (945.0, 25.0, "Eq. 2.20"),
                 "intermediate_lane": (2956.0, 199.0, "Eq. 2.19")}


def roughness_factor(iri: Optional[float], operating_speed_kmh: Optional[float],
                     road_type: str) -> dict:
    if not iri or iri <= BASE_IRI_LIMIT:
        return {"F_R": 1.0, "applied": False, "note": None}
    if road_type in CH2_ROUGHNESS:
        a, b, eq = CH2_ROUGHNESS[road_type]
        f = max(a - b * iri, 0.0) / (a - b * BASE_IRI_LIMIT)
        return {"F_R": f, "applied": True, "note": f"Indo-HCM {eq}, relative to IRI 3."}
    if not operating_speed_kmh:
        return {"F_R": 1.0, "applied": False,
                "note": f"IRI {iri:g} entered but operating speed not measured; F_R needs Eq. 5.4 "
                        f"with the site's operating speed, so it is not applied."}
    r = max((104 - 6.8 * iri) / (104 - 6.8 * BASE_IRI_LIMIT), 0.1)
    v = operating_speed_kmh
    f = hcm.lane_capacity_from_operating_speed(v * r) / hcm.lane_capacity_from_operating_speed(v)
    note = "Eq. 2.15 (relative) + Eq. 5.4."
    if v * r < hcm.SPEED_MODEL_RANGE_KMH[0]:
        note += f" Reduced speed {v * r:.1f} km/h is below Eq. 5.4's 40 km/h range (extrapolated)."
    return {"F_R": f, "applied": True, "reduced_speed_kmh": v * r, "note": note}


def pothole_size_class(width_m: float) -> str:
    """IRC:82-2015 cl. 7.5.3.4 by width (depth cannot be measured from a photo).
    Maintenance priority only: capacity effect is through F_W."""
    return "small" if width_m <= 0.20 else ("medium" if width_m <= 0.50 else "large")
