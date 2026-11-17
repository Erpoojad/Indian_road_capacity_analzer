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
CH2_ROUGHNESS = {
    "single_lane":       {"cap": (945.0, 25.0), "eq": "Eq. 2.20", "speed": None},
    "intermediate_lane": {"cap": (2956.0, 199.0), "eq": "Eq. 2.19", "speed": (91.0, 6.7)},
}


def roughness_factor(iri_m_per_km: Optional[float], operating_speed_kmh: Optional[float],
                     road_type: Optional[str] = None) -> dict:
    if not iri_m_per_km or iri_m_per_km <= BASE_IRI_LIMIT:
        return {"F_R": 1.0, "speed_ratio": 1.0, "applied": False}
    v_known = operating_speed_kmh is not None and operating_speed_kmh > 0
    if road_type in CH2_ROUGHNESS:
        m = CH2_ROUGHNESS[road_type]
        a, b = m["cap"]
        f = max(a - b * iri_m_per_km, 0.0) / (a - b * BASE_IRI_LIMIT)
        if m["speed"]:
            sa, sb = m["speed"]
            r = max(sa - sb * iri_m_per_km, 0.1 * sa) / (sa - sb * BASE_IRI_LIMIT)
        else:
            r = f   # CHOICE: no single-lane speed-roughness equation; speed scaled like capacity
        return {"F_R": f, "speed_ratio": r, "applied": True,
                "reduced_speed_kmh": operating_speed_kmh * r if v_known else None,
                "source": f"Indo-HCM {m['eq']} relative to IRI {BASE_IRI_LIMIT:g}", "note": None}
    r = (104 - 6.8 * iri_m_per_km) / (104 - 6.8 * BASE_IRI_LIMIT)
    r = max(r, 0.1)
    if not v_known:
        # Eq. 5.4 needs the site's operating speed; no assumed speed is used.
        return {"F_R": 1.0, "speed_ratio": r, "applied": False, "reduced_speed_kmh": None,
                "note": f"IRI {iri_m_per_km:g} m/km was entered but the operating speed was not measured, "
                        f"so the roughness factor F_R (Eq. 5.4) is not applied. Measure the operating speed."}
    v0 = operating_speed_kmh
    f = hcm.lane_capacity_from_operating_speed(v0 * r) / hcm.lane_capacity_from_operating_speed(v0)
    note = None
    if v0 * r < hcm.SPEED_MODEL_RANGE_KMH[0]:
        note = (f"Reduced speed {v0 * r:.1f} km/h is below the 40 km/h validity of Eq. 5.4; "
                f"F_R is an extrapolation.")
    return {"F_R": f, "speed_ratio": r, "applied": True, "reduced_speed_kmh": v0 * r, "note": note}



def pothole_size_class(width_m: float) -> str:
    """IRC:82-2015 cl. 7.5.3.4 by width (depth cannot be measured from a photo).
    Maintenance priority only: capacity effect is through F_W."""
    return "small" if width_m <= 0.20 else ("medium" if width_m <= 0.50 else "large")


# ------------------------------------------------------------------
# Indicative vehicles/h (display only). ASSUMED urban mix; capacity logic is
# entirely in PCU/h. Replace with the site's classified count.
# ------------------------------------------------------------------
ASSUMED_COMPOSITION = {"two_wheeler": 0.55, "car": 0.25, "auto": 0.10, "bus": 0.03,
                       "lcv": 0.04, "tat": 0.02, "bicycle": 0.01}


def avg_pcu_per_vehicle(road_type: str) -> float:
    pcu = hcm.pcu_values(road_type)
    return sum(s * pcu.get(k, 1.0) for k, s in ASSUMED_COMPOSITION.items()) / sum(ASSUMED_COMPOSITION.values())
