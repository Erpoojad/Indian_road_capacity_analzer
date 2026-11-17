"""
rca.hcm.capacity
================
One function, evaluate(), runs the whole capacity procedure and returns every
intermediate number with its source, so the dashboard and the PDF can print
the full working.

    Step 1  C0          Table 5.4, or Eq. 5.4 from a measured operating speed
    Step 2  C_HCM       Eq. 5.5: C0 x F_OP x F_BS x F_AC            (Indo-HCM)
    Step 3  C_defect    C_HCM x F_W x F_R                  (project extension)
    Step 4  V/C, LOS    Tables 5.6 / 5.7 — only when a volume was entered

Rule (project decision): a traffic volume is used ONLY when the user entered
or confirmed one for this road. Nothing is assumed or carried over.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from rca.hcm import extensions as ext
from rca.hcm import indo_hcm as hcm


@dataclass
class RoadInputs:
    road_type: str = "divided_4lane"
    width_m: Optional[float] = None            # paved width (per direction if divided)
    on_street_parking: str = "none"
    bus_stop: str = "none"
    access_points: str = "none"
    operating_speed_kmh: Optional[float] = None  # measured at a base section (Eq. 5.4)
    iri: Optional[float] = None
    width_method: str = "proportional"
    volume_pcu_h: Optional[float] = None       # entered/confirmed by the user only
    volume_source: Optional[str] = None
    road_name: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "RoadInputs":
        def num(k):
            v = d.get(k)
            if v in (None, ""):
                return None
            v = float(v)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{k} must be a non-negative number")
            return v
        rt = d.get("road_type") or "divided_4lane"
        if rt not in hcm.ROAD_TYPES:
            raise ValueError(f"Unknown road_type '{rt}'. Valid: {list(hcm.ROAD_TYPES)}")
        for name in ("on_street_parking", "bus_stop", "access_points"):
            hcm.factor(name, d.get(name) or "none")          # validates keys
        return cls(road_type=rt, width_m=num("width_m"),
                   on_street_parking=d.get("on_street_parking") or "none",
                   bus_stop=d.get("bus_stop") or "none",
                   access_points=d.get("access_points") or "none",
                   operating_speed_kmh=num("operating_speed_kmh"), iri=num("iri"),
                   width_method=d.get("width_method") or "proportional",
                   volume_pcu_h=num("volume_pcu_h"), volume_source=d.get("volume_source"),
                   road_name=str(d.get("road_name") or "")[:120])


def evaluate(inp: RoadInputs, obstruction_intervals_m: Sequence[Tuple[float, float]] = (),
             parking_detected: bool = False) -> dict:
    rt = hcm.ROAD_TYPES[inp.road_type]
    width = inp.width_m or rt["paved_width_m"]
    notes: List[str] = []
    steps: List[dict] = []

    w_warn = hcm.check_road_width(inp.road_type, width)
    if w_warn:
        notes.append(w_warn)

    # Step 1 — base capacity
    b = hcm.base_capacity(inp.road_type, inp.operating_speed_kmh, width)
    notes += b["notes"]
    c0 = b["c0_pcu_h"]
    steps.append({"step": "Base capacity C0", "value": c0, "unit": "PCU/h",
                  "source": b["method"], "detail": b["detail"]})

    # Step 2 — Indo-HCM side friction (Eq. 5.5)
    a = hcm.actual_capacity(c0, inp.on_street_parking, inp.bus_stop, inp.access_points)
    c_hcm = a["c_pcu_h"]
    f = a["factors"]
    steps.append({"step": "Indo-HCM capacity C", "value": c_hcm, "unit": "PCU/h", "source": "Eq. 5.5",
                  "detail": f"{c0:.0f} x F_OP {f['F_OP']:.2f} x F_BS {f['F_BS']:.2f} x F_AC {f['F_AC']:.2f}"})
    if parking_detected and inp.on_street_parking == "none":
        notes.append("Parked vehicles were detected but on-street parking is set to 'none'. "
                     "Choose the Table 5.9/5.10 parking class for this segment.")

    # Step 3 — project extension: obstructions and roughness
    fw = ext.width_factor(obstruction_intervals_m, width, c_hcm, rt["divided"], inp.width_method)
    fr = ext.roughness_factor(inp.iri, inp.operating_speed_kmh, inp.road_type)
    notes += fw["notes"] + ([fr["note"]] if fr.get("note") else [])
    c_def = c_hcm * fw["F_W"] * fr["F_R"]
    steps.append({"step": "Defect-adjusted capacity", "value": c_def, "unit": "PCU/h",
                  "source": "Project extension (not Indo-HCM)",
                  "detail": f"{c_hcm:.0f} x F_W {fw['F_W']:.3f} x F_R {fr['F_R']:.3f}"})

    los_table = rt["los_table"]
    dsv_ratio = rt["dsv_vc_ratio"]
    out = {
        "road_type": inp.road_type, "road_label": rt["label"], "basis": rt["basis"],
        "width_m": width, "lanes": rt["lanes"], "chapter": rt["chapter"],
        "c0_pcu_h": c0, "c0_method": b["method"], "factors": {**f, "F_W": fw["F_W"], "F_R": fr["F_R"]},
        "capacity_hcm_pcu_h": c_hcm, "capacity_defect_pcu_h": c_def,
        "capacity_loss_pct": (1 - c_def / c_hcm) * 100 if c_hcm > 0 else 0.0,
        "dsv_hcm_pcu_h": dsv_ratio * c_hcm, "dsv_defect_pcu_h": dsv_ratio * c_def,
        "dsv_ratio": dsv_ratio, "width_factor": fw, "roughness": fr,
        "los_table": rt["los_table_ref"], "steps": steps, "notes": notes,
        "volume_pcu_h": None, "volume_source": None,
        "vc_hcm": None, "vc_defect": None, "los_hcm": None, "los_defect": None,
    }
    if inp.volume_pcu_h is not None:
        v = inp.volume_pcu_h
        vc1 = v / c_hcm if c_hcm > 0 else math.inf
        vc2 = v / c_def if c_def > 0 else math.inf
        out.update(volume_pcu_h=v, volume_source=inp.volume_source or "entered by user",
                   vc_hcm=vc1, vc_defect=vc2,
                   los_hcm=hcm.los_from_vc(vc1, table=los_table),
                   los_defect=hcm.los_from_vc(vc2, table=los_table),
                   spare_capacity_pcu_h=c_def - v)
        steps.append({"step": "V/C and LOS", "value": round(vc2, 3), "unit": "-",
                      "source": f"Table {rt['los_table_ref']}",
                      "detail": f"{v:.0f} / {c_def:.0f} = {vc2:.2f} -> LOS {out['los_defect']} "
                                f"(clear road: {vc1:.2f} -> LOS {out['los_hcm']})"})
    else:
        notes.append("No traffic volume entered for this road, so V/C and LOS are not computed.")
    return _json_safe(out)


def _json_safe(o):
    if isinstance(o, float):
        return None if not math.isfinite(o) else round(o, 4)
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def road_type_options() -> List[dict]:
    return [{"key": k, "label": v["label"], "basis": v["basis"], "lanes": v["lanes"],
             "divided": v["divided"], "paved_width_m": v["paved_width_m"],
             "capacity_pcu_h": v["capacity_pcu_h"], "chapter": v["chapter"]}
            for k, v in hcm.ROAD_TYPES.items()]


def factor_options() -> Dict[str, List[dict]]:
    return {name: [{"key": k, "label": v["label"], "factor": v["factor"], "table": v["table"]}
                   for k, v in table.items()] for name, table in hcm.FACTOR_TABLES.items()}
