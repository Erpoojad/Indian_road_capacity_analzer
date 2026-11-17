"""
rca.report
==========
Department report (PDF, A4) with the full working: inputs, detections,
every capacity step with its Indo-HCM source, LOS, lane markings, traffic
survey, digital-twin result, notes and limitations.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

INK = colors.HexColor("#1f2937")
ACCENT = colors.HexColor("#b45309")
GRID = colors.HexColor("#d1d5db")
HEAD = colors.HexColor("#f3f4f6")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, textColor=INK, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=ACCENT, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12, textColor=INK)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=7.5, leading=10, textColor=colors.HexColor("#4b5563"))


def _table(rows, widths, header=True):
    t = Table([[Paragraph(str(c), BODY) for c in r] for r in rows], colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), HEAD))
    t.setStyle(TableStyle(st))
    return t


def _f(v, nd=0, unit=""):
    if v is None:
        return "—"
    return f"{v:,.{nd}f}{(' ' + unit) if unit else ''}"


def build(path: str, cap: dict, scan: Optional[dict] = None, image_path: Optional[str] = None,
          traffic: Optional[dict] = None, twin: Optional[dict] = None, road_name: str = "",
          metrics: Optional[dict] = None, legacy: Optional[dict] = None) -> str:
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title="Road Capacity Assessment", author="Indian Road Capacity Analyzer")
    W = A4[0] - 32 * mm
    s = []
    s.append(Paragraph("Road Capacity Assessment Report", H1))
    s.append(Paragraph(f"{road_name or 'Unnamed road segment'} &nbsp;·&nbsp; "
                       f"{dt.datetime.now().strftime('%d %b %Y, %H:%M')} &nbsp;·&nbsp; "
                       f"Method: Indo-HCM 2017 (CSIR-CRRI) Chapter {cap['chapter']}", SMALL))
    s.append(Spacer(1, 6))

    # Key results
    rows = [["Result", "Clear road (Indo-HCM)", "With detected defects"],
            ["Capacity (PCU/h, " + cap["basis"] + ")", _f(cap["capacity_hcm_pcu_h"]), _f(cap["capacity_defect_pcu_h"])],
            ["Design service volume (PCU/h)", _f(cap["dsv_hcm_pcu_h"]), _f(cap["dsv_defect_pcu_h"])],
            ["Capacity loss", "—", _f(cap["capacity_loss_pct"], 1, "%")]]
    if cap.get("volume_pcu_h") is not None:
        rows += [["Traffic volume (PCU/h)", _f(cap["volume_pcu_h"]), _f(cap["volume_pcu_h"])],
                 ["V/C", _f(cap["vc_hcm"], 2), _f(cap["vc_defect"], 2)],
                 [f"Level of service (Table {cap['los_table']})", cap["los_hcm"], cap["los_defect"]]]
    else:
        rows.append(["V/C and LOS", "No volume entered", "No volume entered"])
    s.append(Paragraph("1. Summary", H2))
    s.append(_table(rows, [W * 0.4, W * 0.3, W * 0.3]))

    s.append(Paragraph("2. Road segment", H2))
    f = cap["factors"]
    s.append(_table([["Item", "Value"],
                     ["Road type", f"{cap['road_label']} ({cap['lanes']} lane(s), {cap['basis']})"],
                     ["Paved width", _f(cap["width_m"], 2, "m")],
                     ["Side friction (Eq. 5.5)", f"F_OP {f['F_OP']:.2f} · F_BS {f['F_BS']:.2f} · F_AC {f['F_AC']:.2f}"],
                     ["Base capacity method", cap["c0_method"]]], [W * 0.35, W * 0.65]))

    if scan:
        s.append(Paragraph("3. Detected obstructions", H2))
        if image_path:
            try:
                img = Image(image_path)
                r = min(W / img.imageWidth, 85 * mm / img.imageHeight)
                img.drawWidth, img.drawHeight = img.imageWidth * r, img.imageHeight * r
                s.append(img)
            except Exception:
                pass
        objs = [o for o in scan["objects"] if o["class"] != "vehicle"]
        if objs:
            rows = [["Object", "Conf.", "Across road (m)", "Width (m)", "Acts through"]]
            for o in objs:
                cls = o["class"] + (f" ({o['irc82_size']}, IRC:82)" if o.get("irc82_size") else "")
                rows.append([cls, f"{o['conf']:.2f}", f"{o['left_m']:.2f} – {o['right_m']:.2f}",
                             f"{o['width_m']:.2f}", o["acts_through"] or "off carriageway"])
            s.append(_table(rows, [W * 0.28, W * 0.09, W * 0.2, W * 0.12, W * 0.31]))
        else:
            s.append(Paragraph("No obstructions detected.", BODY))
        wf = cap["width_factor"]
        s.append(Spacer(1, 4))
        s.append(Paragraph(f"Usable effective width {wf['usable_width_m']:.2f} m of {wf['clear_effective_width_m']:.2f} m "
                           f"(0.25 m shy distance at kerbs and around obstructions, Indo-HCM 5.1.2) → "
                           f"F_W = {wf['F_W']:.3f} ({wf['method']}).", BODY))
        lm = scan.get("lane_markings") or {}
        if lm.get("available"):
            s.append(Spacer(1, 4))
            s.append(Paragraph(f"<b>Lane markings:</b> {lm['n_lines']} marking line(s) found; road score "
                               f"{lm['visibility_index']:.0f}/100 ({lm['condition']}). {lm['status']}", BODY))
            if lm["lines"]:
                rows = [["Line", "Type", "Colour", "Painted share", "Contrast", "Score"]]
                for ln in lm["lines"]:
                    rows.append([f"L{ln['id']}", ln["type"], ln["colour"], f"{ln['painted_share']:.0%}",
                                 f"{ln['contrast_ratio']:.2f}", f"{ln['score']:.0f} ({ln['condition']})"])
                s.append(_table(rows, [W * 0.1, W * 0.15, W * 0.15, W * 0.18, W * 0.14, W * 0.28]))

    if legacy and legacy.get("per_defect"):
        g = legacy.get("overall_guidance") or {}
        s.append(Paragraph("3A. Recommended actions", H2))
        s.append(Paragraph(f"<b>Overall: {g.get('band', '-')}</b> — {g.get('action', '')} "
                           f"(capacity loss {legacy.get('capacity_loss_pct')}% of base capacity; "
                           f"{g.get('basis', '')}).", BODY))
        rows = [["Defect", "Count", "Factor alone", "Severity", "Action", "Code"]]
        for name, d in legacy["per_defect"].items():
            rows.append([name.replace("_", " "), d["count"], f"{d['factor']:.3f}", d["severity"],
                         d["action"], d["code_ref"]])
        s.append(_table(rows, [W * 0.13, W * 0.07, W * 0.1, W * 0.11, W * 0.36, W * 0.23]))
        rect = (legacy["per_defect"].get("pothole") or {}).get("rectification")
        if rect:
            s.append(Spacer(1, 3))
            s.append(Paragraph(f"<b>Pothole repair ({rect['category']}):</b> {rect['method']} "
                               f"Materials: {rect['materials']} Priority: {rect['priority']}. "
                               f"{rect['work_zone_note']} {rect.get('escalation_note') or ''}", SMALL))
        ds = (legacy["per_defect"].get("pothole") or {}).get("depth_summary")
        if ds:
            s.append(Paragraph("<b>Depth indicator (MiDaS):</b> " + ", ".join(f"{k} {v}" for k, v in ds["counts"].items())
                               + ". " + ds["note"], SMALL))

    s.append(Paragraph("4. Capacity calculation", H2))
    rows = [["Step", "Value", "Source", "Working"]]
    for st in cap["steps"]:
        v = st["value"]
        rows.append([st["step"], f"{v:,.0f} {st['unit']}" if st["unit"] != "-" else f"{v}", st["source"], st["detail"]])
    s.append(_table(rows, [W * 0.2, W * 0.15, W * 0.2, W * 0.45]))

    if traffic:
        s.append(Paragraph("5. Traffic survey (video)", H2))
        rows = [["Direction", "Vehicles", "Flow (veh/h)", "Flow (PCU/h)", "Space-mean speed", "Density"]]
        for d, v in traffic["directions"].items():
            rows.append([{"down": "towards camera", "up": "away from camera"}.get(d, d), ", ".join(f"{k} {n}" for k, n in v["counts"].items()), _f(v["flow_veh_h"]),
                         _f(v["flow_pcu_h"]), _f(v.get("space_mean_speed_kmh"), 1, "km/h"),
                         _f(v.get("density_pcu_km"), 1, "PCU/km")])
        s.append(_table(rows, [W * 0.12, W * 0.3, W * 0.13, W * 0.13, W * 0.16, W * 0.16]))
        s.append(Paragraph(f"Video length {traffic['duration_s']:.0f} s; PCU from Indo-HCM Table 5.3 medians. "
                           + " ".join(traffic.get("notes", [])), SMALL))
        sc = traffic.get("speed_check")
        if sc:
            s.append(Paragraph(f"<b>Speed check</b> (limit {sc['speed_limit_kmh']:g} km/h): {len(sc['over_limit'])} of "
                               f"{sc['vehicles_with_speed']} timed vehicles over the limit. {sc['note']}", BODY))
            if sc["over_limit"]:
                rows = [["Vehicle", "Speed (km/h)", "Over by", "Time in video (s)", "Status"]]
                rows += [[f"#{e['track_id']} {e['class']}", f"{e['speed_kmh']:.1f}", f"{e['over_limit_by_kmh']:.1f}",
                          f"{e['time_in_video_s']:.1f}", e["verification_status"]] for e in sc["over_limit"]]
                s.append(_table(rows, [W * 0.22, W * 0.18, W * 0.15, W * 0.2, W * 0.25]))

    if twin:
        sm = twin["summary"]
        s.append(KeepTogether([
            Paragraph("6. Digital twin (cell transmission model, Greenshields)", H2),
            Paragraph(f"Demand {twin['obstructed']['inputs']['demand_pcu_h']:.0f} PCU/h over "
                      f"{twin['obstructed']['inputs']['duration_s'] / 60:.0f} min on a "
                      f"{twin['obstructed']['inputs']['length_m']:.0f} m segment, free-flow speed "
                      f"{twin['obstructed']['inputs']['vf_kmh']:.0f} km/h. {sm['note']}", BODY),
            _table([["Measure", "Clear road", "With obstruction"],
                    ["Mean travel time (s)", _f(twin["clear"]["mean_travel_time_s"], 1), _f(twin["obstructed"]["mean_travel_time_s"], 1)],
                    ["Mean delay (s/PCU)", _f(twin["clear"]["mean_delay_s_per_pcu"], 1), _f(twin["obstructed"]["mean_delay_s_per_pcu"], 1)],
                    ["Max queue (m)", _f(twin["clear"]["max_queue_m"]), _f(twin["obstructed"]["max_queue_m"])],
                    ["Throughput, last 5 min (PCU/h)", _f(twin["clear"]["throughput_pcu_h_last5min"]),
                     _f(twin["obstructed"]["throughput_pcu_h_last5min"])]], [W * 0.4, W * 0.3, W * 0.3]),
            _queue_chart(twin, W)]))

    s.append(Paragraph("7. Notes and limitations", H2))
    notes = list(dict.fromkeys(cap.get("notes", []) + (scan or {}).get("notes", [])))
    notes += ["F_W and F_R are project extensions, not Indo-HCM factors; validate with field capacity data.",
              (f"Detector accuracy on its validation split: precision {metrics['precision']:.2f}, recall "
               f"{metrics['recall']:.2f}, mAP50 {metrics['mAP50']:.2f}. " if metrics else "")
              + "Detections were reviewed and accepted by the engineer before this calculation."]
    for n in notes:
        s.append(Paragraph("• " + n, SMALL))
    doc.build(s, onFirstPage=_footer, onLaterPages=_footer)
    return path


def _queue_chart(twin, W):
    d = Drawing(W, 120)
    lp = LinePlot()
    lp.x, lp.y, lp.width, lp.height = 40, 20, W - 60, 85
    series = []
    for key in ("clear", "obstructed"):
        qv = twin[key]["queue_m"]
        T = twin[key]["inputs"]["duration_s"] / 60
        series.append([(i * T / max(len(qv) - 1, 1), v) for i, v in enumerate(qv)] or [(0, 0)])
    lp.data = series
    lp.lines[0].strokeColor = colors.HexColor("#2563eb")
    lp.lines[1].strokeColor = ACCENT
    lp.xValueAxis.labelTextFormat = "%d"
    lp.yValueAxis.valueMin = 0
    d.add(lp)
    d.add(String(40, 110, "Queue length (m) vs time (min) — blue: clear road, amber: with obstruction",
                 fontSize=7.5, fillColor=INK))
    return d


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(16 * mm, 8 * mm, "Indian Road Capacity Analyzer · Indo-HCM 2017 · AI-assisted, human-in-the-loop")
    canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()
