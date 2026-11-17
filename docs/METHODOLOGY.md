# Methodology

**Indian Road Capacity Analyzer: an AI-assisted, human-in-the-loop framework for
defect-adjusted capacity estimation of urban roads using Indo-HCM 2017.**

```
 Road photo ──► defect detector (YOLOv8, ONNX) ──► engineer accepts/rejects ──► widths across road (perspective-corrected)
                                                                                     │
 Road data (type, width, parking, bus stop, access) ──► Indo-HCM C0 ──► Eq. 5.5 C ──► × F_W × F_R ──► defect-adjusted C
                                                                                     │
 Traffic video ──► vehicle detector + tracker ──► counts, PCU/h, speed, density ──► engineer confirms volume ──► V/C, LOS
                                                                                     │
                                         Digital twin (CTM + Greenshields): queue, delay, throughput, clear vs obstructed
```

Code map: `rca/hcm/indo_hcm.py` (the manual only), `rca/hcm/extensions.py` (project
additions), `rca/hcm/capacity.py` (the procedure), `rca/vision/` (photo analysis),
`rca/traffic/` (video survey), `rca/twin/ctm.py` (digital twin).

---

## 1. Scope (Indo-HCM 2017 Chapter 5, Section 5.1)

- Urban mid-block segments between controlled intersections (signals ≥ 1 km apart).
- Two-lane undivided and 4/6/8/10-lane divided roads; single-lane and intermediate-lane
  roads from Chapter 2 (interurban values applied to an urban street — stated on every result).
- Plain terrain. Base section (Table 5B.1): paved width 7.0/10.5/14.0/17.5 m per direction (±0.5 m).

## 2. Capacity procedure (Indo-HCM, `capacity.evaluate`)

| Step | Formula | Source |
|---|---|---|
| Base capacity C0 | Table 5.4 (2400 / 2700 / 4200 / 6800 / 10000 PCU/h), or divided roads with a measured operating speed: `C_lane = 20.11 V + 210.8`, × lanes | Table 5.4, Eq. 5.4 |
| Indo-HCM capacity | `C = C0 × F_OP × F_BS × F_AC` | Eq. 5.5, Tables 5.9–5.12 |
| Design service volume | `DSV = 0.70 C` | Table 5.4 |
| Flow | classified count × Table 5.3 median PCU | Tables 5.2/5.3 |
| LOS | V/C bands | Tables 5.6 (undivided) / 5.7 (divided) |

Every step, value and source is printed in the dashboard and the PDF.

## 3. Project extensions (not Indo-HCM)

Indo-HCM has **no factor** for potholes, vendors, carts, garbage or barricades.

`C_defect = C_IndoHCM × F_W × F_R`

**F_W — effective width.** Clear road: drivers keep 0.25 m from each kerb (5.1.2), so the
clear effective width is W − 0.5 m. Each accepted obstruction removes its width plus 0.25 m
on each side; overlaps (with each other or with a kerb band) count once.
`F_W = usable effective width / clear effective width`, or (divided roads)
`F_W = (C0 − lost width × 428.6) / C0`, 428.6 PCU/h per metre being the slope between the
four- and six-lane rows of Table 5.4. Parked vehicles never enter F_W: Indo-HCM's F_OP already
includes the width they occupy. Each object feeds exactly one factor.

**F_R — roughness** (only with measured IRI **and** operating speed): Chapter 2 Eq. 2.15 in
relative form above IRI 3, converted to capacity with Eq. 5.4.

**Pothole size** (IRC:82-2015, by width) sets repair priority only.

## 4. Photo analysis (`rca/vision`)

- **Detector:** YOLOv8s trained on the project's Roboflow datasets (7 classes: barricade,
  pothole, illegal_parking, street_vendor, cart, garbage, vehicle), run with ONNX Runtime.
  Validation: precision 0.81, recall 0.74, mAP50 0.74, mAP50-95 0.55 (`models/defects_metrics.json`).
- **Cleaning:** per-class confidence thresholds; one object is never counted under two
  labels (IoU > 0.6 → keep the more confident); a vendor/cart box that is really a moving
  vehicle is dropped.
- **Perspective:** the engineer may mark the carriageway as a trapezoid. Left and right road
  edges are interpolated row by row, and each object is measured at its bottom edge (where it
  touches the road), so near and far objects are scaled correctly. Without the trapezoid the
  photo width is taken as the carriageway width (stated on the result).
- **Human in the loop:** every detection can be rejected; capacity is recomputed from the
  accepted ones only, and the report shows only accepted detections.

## 5. Lane markings (`rca/vision/lanes.py`)

No lane-marking dataset exists yet, so detection is deterministic image processing:

1. **Road region:** the marked carriageway, or automatically: below the horizon and above the
   car bonnet/dashboard (found as the topmost strong brightness step in the lower image).
2. **Paint pixels:** symmetric local filter `r(x) = 2L(x) − L(x−t) − L(x+t) − |L(x−t) − L(x+t)|`
   (Nieto et al., 2010 style); `t` grows with perspective. A pixel is paint when `r ≥ max(24, 0.2 L)`
   and its colour is white (low saturation) or yellow. Shadow edges are bright on one side only and cancel.
3. **Shape filter:** keep blobs that are long (≥ 1/14 of the road region height), thin (length/width ≥ 4)
   and narrower than 3 % of the photo width — gravel speckle and sun patches are removed.
4. **Lines:** Hough segments (≥ 20° from horizontal) grouped by where they cut two reference rows;
   groups shorter than 20 % of the region height (wipers, debris, arrows) are rejected.
5. **Per line:** colour, type (solid if ≥ 70 % painted, else broken), contrast (paint lightness /
   adjacent asphalt), score = 50 % contrast term `clip((ratio − 1.05)/0.4)` + 50 % continuity term
   (painted share / expected share: solid 1.0, broken 0.4). Road score = mean of lines; no line → 0.

Checked on photos from the project's dataset: the solid white and solid yellow lines of a marked
street are found; two unmarked streets (one with heavy tree shadows) return no line (tests).
**Uncalibrated, and never changes capacity** (Indo-HCM has no lane-marking factor). IRC:35 rates
markings by retro-reflectivity, which a photo cannot measure — calibrate before quoting as a rating.
Without marked road edges, kerbs may be counted as lines.

## 6. Traffic survey from video (`rca/traffic`)

- **Layout** (Annex 5C): count line A across the road; line B a known trap length downstream.
- **Detection:** COCO-pretrained YOLOv8n (car, motorcycle, bus, truck, bicycle), ~10 frames/s.
- **Tracking:** ByteTrack-style two-stage IoU association with constant-velocity prediction
  (Hungarian matching); low-confidence boxes extend tracks but never start one.
- **Counting:** a vehicle is counted once per line, when the bottom-centre of its box crosses
  the line; crossing time is interpolated between frames. Direction from the crossing sense.
- **Classes:** car → car, motorcycle → two-wheeler, bus → bus, bicycle → bicycle, truck → LCV
  or TAT by size (≥ 2.5 × median car box area → TAT). **COCO has no auto-rickshaw class**:
  autos are counted as car or two-wheeler until the vehicle model is retrained on Indian data (IDD).
- **Flow:** veh/h and PCU/h (Table 5.3 medians), per direction; peak 5-min flow × 12 for videos ≥ 10 min.
- **Speeds:** trap speed per vehicle; space-mean speed `D / mean(t)` (Eq. 5E.2); operating speed =
  85th percentile of free cars (headway > 8 s, ≥ 100 cars, 5.4.5).
- **Density:** `k = q / v_s` (PCU/km).
- **Volume rule:** the measured volume is used in V/C only after the engineer clicks
  "Use as volume". Nothing is assumed.

## 7. Digital twin (`rca/twin/ctm.py`)

Cell Transmission Model (Daganzo 1994) with the Greenshields relation used in Indo-HCM Annex 5E.4:
`v = vf (1 − k/kj)`, `q = vf k (1 − k/kj)`, `C = vf kj / 4`. For each road state `kj = 4C / vf`.
The obstructed cells use the defect-adjusted capacity (free-flow speed unchanged, jam density
reduced in proportion). CFL = 1 (`vf·dt = dx`). Clear and obstructed roads are simulated under
the same demand; outputs: time–space density and speed, queue length, throughput, travel time
and delay from cumulative curves. Checked against theory in the tests: bottleneck throughput
equals the reduced capacity; uncongested travel time equals the Greenshields speed.
Demand and free-flow speed must come from the survey — the twin does not run on assumed values.

## 8. Validation plan (for the report and paper)

1. **Detector:** report the validation metrics above; for the paper, add a held-out test set of
   your own Nashik photos (role `test` in `training/class_map.yaml` concept) — the honest figure.
2. **Counting:** compare automatic counts with a manual count of the same video (error %, by class).
3. **Speeds:** compare trap speeds with a radar gun or manual stopwatch timing on 20+ vehicles.
4. **Capacity loss:** at obstructed and clear sections of the same road, measure queue discharge
   flow; report MAPE / RMSE / bias of the predicted capacity loss.

## 9. Limitations

- Indo-HCM values are national medians from small samples (Table 5.1).
- F_W, F_R and the lane-marking index are assumptions until validated in the field.
- One photo gives one cross-section; parking intensity is not visible in a photo.
- Videos shorter than 5 minutes give extrapolated hourly flows.

---

## 10. Errors found in the printed manual

Checked against the manual's own tables and arithmetic (carried over from the project's
earlier verification; the tests in `tests/test_hcm.py` reproduce the corrected figures).

| Location | Printed | Correct |
|---|---|---|
| Eq. 5E.4 (p. 5-30) | PCU = (V_i/V_c)/(A_i/A_c) | (V_c/V_i)/(A_c/A_i). As printed, a bus (area ratio 3.96) would need to travel 18× faster than a car to reach Table 5.3's 4.60; the corrected form gives 4.60 at a realistic V_c/V_i = 1.16 |
| **Example 5.5.2** (p. 5-20) | MAT 12 × 5.90 = **710**; total 5228 PCU/h; V/C 0.77 → **LOS D** | 12 × 5.90 = 70.8; total ≈ **4582** PCU/h; V/C 0.67 → **LOS C** |
| Example 5.5.1 (p. 5-18) | total 3153 PCU/h | Table 5.3 medians give 3206; conclusion (LOS C, then F with bus stop) unchanged |
| Example 5.5.4 (p. 5-23) | total 1459; spare 433 | Table 5.3 medians give 1454 → spare ≈ 436 PCU/h |
| Example 5.5.3 (p. 5-21) | TAT 3.22 in the expression | table uses 3.90; the 1971 total matches 3.90 |
| Table 5.3 (p. 5-6) | two-lane cycle rickshaw median "-" | Table 5.2 median 1.80 used |
| Shares in Eq. 5.2/5.3 | "Percentage composition" | must be fractions (Example 5.5.1 uses 0.360) |
| Eq. 2.11 (p. 2-16) | intermediate lane "around 2150" at 75 km/h | 830 + 16.4 × 75 = 2060; the stated 2150 is used as base |
