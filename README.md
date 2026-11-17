# Indian Road Capacity Analyzer

**An AI-assisted and human-in-the-loop framework for defect-adjusted capacity estimation
and real-time decision making of urban roads using Indo-HCM 2017.**

Upload a road photo and the system detects potholes, street vendors, carts, garbage,
barricades and parked vehicles, and measures how much of the carriageway each one takes.
The engineer accepts or rejects every detection. The system then computes the
**Indo-HCM 2017 capacity**, the **defect-adjusted capacity**, V/C and **LOS**. A traffic
video gives the classified volume (PCU/h), speeds and density. A **digital twin** shows the
queue and delay the obstruction causes, and a **PDF report** collects all of it for the
road department.

| Part | What it does | Method |
|---|---|---|
| Defect detection | 7 classes, widths across the road, perspective-corrected | YOLOv8s (ONNX), mAP50 0.74 |
| Capacity | C0, Eq. 5.5 side friction, DSV, V/C, LOS | Indo-HCM 2017 Ch. 5 (Ch. 2 for single/intermediate lane) |
| Defect adjustment | effective-width factor F_W, roughness F_R | project extension, shown separately |
| Lane markings | each marking line: solid/broken, white/yellow, condition score 0–100 | symmetric ridge filter + shape filter + Hough (no training needed) |
| Traffic video | counts by class & direction, PCU/h, trap speeds, density, operating speed | YOLOv8n + ByteTrack-style tracker |
| Digital twin | queue, delay, throughput, time–space diagram, animation | cell transmission model + Greenshields |
| Report | A4 PDF with every step and source | ReportLab |

Full method, assumptions, validation plan and manual errata: **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**.

## Run on your computer

```bash
pip install -r requirements.txt
uvicorn rca.api:app --port 8000
# open http://localhost:8000
```

## Deploy on Render (free)

1. Push this repository to GitHub.
2. Render → **New → Blueprint** → choose the repository (it reads `render.yaml`).
3. Wait for the build; open the URL. Health check: `/api/health`.

The server uses ONNX Runtime, not PyTorch, so it fits the 512 MB free plan (peak ≈ 375 MB measured).
The free plan sleeps after 15 min idle; the first request then takes ~1 min.

## Retrain the detector (better accuracy)

Open `training/retrain_detector_colab.ipynb` in Google Colab → *Runtime → Change runtime type → T4 GPU*
→ *Run all*. It reads the datasets from your Google Drive, merges them by class **name**
(`training/class_map.yaml`), converts polygon labels to boxes, keeps augmented copies of one
photo in the same split, trains, and saves `defects.onnx` + `defects_metrics.json` to Drive.
Copy both into `models/` and redeploy. The report picks up the new accuracy automatically.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

29 tests: Indo-HCM tables and worked Examples 5.5.1–5.5.4, LOS band edges, obstruction
factors, exact vehicle counting and trap speeds on synthetic traffic, perspective mapping,
lane-marking detection on synthetic and real photos, digital-twin physics (bottleneck throughput, Greenshields speed),
and the API end to end with the real models.

## Repository layout

```
rca/
  hcm/        indo_hcm.py (manual), extensions.py (project factors), capacity.py (procedure)
  vision/     onnx_yolo.py, road_scan.py (photo → metres), lanes.py (lane markings)
  traffic/    tracker.py, counter.py (video survey)
  twin/       ctm.py (digital twin)
  report.py   PDF
  api.py      FastAPI backend
  web/static  dashboard (HTML/CSS/JS, no build step)
models/       defects.onnx, vehicles.onnx, defects_metrics.json
training/     Colab notebook, merge_datasets.py, class_map.yaml
tests/
docs/METHODOLOGY.md
```

## Known limitations

- COCO vehicle model has no **auto-rickshaw** class; retrain on Indian data (IDD) to separate autos.
- Lane-marking index is uncalibrated and does not change capacity.
- F_W and F_R are project extensions: validate with field queue-discharge measurements.
