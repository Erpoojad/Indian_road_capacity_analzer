"""
Modules carried over from the earlier version that are NOT wired into the web
app (same status as before). Each must be given its inputs and validated
before any of its output is presented as a project result.

  speed_challan.py       ultralytics-tracker version of the speed check. The
                         same line-crossing rule now runs live in
                         rca/traffic/counter.py (speed_check): set a speed
                         limit in the dashboard's video tab.
  plate_recognition.py   ANPR evidence crops; needs easyocr (PyTorch) - too
                         large for the free server. Run it on a laptop.
  footpath_pedestrian.py needs a pedestrian model, shapely, and operator-drawn
                         footpath / carriageway polygons per camera.
  report_generator.py    CSV department reports; superseded by the PDF
                         department report (/api/export/department-report.pdf).
  digital_twin_bridge.py launches local MATLAB (MATLAB_EXE) on matlab_twin/;
                         the web twin is rca/twin/ctm.py, and the MATLAB /
                         RoadRunner files come in /api/export/bundle.zip.
"""
