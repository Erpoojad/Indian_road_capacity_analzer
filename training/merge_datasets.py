"""
merge_datasets.py — merge the Roboflow YOLOv8 exports into ONE clean dataset.

* Classes are matched by NAME (each dataset gives class numbers its own
  meaning), using training/class_map.yaml.
* Segmentation polygons are converted to boxes (min/max of the points).
* Roboflow stores 2-3 augmented copies of each photo ("<name>_jpg.rf.<hash>").
  Copies are grouped by original photo and a whole group goes to either
  train or valid — never both — so validation accuracy is honest.
* Duplicate photos across datasets: the first dataset listed wins.

Usage:  python merge_datasets.py --config class_map.yaml --out /content/merged
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def original_stem(name: str) -> str:
    return re.split(r"_(jpg|jpeg|png)\.rf\.|\.rf\.", Path(name).stem, maxsplit=1)[0].lower()


def to_box(vals):
    if len(vals) == 4:
        return vals
    xs, ys = vals[0::2], vals[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="class_map.yaml")
    ap.add_argument("--out", default="merged")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    master = cfg["master"]
    root = Path(cfg["root"])
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    groups = defaultdict(list)          # original photo -> [(img, label lines)]
    seen_hash = set()
    stats = Counter()
    for ds in cfg["datasets"]:
        d = root / ds["folder"]
        meta = yaml.safe_load(open(d / "data.yaml"))
        names = meta["names"] if isinstance(meta["names"], list) else [meta["names"][k] for k in sorted(meta["names"])]
        idmap = {}
        for i, n in enumerate(names):
            target = ds["map"].get(str(n))
            if target is None:
                raise SystemExit(f"[{ds['folder']}] class '{n}' is not in its map: add it to class_map.yaml")
            idmap[i] = None if target == "drop" else master.index(target)
        for img in sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT and p.parent.name == "images"):
            h = hashlib.md5(img.read_bytes()).hexdigest()
            if h in seen_hash:
                stats["duplicate_images_skipped"] += 1
                continue
            seen_hash.add(h)
            lab = img.parent.parent / "labels" / (img.stem + ".txt")
            lines = []
            if lab.exists():
                for ln in lab.read_text().split("\n"):
                    p = ln.split()
                    if len(p) < 5:
                        continue
                    c = idmap.get(int(p[0]))
                    if c is None:
                        stats["boxes_dropped"] += 1
                        continue
                    x, y, w, hh = to_box([float(v) for v in p[1:]])
                    if w <= 0 or hh <= 0:
                        continue
                    lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {hh:.6f}")
                    stats[f"boxes_{master[c]}"] += 1
            groups[(ds["folder"], original_stem(img.name))].append((img, lines))
    keys = sorted(groups)
    random.Random(cfg.get("seed", 0)).shuffle(keys)
    n_val = int(len(keys) * cfg.get("valid_fraction", 0.15))
    split = {k: ("valid" if i < n_val else "train") for i, k in enumerate(keys)}
    for k, items in groups.items():
        s = split[k]
        for img, lines in items:
            name = hashlib.md5(str(img).encode()).hexdigest()[:10] + "_" + img.name
            (out / s / "images").mkdir(parents=True, exist_ok=True)
            (out / s / "labels").mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out / s / "images" / name)
            (out / s / "labels" / (Path(name).stem + ".txt")).write_text("\n".join(lines))
            stats[f"images_{s}"] += 1
    yaml.safe_dump({"path": str(out.resolve()), "train": "train/images", "val": "valid/images",
                    "names": {i: n for i, n in enumerate(master)}}, open(out / "data.yaml", "w"), sort_keys=False)
    report = "\n".join(f"{k}: {v}" for k, v in sorted(stats.items()))
    (out / "merge_report.txt").write_text(report)
    print(report)
    print(f"\nMerged dataset: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
