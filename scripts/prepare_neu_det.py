"""Fetch the NEU-DET detection dataset (images + Pascal-VOC XML boxes) and convert it
to the Ultralytics YOLO format for the detection track.

Source: the public GitHub repo siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection,
which hosts IMAGES/ANNOTATIONS (train) and Validation_Images/Validation_Annotations (val).
The repo is large (~420 MB of models/notebooks we don't need), so we pull ONLY those four
data folders via raw.githubusercontent in parallel (~45 MB).

Usage:
    python scripts/prepare_neu_det.py
    python scripts/yolo_train.py --data data/neu_det/data.yaml --epochs 50 --imgsz 256
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import pathlib
import random
import re
import shutil
import time
import xml.etree.ElementTree as ET

import requests
from PIL import Image

OWNER, REPO, BRANCH = "siddhartamukherjee", "NEU-DET-Steel-Surface-Defect-Detection", "master"
# (annotations dir, images dir, split)
SPLITS = [("ANNOTATIONS", "IMAGES", "train"), ("Validation_Annotations", "Validation_Images", "val")]
NEEDED = tuple(p + "/" for pair in SPLITS for p in pair[:2])
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def list_tree():
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/git/trees/{BRANCH}?recursive=1"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    if data.get("truncated"):
        print("WARN: git tree truncated — some files may be missing")
    return [t["path"] for t in data["tree"] if t["type"] == "blob"]


def fetch_one(path, raw_root, sess):
    dst = raw_root / path
    if dst.exists() and dst.stat().st_size > 0:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{path}"
    for attempt in range(4):
        try:
            r = sess.get(url, timeout=60)
            if r.status_code == 200:
                dst.write_bytes(r.content)
                return True
        except Exception:
            pass
        time.sleep(0.5 * (attempt + 1))
    print("  FAILED:", path)
    return False


def download(raw_root, workers):
    paths = [p for p in list_tree() if p.startswith(NEEDED)]
    print(f"fetching {len(paths)} data files from GitHub raw ...")
    sess = requests.Session()
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_one, p, raw_root, sess) for p in paths]
        for i, f in enumerate(cf.as_completed(futs), 1):
            ok += 1 if f.result() else 0
            if i % 500 == 0:
                print(f"  {i}/{len(paths)}")
    print(f"downloaded {ok}/{len(paths)} files")


def parse_voc(xml_path, class_to_idx, img_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is not None and size.find("width") is not None:
        W, H = float(size.find("width").text), float(size.find("height").text)
    else:
        W, H = Image.open(img_path).size
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        idx = class_to_idx[name]
        b = obj.find("bndbox")
        xmin, ymin = float(b.find("xmin").text), float(b.find("ymin").text)
        xmax, ymax = float(b.find("xmax").text), float(b.find("ymax").text)
        cx, cy = (xmin + xmax) / 2 / W, (ymin + ymax) / 2 / H
        w, h = (xmax - xmin) / W, (ymax - ymin) / H
        lines.append(f"{idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def find_image(images_dir, stem):
    for ext in IMG_EXT:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="Prepare NEU-DET in YOLO format")
    ap.add_argument("--out-raw", default="data/raw/NEU-DET")
    ap.add_argument("--out", default="data/neu_det")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_root = pathlib.Path(args.out_raw)
    download(raw_root, args.workers)

    # discover class names across all annotation XMLs (stable, sorted -> fixed indices)
    classes = set()
    for ann_dir, _, _ in SPLITS:
        for xml in (raw_root / ann_dir).glob("*.xml"):
            for obj in ET.parse(xml).getroot().findall("object"):
                classes.add(obj.find("name").text.strip())
    class_names = sorted(classes)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print("classes:", class_names)

    # Pool ALL images from both source folders (the repo's own val split is only ~30
    # images), then make a fresh stratified 80/20 split keyed on the filename class prefix.
    pairs = []
    for ann_dir, img_dir, _ in SPLITS:
        for xml in sorted((raw_root / ann_dir).glob("*.xml")):
            img = find_image(raw_root / img_dir, xml.stem)
            if img is not None:
                pairs.append((xml, img))

    by_cls = collections.defaultdict(list)
    for xml, img in pairs:
        m = re.match(r"^(.*?)_\d+$", xml.stem)
        by_cls[m.group(1) if m else xml.stem].append((xml, img))

    plan = {"train": [], "val": []}
    for cls in sorted(by_cls):
        items = sorted(by_cls[cls], key=lambda t: t[0].name)
        random.Random(args.seed).shuffle(items)
        k = int(round(len(items) * args.val_frac))
        plan["val"] += items[:k]
        plan["train"] += items[k:]

    out = pathlib.Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    counts = {}
    for split, split_pairs in plan.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for xml, img in split_pairs:
            lines = parse_voc(xml, class_to_idx, img)
            shutil.copy2(img, out / "images" / split / img.name)
            (out / "labels" / split / f"{xml.stem}.txt").write_text("\n".join(lines))
        counts[split] = len(split_pairs)

    yaml_path = out / "data.yaml"
    names_block = "\n".join(f"  {i}: {c}" for i, c in enumerate(class_names))
    yaml_path.write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names_block}\n"
    )
    print(f"train images: {counts.get('train')}  val images: {counts.get('val')}")
    print(f"wrote {yaml_path}")
    sample = next((out / "labels" / "train").glob("*.txt"), None)
    if sample:
        print(f"sample label ({sample.name}): {sample.read_text().splitlines()[:2]}")


if __name__ == "__main__":
    main()
