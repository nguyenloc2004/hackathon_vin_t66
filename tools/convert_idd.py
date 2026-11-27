"""IDD Segmentation (India Driving Dataset) hoặc Cityscapes -> YOLO-seg 11 lớp của cuộc thi.

Cả hai dùng chung định dạng polygon JSON kiểu Cityscapes:
    <root>/leftImg8bit/<split>/<thư mục>/<tên>_leftImg8bit.(png|jpg)
    <root>/gtFine/<split>/<thư mục>/<tên>_gtFine_polygons.json

Polygon trong JSON được vẽ theo thứ tự xuất hiện (vật sau che vật trước). Script vẽ lại MỌI polygon theo đúng
thứ tự đó (vật ngoài 11 lớp như cột, cây được vẽ làm "occluder"), nên mask chỉ còn phần nhìn thấy.

Ví dụ:
    python tools/convert_idd.py --root D:/data/IDD_Segmentation --split train --out D:/data/converted/idd
    python tools/convert_idd.py --root D:/data/cityscapes --split train --out D:/data/converted/cityscapes --prefix cs
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import common as C

ROLE_MAP = {
    "person": "person",
    "rider": C.RIDER,
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "autorickshaw": "other",
    "caravan": "other",
    "trailer": "other",
    "vehicle fallback": "other",
    "traffic sign": "sign",
    "traffic light": "light",
    "train": C.OCCLUDER,
    "animal": C.OCCLUDER,
    "license plate": None,      # nằm trên xe: không vẽ để biển số vẫn thuộc mask xe
}
SKIP_LABELS = {"unlabeled"}


def role_of(label):
    label = label.strip().lower()
    if label in SKIP_LABELS:
        return None
    if label.endswith("group"):             # persongroup, cargroup, ridergroup...
        return C.CROWD
    return ROLE_MAP.get(label, C.OCCLUDER)


def find_image(img_root, json_path, gt_root):
    rel = json_path.parent.relative_to(gt_root)
    base = json_path.name.replace("_gtFine_polygons.json", "")
    for ext in (".png", ".jpg", ".jpeg"):
        p = img_root / rel / f"{base}_leftImg8bit{ext}"
        if p.is_file():
            return p
    return None


def convert_one(json_path, img_root, gt_root, out_dir, long_side, prefix):
    stats = C.Stats(prefix)
    img_path = find_image(img_root, json_path, gt_root)
    stem = f"{prefix}_{json_path.parent.name}_{json_path.name.replace('_gtFine_polygons.json', '')}"
    if img_path is None:
        stats.rejected["missing_image"] += 1
        return stats
    ann = json.loads(json_path.read_text(encoding="utf-8"))
    img = C.read_image(img_path)
    scale = C.resize_scale(*img.shape[:2], long_side)
    img = C.resize_image(img, scale)
    scene = C.Scene(*img.shape[:2])
    sx = img.shape[1] / ann.get("imgWidth", img.shape[1] / scale)
    sy = img.shape[0] / ann.get("imgHeight", img.shape[0] / scale)
    for obj in ann.get("objects", []):
        if obj.get("deleted"):
            continue
        role = role_of(obj.get("label", ""))
        poly = obj.get("polygon")
        if role is None or not poly or len(poly) < 3:
            continue
        pts = np.asarray(poly, dtype=np.float64) * [sx, sy]
        scene.paint_polygons(role, [pts])
    C.export_scene(stem, img, scene, out_dir, stats)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="idd")
    ap.add_argument("--long-side", type=int, default=C.DEFAULT_LONG_SIDE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.root)
    gt_root, img_root = root / "gtFine" / args.split, root / "leftImg8bit" / args.split
    jsons = sorted(gt_root.rglob("*_gtFine_polygons.json"))
    if args.limit:
        jsons = jsons[:args.limit]
    labels = {}
    for j in jsons[:200]:
        for o in json.loads(j.read_text(encoding="utf-8")).get("objects", []):
            labels[o.get("label", "")] = role_of(o.get("label", ""))
    print("Nhãn gặp trong 200 file đầu -> role:")
    for k in sorted(labels):
        print(f"  {k:<28} -> {labels[k]}")

    fn = partial(convert_one, img_root=img_root, gt_root=gt_root, out_dir=Path(args.out),
                 long_side=args.long_side, prefix=args.prefix)
    total = C.Stats(f"{args.prefix}/{args.split}")
    with Pool(args.workers) as pool:
        for i, st in enumerate(pool.imap_unordered(fn, jsons, chunksize=8), 1):
            total.merge(st)
            if i % 500 == 0:
                print(f"  {i}/{len(jsons)}")
    total.write(args.out)
    total.print()


if __name__ == "__main__":
    main()
