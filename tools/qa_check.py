"""Kiểm tra tự động một bộ dữ liệu YOLO-seg trước khi gộp / huấn luyện (plan/data_pipeline.md §8).

Lỗi (thoát với mã 1): id lớp ngoài 0-10, polygon < 3 điểm, tọa độ ngoài [0, 1], polygon diện tích ~0,
hai instance chồng nhau nặng (> 20% diện tích vật nhỏ hơn).
Cảnh báo: chồng nhau nhẹ (> 5%), vật dưới ngưỡng kích thước, ảnh > 100 instance, Motorcyclist/Cyclist có dáng
"chỉ có người" (quá cao so với rộng), nhãn không có ảnh, lớp có ít instance.

Ví dụ:
    python tools/qa_check.py --data D:/data/converted/vistas
    python tools/qa_check.py --data D:/data/datasets/merged_v1 --min-instances 400
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import common as C

RASTER_LONG = 1280  # raster tối đa cạnh dài 1280 (thang nhỏ hơn làm sai lệch đường biên chung giữa hai vật)


def check_image(img_path, lab_path, errors, warns, per_class, images_per_class, args):
    w, h = Image.open(img_path).size
    if not lab_path.is_file():
        return 0
    s = min(1.0, RASTER_LONG / max(w, h))
    rw, rh = max(1, round(w * s)), max(1, round(h * s))
    count = np.zeros((rh, rw), dtype=np.uint8)
    masks = []
    lines = lab_path.read_text(encoding="utf-8").splitlines()
    n = 0
    for ln, line in enumerate(lines, 1):
        parts = line.split()
        if not parts:
            continue
        where = f"{lab_path.name}:{ln}"
        try:
            cls = int(parts[0])
            vals = np.asarray([float(v) for v in parts[1:]], dtype=np.float64)
        except ValueError:
            errors["không đọc được số"].append(where)
            continue
        if not 0 <= cls < len(C.CLASSES):
            errors["id lớp ngoài 0-10"].append(where)
            continue
        if len(vals) % 2 or len(vals) < 6:
            errors["polygon < 3 điểm hoặc số tọa độ lẻ"].append(where)
            continue
        pts = vals.reshape(-1, 2)
        if (pts < -1e-3).any() or (pts > 1 + 1e-3).any():
            errors["tọa độ ngoài [0, 1]"].append(where)
        px = pts * [w, h]
        if abs(cv2.contourArea(px.astype(np.float32))) < 1.0:
            errors["polygon diện tích ~0"].append(where)
            continue
        n += 1
        per_class[cls] += 1
        images_per_class[cls].add(img_path.stem)
        bw, bh = np.ptp(px[:, 0]), np.ptp(px[:, 1])
        if min(bw, bh) < C.MIN_SIDE_FRAC * max(w, h):
            warns["vật dưới ngưỡng kích thước (bộ dữ liệu chưa lọc?)"].append(where)
        if cls in (5, 7) and bh > 3.2 * bw:
            warns["Motorcyclist/Cyclist cao bất thường (mask chỉ có người?)"].append(where)
        m = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillPoly(m, [np.round(pts * [rw, rh] * 16).astype(np.int32)], 1, shift=4)
        count += m
        masks.append((where, m))
    if n > C.MAX_INSTANCES:
        warns[f"ảnh > {C.MAX_INSTANCES} instance"].append(lab_path.name)
    if masks and count.max() > 1:
        over = count > 1
        for where, m in masks:
            area = int(m.sum())
            ov = int((over & (m > 0)).sum())
            if area and ov / area > 0.2 and ov > 20:
                errors["chồng nhau nặng (> 20%)"].append(where)
            elif area and ov / area > 0.05 and ov > 20:
                warns["chồng nhau nhẹ (> 5%)"].append(where)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="thư mục có images/ và labels/")
    ap.add_argument("--min-instances", type=int, default=400)
    ap.add_argument("--show", type=int, default=5, help="số ví dụ in ra cho mỗi loại lỗi")
    args = ap.parse_args()

    root = Path(args.data)
    errors, warns = defaultdict(list), defaultdict(list)
    per_class, images_per_class = Counter(), defaultdict(set)
    n_img = n_bg = 0
    img_stems = set()
    for img_path, lab_path in C.iter_dataset(root):
        n_img += 1
        img_stems.add(lab_path.relative_to(root / "labels").with_suffix("").as_posix())
        if check_image(img_path, lab_path, errors, warns, per_class, images_per_class, args) == 0:
            n_bg += 1
    labels_dir = root / "labels"
    orphan = [p.name for p in labels_dir.rglob("*.txt")
              if p.relative_to(labels_dir).with_suffix("").as_posix() not in img_stems] if labels_dir.is_dir() else []
    if orphan:
        warns["nhãn không có ảnh"] = orphan
    for c in range(len(C.CLASSES)):
        if per_class[c] < args.min_instances:
            warns[f"lớp ít instance (< {args.min_instances})"].append(f"{C.CLASSES[c]}: {per_class[c]}")

    print(f"{root}: {n_img} ảnh ({n_bg} ảnh nền/không nhãn), {sum(per_class.values())} instance")
    print(f"{'id':>2}  {'lớp':<14} {'instance':>9} {'số ảnh':>8}")
    for c, name in enumerate(C.CLASSES):
        print(f"{c:>2}  {name:<14} {per_class[c]:>9} {len(images_per_class[c]):>8}")
    for title, bucket in (("LỖI", errors), ("CẢNH BÁO", warns)):
        for k, v in bucket.items():
            print(f"[{title}] {k}: {len(v)}  ví dụ: {v[:args.show]}")
    report = {"images": n_img, "background_images": n_bg,
              "instances_per_class": {C.CLASSES[c]: per_class[c] for c in range(len(C.CLASSES))},
              "images_per_class": {C.CLASSES[c]: len(images_per_class[c]) for c in range(len(C.CLASSES))},
              "errors": {k: len(v) for k, v in errors.items()}, "warnings": {k: len(v) for k, v in warns.items()},
              "error_examples": {k: v[:50] for k, v in errors.items()},
              "warning_examples": {k: v[:50] for k, v in warns.items()}}
    (root / "qa_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Báo cáo: {root / 'qa_report.json'}")
    if errors:
        print("KHÔNG ĐẠT: sửa các lỗi trên trước khi gộp.")
        sys.exit(1)
    print("ĐẠT (không có lỗi; xem lại cảnh báo).")


if __name__ == "__main__":
    main()
