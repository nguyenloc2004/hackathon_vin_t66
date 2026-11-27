"""Vẽ mask nhãn lên ảnh để kiểm tra bằng mắt (bắt buộc xem 30 ảnh mỗi nguồn trước khi gộp).

Ví dụ:
    python tools/visualize.py --data D:/data/converted/vistas --n 30 --out D:/data/preview/vistas
    python tools/visualize.py --data D:/data/converted/idd --classes 5 7 --n 20 --out D:/data/preview/idd_riders
    python tools/visualize.py --data D:/data/converted/vn_pseudo --stems vn_abc_000120 vn_abc_000140 --out preview
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

import common as C

SHORT = ["car", "bus", "truck", "other", "moto", "MOTO+P", "bike", "BIKE+P", "ped", "sign", "light"]


def draw(img, labels, alpha=0.45):
    h, w = img.shape[:2]
    overlay = img.copy()
    for cls, pts in labels:
        poly = np.round(pts * [w, h]).astype(np.int32)
        cv2.fillPoly(overlay, [poly], C.CLASS_COLORS[cls % len(C.CLASS_COLORS)])
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    for cls, pts in labels:
        color = C.CLASS_COLORS[cls % len(C.CLASS_COLORS)]
        poly = np.round(pts * [w, h]).astype(np.int32)
        cv2.polylines(out, [poly], True, color, 2)
        x, y = poly[:, 0].min(), max(12, poly[:, 1].min() - 3)
        cv2.putText(out, SHORT[cls], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, SHORT[cls], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="thư mục có images/ và labels/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--stems", nargs="*", help="chỉ vẽ các ảnh này")
    ap.add_argument("--classes", type=int, nargs="*", help="chỉ lấy ảnh có ít nhất một lớp trong danh sách")
    ap.add_argument("--side-by-side", action="store_true", help="ảnh gốc bên trái, ảnh có mask bên phải")
    ap.add_argument("--seed", type=int, default=71)
    args = ap.parse_args()

    pairs = list(C.iter_dataset(args.data))
    if args.stems:
        want = set(args.stems)
        pairs = [p for p in pairs if p[0].stem in want]
    if args.classes:
        want_c = set(args.classes)
        pairs = [p for p in pairs if want_c & set(C.class_counts(p[1]))]
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[:args.n] if not args.stems else pairs
    out = Path(args.out)
    for img_path, lab_path in pairs:
        img = C.read_image(img_path)
        vis = draw(img, C.read_label(lab_path))
        if args.side_by_side:
            vis = np.concatenate([img, vis], axis=1)
        C.save_image(out / f"{img_path.stem}.jpg", vis)
    legend = " | ".join(f"{i}={s}" for i, s in enumerate(SHORT))
    print(f"Đã vẽ {len(pairs)} ảnh -> {out}\nChú thích: {legend}  (MOTO+P = Motorcyclist, BIKE+P = Cyclist)")


if __name__ == "__main__":
    main()
