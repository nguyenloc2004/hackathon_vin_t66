"""Nhập một bộ dữ liệu đã ở định dạng YOLO-seg (xuất từ CVAT, Roboflow, Kaggle...) về chuẩn của đội.

Làm gì:
  * đổi id lớp theo TÊN lớp trong data.yaml của bộ nguồn -> 11 lớp của cuộc thi (bảng --map);
  * vẽ lại toàn bộ polygon để mask KHÔNG chồng nhau, bỏ vật nhỏ dưới ngưỡng, resize cạnh dài 1280;
  * nếu nguồn có lớp "rider" riêng thì ghép rider + xe như các converter khác.

--map nhận file JSON {"tên lớp nguồn": "đích"}; đích là tên lớp cuộc thi (Car, Motorcyclist...), một role
(rider, crowd, occluder) hoặc null (bỏ). Không có --map: chỉ nhận lớp trùng tên với lớp cuộc thi (không phân biệt hoa thường).

Thứ tự vẽ (--order): "area" = vật lớn vẽ trước, vật nhỏ vẽ sau (mặc định, hợp với ảnh giao thông);
"file" = theo thứ tự dòng trong file nhãn (dùng cho nhãn CVAT khi đã chỉnh z-order cẩn thận).

Ví dụ (nhãn đã duyệt xuất từ CVAT, định dạng Ultralytics YOLO Segmentation):
    python tools/import_yolo.py --src D:/data/cvat_exports/review_01 --out D:/data/converted/vn_reviewed --prefix vn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

import common as C

LOWER_CLASS = {n.lower(): n for n in C.CLASSES}
SPECIAL = {C.RIDER, C.CROWD, C.OCCLUDER}


def load_names(src, names_yaml):
    path = Path(names_yaml) if names_yaml else next((p for p in [Path(src) / "data.yaml", Path(src) / "dataset.yaml"] if p.is_file()), None)
    if path is None:
        raise SystemExit("Không tìm thấy data.yaml của bộ nguồn: dùng --names-yaml")
    names = yaml.safe_load(path.read_text(encoding="utf-8"))["names"]
    return {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))


def target_role(target):
    if target is None:
        return None
    t = str(target).strip()
    if t.lower() in LOWER_CLASS:
        return C.CLASS_ROLE[C.NAME2ID[LOWER_CLASS[t.lower()]]]
    if t in SPECIAL or t in C.ROLE_CLASS:
        return t
    raise SystemExit(f"Đích ánh xạ không hợp lệ: {target!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="thư mục gốc chứa images/ và labels/ (có thể có train/ val/ bên trong)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", required=True, help="tiền tố tên file, ví dụ vn, rf_hcm")
    ap.add_argument("--map", help="file JSON ánh xạ tên lớp nguồn -> đích")
    ap.add_argument("--names-yaml")
    ap.add_argument("--order", choices=["area", "file"], default="area")
    ap.add_argument("--unmatched-rider", choices=["reject", "person", "drop"], default="reject")
    ap.add_argument("--long-side", type=int, default=C.DEFAULT_LONG_SIDE)
    args = ap.parse_args()

    names = load_names(args.src, args.names_yaml)
    mapping = json.loads(Path(args.map).read_text(encoding="utf-8")) if args.map else {}
    id_role = {}
    print("Ánh xạ lớp nguồn -> role:")
    for i, n in names.items():
        tgt = mapping[n] if n in mapping else (n if n.lower() in LOWER_CLASS else None)
        id_role[i] = target_role(tgt)
        print(f"  {i:>3} {n:<28} -> {id_role[i]}")

    stats = C.Stats(args.prefix)
    out = Path(args.out)
    for img_path, lab_path in C.iter_dataset(args.src):
        stem = f"{args.prefix}_{img_path.stem}"
        img = C.read_image(img_path)
        img = C.resize_image(img, C.resize_scale(*img.shape[:2], args.long_side))
        h, w = img.shape[:2]
        items = []
        for cls, pts in C.read_label(lab_path):
            role = id_role.get(cls)
            if role is None or len(pts) < 3:
                continue
            px = pts * [w, h]
            items.append((abs(cv2.contourArea(px.astype(np.float32))), role, px))
        if args.order == "area":
            items.sort(key=lambda t: -t[0])
        scene = C.Scene(h, w)
        for _, role, px in items:
            scene.paint_polygons(role, [px])
        C.export_scene(stem, img, scene, out, stats, unmatched_rider=args.unmatched_rider)
    stats.write(out)
    stats.print()


if __name__ == "__main__":
    main()
