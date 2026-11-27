"""Mapillary Vistas v2.0 -> YOLO-seg 11 lớp của cuộc thi.

Cấu trúc thư mục Vistas (kiểm tra lại trong README của bộ dữ liệu):
    <root>/config_v2.0.json
    <root>/training/images/*.jpg      <root>/training/v2.0/instances/*.png
    <root>/validation/images/*.jpg    <root>/validation/v2.0/instances/*.png

Ảnh instance PNG 16 bit: label_id = pixel // 256, instance_id = pixel % 256 (theo README Vistas).
Chạy `--check` trước trên vài ảnh để xác nhận: nhãn suy ra phải khớp với v2.0/labels/*.png.

Ví dụ:
    python tools/convert_vistas.py --root D:/data/vistas --split validation --out D:/data/converted/vistas --check 5
    python tools/convert_vistas.py --root D:/data/vistas --split training --out D:/data/converted/vistas --workers 8
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

import common as C

# Ánh xạ theo TIỀN TỐ tên nhãn Vistas -> role. Dòng trên được ưu tiên hơn dòng dưới.
ROLE_RULES = [
    ("human--person--person-group", C.CROWD),
    ("object--vehicle--vehicle-group", C.CROWD),
    ("human--person--individual", "person"),
    ("human--person", "person"),
    ("human--rider--", C.RIDER),
    ("object--vehicle--car", "car"),
    ("object--vehicle--bus", "bus"),
    ("object--vehicle--truck", "truck"),
    ("object--vehicle--motorcycle", "motorcycle"),
    ("object--vehicle--bicycle", "bicycle"),
    ("object--vehicle--trailer", "other"),
    ("object--vehicle--caravan", "other"),
    ("object--vehicle--other-vehicle", "other"),
    ("object--vehicle--wheeled-slow", "other"),
    ("object--vehicle--on-rails", C.OCCLUDER),     # tàu điện: không có ở VN, không gán (ghi quyết định vào conventions)
    ("object--vehicle--boat", None),
    ("object--traffic-sign--information-parking", C.OCCLUDER),
    ("object--traffic-sign--ambiguous", C.OCCLUDER),
    ("object--traffic-sign--", "sign"),
    ("object--traffic-light--", "light"),
]


def role_of(name):
    for prefix, role in ROLE_RULES:
        if name.startswith(prefix):
            return role
    return C.OCCLUDER  # mọi vật có instance khác (cột, thùng rác...) chỉ dùng để che


def load_config(root):
    cfg_path = Path(root) / "config_v2.0.json"
    labels = json.loads(cfg_path.read_text(encoding="utf-8"))["labels"]
    names = [lab["name"] for lab in labels]
    has_inst = [bool(lab.get("instances", False)) for lab in labels]
    roles = [role_of(n) if inst else None for n, inst in zip(names, has_inst)]
    return names, roles


def read_png16(path):
    arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"Không đọc được {path}")
    return arr.astype(np.int64)


def build_scene(inst_png, label_roles, scale):
    raw = read_png16(inst_png)
    label_ids = raw // 256
    roles_of_label = np.array([r is not None for r in label_roles] + [False], dtype=bool)
    keep = roles_of_label[np.clip(label_ids, 0, len(label_roles))]
    values = np.unique(raw[keep])                       # mỗi giá trị pixel = một instance
    id_map = np.zeros(raw.shape, dtype=np.int32)
    roles = [label_roles[int(v) // 256] for v in values]
    if len(values):
        idx = np.searchsorted(values, raw[keep])
        id_map[keep] = idx + 1
    h, w = id_map.shape
    if scale < 1.0:
        id_map = C.resize_id_map(id_map, (round(w * scale), round(h * scale)))
    return C.Scene.from_id_map(id_map, roles)


def convert_one(img_path, inst_dir, out_dir, label_roles, long_side, prefix):
    stats = C.Stats("vistas")
    stem = f"{prefix}_{img_path.stem}"
    inst_png = inst_dir / f"{img_path.stem}.png"
    if not inst_png.is_file():
        stats.rejected["missing_annotation"] += 1
        return stats
    img = C.read_image(img_path)
    scale = C.resize_scale(*img.shape[:2], long_side)
    img = C.resize_image(img, scale)
    scene = build_scene(inst_png, label_roles, scale)
    if scene.id_map.shape != img.shape[:2]:
        scene.id_map = C.resize_id_map(scene.id_map, (img.shape[1], img.shape[0]))
    C.export_scene(stem, img, scene, out_dir, stats)
    return stats


def check(root, split, names, n):
    """So nhãn suy ra từ instance PNG với v2.0/labels/*.png để xác nhận cách giải mã."""
    inst_dir = Path(root) / split / "v2.0" / "instances"
    lab_dir = Path(root) / split / "v2.0" / "labels"
    for p in sorted(inst_dir.glob("*.png"))[:n]:
        raw = read_png16(p)
        lab = cv2.imdecode(np.fromfile(str(lab_dir / p.name), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if lab is None:
            print(f"  {p.name}: không có file labels để so (bỏ qua)")
            continue
        lab = lab if lab.ndim == 2 else None
        if lab is None:
            print(f"  {p.name}: labels PNG là ảnh màu, không so được trực tiếp")
            continue
        agree = float(np.mean((raw // 256) == lab))
        found = sorted({names[i] for i in np.unique(raw // 256) if i < len(names)})
        print(f"  {p.name}: khớp {agree:.1%}; nhãn: {found[:12]}{' ...' if len(found) > 12 else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="training", choices=["training", "validation"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--long-side", type=int, default=C.DEFAULT_LONG_SIDE)
    ap.add_argument("--limit", type=int, default=0, help="chỉ xử lý N ảnh đầu (để thử)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check", type=int, default=0, help="chỉ kiểm tra cách giải mã trên N ảnh rồi thoát")
    args = ap.parse_args()

    names, label_roles = load_config(args.root)
    print("Ánh xạ nhãn Vistas có instance -> role:")
    for n, r in zip(names, label_roles):
        if r is not None and r != C.OCCLUDER:
            print(f"  {n:<55} -> {r}")
    if args.check:
        check(args.root, args.split, names, args.check)
        return

    img_dir = Path(args.root) / args.split / "images"
    inst_dir = Path(args.root) / args.split / "v2.0" / "instances"
    images = sorted(img_dir.glob("*.jpg"))
    if args.limit:
        images = images[:args.limit]
    prefix = "vistas" if args.split == "training" else "vistasv"
    fn = partial(convert_one, inst_dir=inst_dir, out_dir=Path(args.out), label_roles=label_roles,
                 long_side=args.long_side, prefix=prefix)
    total = C.Stats(f"vistas/{args.split}")
    with Pool(args.workers) as pool:
        for i, st in enumerate(pool.imap_unordered(fn, images, chunksize=8), 1):
            total.merge(st)
            if i % 500 == 0:
                print(f"  {i}/{len(images)}")
    total.write(args.out)
    total.print()


if __name__ == "__main__":
    main()
