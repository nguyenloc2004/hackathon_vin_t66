"""Gán nhãn nháp bằng mô hình (chạy trên GPU: Kaggle/Colab). Cần: pip install ultralytics==8.4.214

Hai loại mô hình, tự nhận theo tên lớp của mô hình:
  * "coco": mô hình pretrained COCO (ví dụ yolo26x-seg.pt). person/bicycle/car/motorcycle/bus/truck/traffic light/
    stop sign được ánh xạ về 11 lớp; người ngồi trên xe 2 bánh được đoán bằng heuristic rồi ghép thành Motorcyclist/Cyclist.
    Không có Other vehicle và hầu hết biển báo -> các ảnh này BẮT BUỘC người duyệt bổ sung.
  * "comp": teacher đã huấn luyện trên đúng 11 lớp của cuộc thi.

Chính sách tin cậy (plan/data_pipeline.md §6):
  * giữ dự đoán có conf >= --conf-keep;
  * ảnh có dự đoán trong [--conf-low, --conf-keep) được ghi vào review_queue.csv (cần người duyệt trước khi dùng);
  * ảnh không có dự đoán nghi ngờ được ghi vào confident.txt (dùng được ngay).

Chế độ --fill-classes: thêm dự đoán của một số lớp vào nhãn ĐÃ CÓ (--labels-in), ví dụ thêm biển báo (9) cho COCO.
Nhãn có sẵn được ưu tiên (vẽ đè lên dự đoán); dự đoán trùng > 50% với nhãn có sẵn bị bỏ.

Ví dụ:
    python tools/pseudo_label.py --model yolo26x-seg.pt --images /kaggle/input/vn-frames --out /kaggle/working/vn_pseudo
    python tools/pseudo_label.py --model teacher.pt --images D:/data/converted/coco/images \
        --labels-in D:/data/converted/coco/labels --fill-classes 9 10 --out D:/data/converted/coco_filled
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import common as C

COCO_ROLE = {"person": "person", "bicycle": "bicycle", "car": "car", "motorcycle": "motorcycle", "bus": "bus",
             "truck": "truck", "traffic light": "light", "stop sign": "sign"}


def detect_mode(names):
    vals = [names[i] for i in sorted(names)]
    return "comp" if vals == C.CLASSES else "coco"


def predictions(model, img, args):
    r = model.predict(img, imgsz=args.imgsz, conf=args.conf_low, iou=0.6, retina_masks=True, augment=args.augment,
                      max_det=300, device=args.device, verbose=False)[0]
    if r.masks is None or len(r.boxes) == 0:
        return []
    masks = r.masks.data.cpu().numpy() > 0.5
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    xyxy = r.boxes.xyxy.cpu().numpy()
    return [(int(c), float(s), b, m) for c, s, b, m in zip(cls, conf, xyxy, masks)]


def build_scene(preds, names, mode, img_shape, args, existing=None):
    """-> (scene, số dự đoán nghi ngờ, các lớp đích xuất hiện)."""
    h, w = img_shape
    keep, uncertain = [], 0
    for c, s, b, m in preds:
        name = names[c]
        role = C.CLASS_ROLE[c] if mode == "comp" else COCO_ROLE.get(name)
        if role is None:
            continue
        if args.fill_classes and C.ROLE_CLASS.get(role) not in args.fill_classes:
            continue
        if s < args.conf_keep:
            uncertain += 1
            continue
        keep.append([role, s, b, m])
    if mode == "coco":
        persons = [k for k in keep if k[0] == "person"]
        twos = [k for k in keep if k[0] in ("motorcycle", "bicycle")]
        for i in C.riders_from_boxes([p[2] for p in persons], [t[2] for t in twos]):
            persons[i][0] = C.RIDER
    occupied = np.zeros((h, w), dtype=bool)
    if existing:
        for _, pts in existing:
            tmp = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(tmp, [np.round(pts * [w, h]).astype(np.int32)], 1)
            occupied |= tmp > 0
        keep = [k for k in keep if (k[3] & occupied).sum() <= 0.5 * max(1, k[3].sum())]
    scene = C.Scene(h, w)
    for role, s, b, m in sorted(keep, key=lambda k: -k[3].sum()):   # vật lớn vẽ trước, vật nhỏ đè lên
        scene.paint_mask(role, m, meta=s)
    for cls, pts in existing or []:                                     # nhãn có sẵn luôn thắng
        scene.paint_polygons(C.CLASS_ROLE[cls], [pts * [w, h]])
    return scene, uncertain


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="", help="thêm tiền tố vào tên file (để trống = giữ tên)")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf-low", type=float, default=0.15)
    ap.add_argument("--conf-keep", type=float, default=0.5)
    ap.add_argument("--augment", action="store_true", help="TTA: chậm hơn ~3 lần, nhãn tốt hơn")
    ap.add_argument("--device", default=None)
    ap.add_argument("--labels-in", help="thư mục nhãn có sẵn (dùng với --fill-classes)")
    ap.add_argument("--fill-classes", type=int, nargs="*", default=[])
    ap.add_argument("--long-side", type=int, default=C.DEFAULT_LONG_SIDE)
    args = ap.parse_args()
    if args.fill_classes and not args.labels_in:
        raise SystemExit("--fill-classes cần --labels-in")

    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(args.model)
    names = model.names
    mode = detect_mode(names)
    print(f"Mô hình {args.model}: chế độ {mode} ({len(names)} lớp)")
    out = Path(args.out)
    stats = C.Stats(f"pseudo/{Path(args.images).name}")
    queue, confident = [], []
    for k, img_path in enumerate(C.list_images(args.images), 1):
        stem = f"{args.prefix}_{img_path.stem}" if args.prefix else img_path.stem
        img = C.read_image(img_path)
        img = C.resize_image(img, C.resize_scale(*img.shape[:2], args.long_side))
        existing = C.read_label(Path(args.labels_in) / f"{img_path.stem}.txt") if args.labels_in else None
        scene, uncertain = build_scene(predictions(model, img, args), names, mode, img.shape[:2], args, existing)
        res = C.export_scene(stem, img, scene, out, stats, unmatched_rider="person")
        if res.reject:
            continue
        rare = sorted({C.CLASSES[c] for c in res.classes if c in C.RARE_CLASSES})
        night = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()) < 60
        if uncertain or mode == "coco":
            priority = uncertain + 3 * len(rare) + 2 * night
            queue.append((stem, uncertain, len(res.lines), "|".join(rare), int(night), priority))
        else:
            confident.append(stem)
        if k % 200 == 0:
            print(f"  {k} ảnh")
    stats.write(out)
    stats.print()
    queue.sort(key=lambda r: -r[-1])
    with open(out / "review_queue.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "n_uncertain", "n_kept", "rare_classes", "night", "priority"])
        w.writerows(queue)
    (out / "confident.txt").write_text("\n".join(confident), encoding="utf-8")
    print(f"Cần duyệt: {len(queue)} ảnh (review_queue.csv, đã xếp theo ưu tiên); dùng được ngay: {len(confident)} ảnh")


if __name__ == "__main__":
    main()
