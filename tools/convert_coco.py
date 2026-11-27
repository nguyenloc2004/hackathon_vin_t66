"""COCO 2017 (chỉ ảnh đường phố) -> YOLO-seg 11 lớp của cuộc thi.

COCO KHÔNG phân biệt rider với người đi bộ, và chỉ có lớp "stop sign" (thiếu hầu hết biển báo khác).
Vì vậy:
  * rider được đoán bằng heuristic hộp (common.riders_from_boxes) rồi ghép mask với xe;
  * output này PHẢI được chạy pseudo_label.py --fill-classes 9 (gán nháp thêm biển báo) trước khi gộp vào train,
    nếu không mô hình sẽ học rằng biển báo là nền.

COCO mask chồng nhau: vẽ theo diện tích giảm dần (vật nhỏ thường ở phía trước).

Ví dụ:
    python tools/convert_coco.py --ann D:/data/coco/annotations/instances_train2017.json \
        --images D:/data/coco/train2017 --out D:/data/converted/coco --max-images 1500
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import common as C

ROLE_BY_NAME = {"person": "person", "bicycle": "bicycle", "car": "car", "motorcycle": "motorcycle",
                "bus": "bus", "truck": "truck", "traffic light": "light", "stop sign": "sign",
                "train": C.OCCLUDER}
VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle"}
TWO_WHEELERS = {"motorcycle", "bicycle"}


def seg_to_polys(seg):
    return [np.asarray(s, dtype=np.float64).reshape(-1, 2) for s in seg if len(s) >= 6]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="coco")
    ap.add_argument("--min-objects", type=int, default=4, help="số đối tượng giao thông tối thiểu mỗi ảnh")
    ap.add_argument("--min-vehicles", type=int, default=2)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--long-side", type=int, default=C.DEFAULT_LONG_SIDE)
    args = ap.parse_args()

    coco = json.loads(Path(args.ann).read_text(encoding="utf-8"))
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        if cat_name[a["category_id"]] in ROLE_BY_NAME:
            anns[a["image_id"]].append(a)

    # Lọc ảnh đường phố: đủ đối tượng giao thông và đủ xe
    candidates = []
    for img_id, lst in anns.items():
        names = [cat_name[a["category_id"]] for a in lst if not a["iscrowd"]]
        if len(names) >= args.min_objects and sum(n in VEHICLES for n in names) >= args.min_vehicles:
            candidates.append(img_id)
    candidates.sort()
    if args.max_images:
        rng = np.random.default_rng(71)
        rng.shuffle(candidates)
        candidates = sorted(candidates[:args.max_images])
    print(f"Ảnh đường phố được chọn: {len(candidates)}")

    stats = C.Stats(args.prefix)
    out = Path(args.out)
    for k, img_id in enumerate(candidates, 1):
        info = images[img_id]
        stem = f"{args.prefix}_{Path(info['file_name']).stem}"
        lst = anns[img_id]
        H, W = info["height"], info["width"]
        crowd = sum(a["area"] for a in lst if a["iscrowd"])
        if crowd > 0.01 * H * W:
            stats.add(stem, C.SceneResult(reject="crowd"))
            continue
        lst = [a for a in lst if not a["iscrowd"]]
        boxes = {a["id"]: (a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]) for a in lst}
        persons = [a for a in lst if cat_name[a["category_id"]] == "person"]
        twos = [a for a in lst if cat_name[a["category_id"]] in TWO_WHEELERS]
        rider_idx = C.riders_from_boxes([boxes[a["id"]] for a in persons], [boxes[a["id"]] for a in twos])
        rider_ids = {persons[i]["id"] for i in rider_idx}

        img = C.read_image(Path(args.images) / info["file_name"])
        scale = C.resize_scale(H, W, args.long_side)
        img = C.resize_image(img, scale)
        scene = C.Scene(*img.shape[:2])
        sx, sy = img.shape[1] / W, img.shape[0] / H
        for a in sorted(lst, key=lambda a: -a["area"]):
            role = C.RIDER if a["id"] in rider_ids else ROLE_BY_NAME[cat_name[a["category_id"]]]
            polys = [p * [sx, sy] for p in seg_to_polys(a["segmentation"])]
            scene.paint_polygons(role, polys)
        # rider do heuristic đoán: nếu mask không chạm xe nào thì coi là người đi bộ
        C.export_scene(stem, img, scene, out, stats, unmatched_rider="person")
        if k % 500 == 0:
            print(f"  {k}/{len(candidates)}")
    stats.write(out)
    stats.print()
    print("NHẮC: chạy pseudo_label.py --fill-classes 9 trên output này trước khi gộp vào train.")


if __name__ == "__main__":
    main()
