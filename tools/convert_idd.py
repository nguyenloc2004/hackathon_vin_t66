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

import cv2
import numpy as np
from scipy import ndimage

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
    if label.endswith("group") and label != "polegroup":  # persongroup, cargroup, ridergroup...
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


PILLION_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
ENCLOSED_VEHICLES = {"car", "bus", "truck", "other"}


def _fix_idd_persons(
    scene: C.Scene,
    raw_boxes: dict[int, tuple[float, float, float, float]],
    raw_labels: dict[int, str],
) -> None:
    """Sửa các khác biệt quy ước của IDD gốc so với guideline cuộc thi:
    0) Rider trên xe 2 bánh đi sát xe tuk-tuk (other) phải ưu tiên ghép vào xe 2 bánh trước, không để other hút mất.
    1) Người ngồi sau xe 2 bánh (pillion) đôi khi bị IDD gán là 'person' thay vì 'rider' -> gộp vào xe đang có rider.
    2) Người ngồi bên trong car/bus/truck/other (hoặc ngồi thò chân ở cửa hông autorickshaw) -> trả pixel về xe.
    """
    n = len(scene.roles)
    if n <= 1:
        return
    areas = np.bincount(scene.id_map.ravel(), minlength=n)
    slices = ndimage.find_objects(scene.id_map, max_label=n - 1)
    _, pairs = C._rider_pairs(scene, slices, areas)
    # Ưu tiên tuyệt đối xe 2 bánh (motorcycle, bicycle) trước 'other' cho nhãn 'rider' của IDD
    prioritized = sorted(pairs, key=lambda t: (scene.roles[t[2]] in ("motorcycle", "bicycle"), t[0]), reverse=True)
    v_riders: dict[int, list[int]] = {}
    assigned_r: dict[int, int] = {}
    for _, r, v in prioritized:
        if r not in assigned_r:
            assigned_r[r] = v
            v_riders.setdefault(v, []).append(r)

    filled_cache: dict[int, tuple[tuple[int, int, int, int], np.ndarray]] = {}

    for p in range(1, n):
        if scene.roles[p] != "person" or areas[p] == 0 or p not in raw_boxes:
            continue
        px0, py0, px1, py1 = raw_boxes[p]
        pw, ph = px1 - px0, py1 - py0
        pcx = (px0 + px1) / 2.0

        # 1) Pillion ngồi trên xe 2 bánh đã có rider
        merged = False
        for v, rlist in v_riders.items():
            if scene.roles[v] not in ("motorcycle", "bicycle") or areas[v] == 0 or v not in raw_boxes:
                continue
            vx0, vy0, vx1, vy1 = raw_boxes[v]
            vw, vh = vx1 - vx0, vy1 - vy0
            if vh <= 0 or vw <= 0:
                continue
            if not (vx0 + 0.02 * vw <= pcx <= vx1 - 0.02 * vw):
                continue
            iw_v = max(0.0, min(px1, vx1) - max(px0, vx0))
            if iw_v / max(1e-6, pw) < 0.70:
                continue
            ok_r = False
            for r in rlist:
                if r not in raw_boxes:
                    continue
                rx0, ry0, rx1, ry1 = raw_boxes[r]
                rw, rh = rx1 - rx0, ry1 - ry0
                if rh <= 0:
                    continue
                iw_r = max(0.0, min(px1, rx1) - max(px0, rx0))
                ih_r = max(0.0, min(py1, ry1) - max(py0, ry0))
                if iw_r / max(1e-6, min(pw, rw)) < 0.25 or ih_r / max(1e-6, min(ph, rh)) < 0.65:
                    continue
                if not (0.65 * rh <= ph <= 1.35 * rh and abs(py0 - ry0) <= 0.22 * rh):
                    continue
                if not (vy0 + 0.25 * vh <= py1 <= vy1 + 0.05 * vh):
                    continue
                ok_r = True
                break
            if not ok_r:
                continue
            group_ids = [v] + rlist
            gx0 = max(0, int(min(px0, *(raw_boxes[g][0] for g in group_ids))) - 4)
            gy0 = max(0, int(min(py0, *(raw_boxes[g][1] for g in group_ids))) - 4)
            gx1 = min(scene.w, int(max(px1, *(raw_boxes[g][2] for g in group_ids))) + 5)
            gy1 = min(scene.h, int(max(py1, *(raw_boxes[g][3] for g in group_ids))) + 5)
            sub = scene.id_map[gy0:gy1, gx0:gx1]
            contact = int(np.count_nonzero((cv2.dilate((sub == p).astype(np.uint8), PILLION_KERNEL) > 0) & np.isin(sub, group_ids)))
            if contact >= max(10, 0.03 * areas[p]):
                sub[sub == p] = v
                scene.roles[p] = None
                areas[v] += areas[p]
                areas[p] = 0
                merged = True
                break
        if merged:
            continue

        # 2a) Người nằm trọn (>= 90%) bên trong viền ngoài của car / bus / truck / other (autorickshaw)
        sl_p = slices[p - 1]
        if sl_p is None:
            continue
        vpx0, vpy0, vpx1, vpy1 = C._bbox(sl_p)
        for v in range(1, n):
            if scene.roles[v] not in ENCLOSED_VEHICLES or areas[v] == 0:
                continue
            sl_v = slices[v - 1]
            if sl_v is None:
                continue
            if v not in filled_cache:
                vb = C._bbox(sl_v)
                vmask = (scene.id_map[sl_v] == v).astype(np.uint8)
                cnts, _ = cv2.findContours(vmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filled = np.zeros_like(vmask)
                cv2.drawContours(filled, cnts, -1, 1, -1)
                filled_cache[v] = (vb, filled)
            (vvx0, vvy0, vvx1, vvy1), filled = filled_cache[v]
            ix0, iy0 = max(vpx0, vvx0), max(vpy0, vvy0)
            ix1, iy1 = min(vpx1, vvx1), min(vpy1, vvy1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            pm = scene.id_map[iy0:iy1, ix0:ix1] == p
            fm = filled[iy0 - vvy0:iy1 - vvy0, ix0 - vvx0:ix1 - vvx0] > 0
            if int(np.count_nonzero(pm & fm)) / max(1, areas[p]) >= 0.90:
                scene.id_map[sl_p][scene.id_map[sl_p] == p] = v
                scene.roles[p] = None
                areas[v] += areas[p]
                areas[p] = 0
                filled_cache.pop(v, None)
                merged = True
                break
        if merged:
            continue

        # 2b) Hành khách ngồi trong khoang hở của autorickshaw (thò gối/vai ra cửa hông, chân lơ lửng trên mặt đất)
        for v in range(1, n):
            if raw_labels.get(v) != "autorickshaw" or areas[v] == 0 or v not in raw_boxes:
                continue
            vx0, vy0, vx1, vy1 = raw_boxes[v]
            vw, vh = vx1 - vx0, vy1 - vy0
            if vw <= 0 or vh <= 0:
                continue
            if not (vx0 - 0.06 * vw <= px0 and px1 <= vx1 + 0.06 * vw):
                continue
            if not (py0 >= vy0 + 0.10 * vh and py1 <= vy1 - 0.12 * vh and ph <= 0.72 * vh):
                continue
            gx0, gy0 = max(0, int(min(px0, vx0)) - 4), max(0, int(min(py0, vy0)) - 4)
            gx1, gy1 = min(scene.w, int(max(px1, vx1)) + 5), min(scene.h, int(max(py1, vy1)) + 5)
            sub = scene.id_map[gy0:gy1, gx0:gx1]
            contact = int(np.count_nonzero((cv2.dilate((sub == p).astype(np.uint8), PILLION_KERNEL) > 0) & (sub == v)))
            if contact >= max(8, 0.04 * areas[p]):
                scene.id_map[sl_p][scene.id_map[sl_p] == p] = v
                scene.roles[p] = None
                areas[v] += areas[p]
                areas[p] = 0
                filled_cache.pop(v, None)
                break

    # Thực hiện gộp luôn các rider đã ưu tiên ghép với xe 2 bánh để finalize không gán nhầm sang 'other'
    for r, v in assigned_r.items():
        if scene.roles[v] in ("motorcycle", "bicycle"):
            sl_r = slices[r - 1]
            if sl_r is not None:
                scene.id_map[sl_r][scene.id_map[sl_r] == r] = v
            scene.roles[r] = None
    for v in v_riders:
        if scene.roles[v] == "motorcycle":
            scene.roles[v] = "motorcyclist"
        elif scene.roles[v] == "bicycle":
            scene.roles[v] = "cyclist"


def convert_one(task, out_dir, long_side, prefix):
    json_path, img_root, gt_root = task
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
    raw_boxes: dict[int, tuple[float, float, float, float]] = {}
    raw_labels: dict[int, str] = {}
    for obj in ann.get("objects", []):
        if obj.get("deleted"):
            continue
        raw_lbl = obj.get("label", "").strip().lower()
        role = role_of(raw_lbl)
        poly = obj.get("polygon")
        if role is None or not poly or len(poly) < 3:
            continue
        pts = np.asarray(poly, dtype=np.float64) * [sx, sy]
        iid = scene.paint_polygons(role, [pts])
        if iid is not None:
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            raw_boxes[iid] = (float(x0), float(y0), float(x1), float(y1))
            raw_labels[iid] = raw_lbl
    _fix_idd_persons(scene, raw_boxes, raw_labels)
    C.export_scene(stem, img, scene, out_dir, stats)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", nargs="+", default=["train"], help="một hoặc nhiều split, ví dụ: --split train val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="idd")
    ap.add_argument("--long-side", type=int, default=C.DEFAULT_LONG_SIDE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.root)
    splits = args.split if isinstance(args.split, list) else [args.split]
    tasks = []
    for sp in splits:
        gt_root, img_root = root / "gtFine" / sp, root / "leftImg8bit" / sp
        for j in sorted(gt_root.rglob("*_gtFine_polygons.json")):
            tasks.append((j, img_root, gt_root))
    if args.limit:
        tasks = tasks[:args.limit]
    labels = {}
    for j, _, _ in tasks[:200]:
        for o in json.loads(j.read_text(encoding="utf-8")).get("objects", []):
            labels[o.get("label", "")] = role_of(o.get("label", ""))
    print(f"Splits: {splits} ({len(tasks)} file JSON). Nhãn gặp trong 200 file đầu -> role:")
    for k in sorted(labels):
        print(f"  {k:<28} -> {labels[k]}")

    fn = partial(convert_one, out_dir=Path(args.out), long_side=args.long_side, prefix=args.prefix)
    total = C.Stats(f"{args.prefix}/{'+'.join(splits)}")
    with Pool(args.workers) as pool:
        for i, st in enumerate(pool.imap_unordered(fn, tasks, chunksize=8), 1):
            total.merge(st)
            if i % 500 == 0:
                print(f"  {i}/{len(tasks)}")
    total.write(args.out)
    total.print()


if __name__ == "__main__":
    main()
