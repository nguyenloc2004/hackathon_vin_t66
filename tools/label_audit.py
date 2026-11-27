"""Tầng L6 của QA: dùng mô hình soi lỗi nhãn (sót vật, sai lớp, biên lệch, nhãn thừa).

So dự đoán của mô hình với nhãn có sẵn; chỗ hai bên lệch nhau nhiều là chỗ NHIỀU KHẢ NĂNG nhãn sai. Kết quả là
hàng duyệt đã xếp ưu tiên (audit_queue.csv, cột đầu là stem giống review_queue.csv) để đưa vào CVAT. Đây là danh
sách gợi ý: người duyệt quyết định, không tự sửa nhãn.

    missing_label   mô hình tự tin (>= --conf-miss) có vật, nhưng không có nhãn nào phủ chỗ đó         -> nghi SÓT NHÃN
    wrong_class     dự đoán khớp vị trí một nhãn (IoU >= --match-iou) nhưng khác lớp, conf >= --conf-cls -> nghi SAI LỚP
    loose_boundary  khớp cùng lớp nhưng IoU < --good-iou, conf >= --conf-cls                             -> nghi BIÊN LỆCH
    spurious_label  nhãn đủ lớn mà mô hình không thấy gì ở đó (IoU < 0,1 với mọi dự đoán)                -> nghi NHÃN THỪA

QUAN TRỌNG: mô hình KHÔNG được huấn luyện trên chính dữ liệu đang soi (nó đã học thuộc nhãn sai nên sẽ không thấy).
  - teacher (train trên public) -> soi ảnh Việt Nam đã duyệt, gold val;
  - baseline chỉ train Vistas -> soi IDD, COCO; ... (plan/data_pipeline.md §8, tầng L6)

Hai cách lấy dự đoán:
  --model teacher.pt   chạy ultralytics (Kaggle/Colab GPU); dự đoán được lưu vào <out>/preds để soi lại không cần GPU
  --preds DIR          file .txt dạng YOLO-seg có conf ở cuối dòng (predict save_txt=True save_conf=True)

Ví dụ:
    python tools/label_audit.py --data /kaggle/input/gold-val-v1 --model teacher.pt --augment --out /kaggle/working/audit_gold
    python tools/label_audit.py --data D:/data/gold_val/gold_val_v1 --preds D:/audit_gold/preds --render 40
    python tools/label_audit.py --data D:/data/datasets/merged_v2 --split val --preds ... --out D:/audit/merged_v2_val
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import common as C

RASTER_LONG = 640          # đủ cho IoU ở mức 0,1–0,7; nhanh hơn 4 lần so với 1280
FINDING_COLORS = {"missing_label": (0, 230, 255), "wrong_class": (255, 0, 255), "loose_boundary": (255, 200, 0),
                  "spurious_label": (0, 0, 255)}
TYPES = list(FINDING_COLORS)


# ------------------------------------------------------------------------------------------------ đọc dự đoán
def read_preds(path):
    """File YOLO-seg có conf ở cuối dòng (số giá trị sau id lớp là lẻ). Không có conf -> conf = 1."""
    out = []
    if not Path(path).is_file():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        vals = [float(v) for v in parts[1:]]
        conf = vals.pop() if len(vals) % 2 else 1.0
        out.append((int(float(parts[0])), conf, np.asarray(vals, dtype=np.float64).reshape(-1, 2)))
    return out


def write_preds(path, preds):
    lines = [f"{c} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in p) + f" {s:.4f}" for c, s, p in preds]
    C.write_label(path, lines)


def model_preds(model, img_path, args):
    r = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf_low, iou=0.6, retina_masks=True,
                      augment=args.augment, max_det=300, device=args.device, verbose=False)[0]
    if r.masks is None or len(r.boxes) == 0:
        return []
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    return [(int(c), float(s), np.asarray(p, dtype=np.float64)) for c, s, p in zip(cls, conf, r.masks.xyn)
            if len(p) >= 3]


# ------------------------------------------------------------------------------------------------ so khớp
def rasterize(polys, rw, rh):
    out = np.zeros((len(polys), rh * rw), dtype=bool)
    for i, p in enumerate(polys):
        m = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillPoly(m, [np.round(np.clip(p, 0, 1) * [rw, rh] * 16).astype(np.int32)], 1, shift=4)
        out[i] = m.ravel() > 0
    return out


def min_side_px(p, w, h):
    px = p * [w, h]
    return float(min(np.ptp(px[:, 0]), np.ptp(px[:, 1]))) + 1


def audit_image(gt, preds, w, h, args):
    """gt: list (cls, poly chuẩn hóa); preds: list (cls, conf, poly chuẩn hóa). -> list phát hiện (dict)."""
    preds = [p for p in preds if p[1] >= args.conf_low]
    if not gt and not preds:
        return []
    s = min(1.0, RASTER_LONG / max(w, h))
    rw, rh = max(1, round(w * s)), max(1, round(h * s))
    G = rasterize([g[1] for g in gt], rw, rh)
    P = rasterize([p[2] for p in preds], rw, rh)
    ga, pa = G.sum(1).astype(np.float64), P.sum(1).astype(np.float64)
    inter = G.astype(np.float32) @ P.T.astype(np.float32) if len(gt) and len(preds) else np.zeros((len(gt), len(preds)))
    iou = inter / np.maximum(1, ga[:, None] + pa[None] - inter)
    min_px = C.MIN_SIDE_FRAC * max(w, h)

    def weight(*classes):
        return 1.5 if any(c in C.RARE_CLASSES or c == 5 for c in classes) else 1.0

    found = []
    # ghép tham lam 1-1 theo IoU giảm dần, không xét lớp (để bắt được sai lớp)
    gm, pm = {}, {}
    for g, p in sorted(zip(*np.nonzero(iou >= args.match_iou)), key=lambda t: -iou[t]):
        if g not in gm and p not in pm:
            gm[g], pm[p] = p, g
    for g, p in gm.items():
        gc, (pc, conf, poly) = gt[g][0], preds[p]
        if conf < args.conf_cls:
            continue
        if gc != pc:
            found.append({"type": "wrong_class", "gt_line": int(g) + 1, "gt_class": gc, "pred_class": pc, "conf": conf,
                          "iou": iou[g, p], "score": 1.2 * conf * weight(gc, pc), "poly": poly})
        elif iou[g, p] < args.good_iou and min_side_px(gt[g][1], w, h) >= 2 * min_px:
            found.append({"type": "loose_boundary", "gt_line": int(g) + 1, "gt_class": gc, "pred_class": pc, "conf": conf,
                          "iou": iou[g, p], "score": conf * (1 - iou[g, p]) * weight(gc), "poly": poly})
    gt_union = G.any(0) if len(gt) else np.zeros(rh * rw, dtype=bool)
    for p, (pc, conf, poly) in enumerate(preds):
        if p in pm or conf < args.conf_miss or min_side_px(poly, w, h) < min_px or not pa[p]:
            continue
        covered = np.count_nonzero(P[p] & gt_union) / pa[p]
        if covered < args.max_covered:
            found.append({"type": "missing_label", "gt_line": "", "gt_class": "", "pred_class": pc, "conf": conf,
                          "iou": float(iou[:, p].max()) if len(gt) else 0.0, "score": conf * weight(pc), "poly": poly})
    if args.spurious:
        for g, (gc, gpoly) in enumerate(gt):
            if g in gm or min_side_px(gpoly, w, h) < 3 * min_px:
                continue
            best = float(iou[g].max()) if len(preds) else 0.0
            if best < 0.1:
                found.append({"type": "spurious_label", "gt_line": g + 1, "gt_class": gc, "pred_class": "",
                              "conf": "", "iou": best, "score": 0.4 * weight(gc), "poly": gpoly})
    return found


# ------------------------------------------------------------------------------------------------ vẽ
def render(img, gt, found, out_path):
    import visualize  # noqa: PLC0415

    h, w = img.shape[:2]
    right = img.copy()
    for f in found:
        poly = np.round(f["poly"] * [w, h]).astype(np.int32)
        color = FINDING_COLORS[f["type"]]
        cv2.polylines(right, [poly], True, color, 2)
        if f["type"] == "wrong_class":
            txt = f"{C.CLASSES[f['gt_class']]}->{C.CLASSES[f['pred_class']]}?"
        elif f["type"] == "spurious_label":
            txt = f"thua? {C.CLASSES[f['gt_class']]}"
        else:
            txt = f"{f['type'].split('_')[0]} {C.CLASSES[f['pred_class']]} {f['conf']:.2f}"
        x, y = int(poly[:, 0].min()), max(14, int(poly[:, 1].min()) - 4)
        cv2.putText(right, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(right, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    C.save_image(out_path, np.concatenate([visualize.draw(img, gt), right], axis=1))


# ------------------------------------------------------------------------------------------------ chạy
def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="thư mục có images/ và labels/")
    ap.add_argument("--split", default="", help="với bộ đã gộp: train hoặc val")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="mô hình 11 lớp (.pt), cần ultralytics")
    src.add_argument("--preds", help="thư mục dự đoán .txt (YOLO-seg + conf)")
    ap.add_argument("--out", help="mặc định <data>/qa/audit")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--augment", action="store_true", help="TTA: chậm hơn, ít báo nhầm hơn")
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf-low", type=float, default=0.25, help="bỏ dự đoán dưới mức này")
    ap.add_argument("--conf-miss", type=float, default=0.6, help="ngưỡng conf để báo sót nhãn")
    ap.add_argument("--conf-cls", type=float, default=0.5, help="ngưỡng conf để báo sai lớp / biên lệch")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--good-iou", type=float, default=0.7)
    ap.add_argument("--max-covered", type=float, default=0.2,
                    help="dự đoán bị nhãn có sẵn phủ < tỉ lệ này mới tính là sót nhãn")
    ap.add_argument("--no-spurious", dest="spurious", action="store_false", help="không báo nhãn thừa")
    ap.add_argument("--render", type=int, default=0, help="vẽ N ảnh ưu tiên cao nhất ra <out>/preview")
    args = ap.parse_args(argv)

    root = Path(args.data)
    images_root = root / "images" / args.split if args.split else root / "images"
    labels_root = root / "labels" / args.split if args.split else root / "labels"
    out = Path(args.out) if args.out else root / "qa" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    model = None
    if args.model:
        from ultralytics import YOLO  # noqa: PLC0415

        model = YOLO(args.model)
        names = [model.names[i] for i in sorted(model.names)]
        if names != C.CLASSES:
            raise SystemExit(f"Mô hình phải có đúng 11 lớp của cuộc thi theo thứ tự, đang có: {names}")
    print("Nhắc lại: mô hình KHÔNG được huấn luyện trên chính dữ liệu đang soi.")

    imgs = C.list_images(images_root)
    rows, per_image, cache, no_pred = [], {}, {}, 0
    for k, img_path in enumerate(imgs, 1):
        rel = img_path.relative_to(images_root).with_suffix("")
        gt = C.read_label(C.label_path_for(img_path, images_root, labels_root))
        if model is not None:
            preds = model_preds(model, img_path, args)
            write_preds(out / "preds" / f"{rel.as_posix()}.txt", preds)
        else:
            pf = Path(args.preds) / f"{rel.as_posix()}.txt"
            if not pf.is_file():   # chưa dự đoán ảnh này: bỏ qua, không coi là "mô hình không thấy gì"
                no_pred += 1
                continue
            preds = read_preds(pf)
        with Image.open(img_path) as im:
            w, h = im.size
        found = audit_image(gt, preds, w, h, args)
        if found:
            per_image[rel.as_posix()] = found
            cache[rel.as_posix()] = (img_path, gt)
            for f in found:
                rows.append({"stem": rel.as_posix(), "type": f["type"], "gt_line": f["gt_line"],
                             "gt_class": C.CLASSES[f["gt_class"]] if f["gt_class"] != "" else "",
                             "pred_class": C.CLASSES[f["pred_class"]] if f["pred_class"] != "" else "",
                             "conf": f"{f['conf']:.3f}" if f["conf"] != "" else "", "iou": f"{f['iou']:.3f}",
                             "score": f"{f['score']:.3f}"})
        if k % 200 == 0:
            print(f"  {k}/{len(imgs)} ảnh, {len(per_image)} ảnh có nghi vấn")

    with open(out / "findings.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=["stem", "type", "gt_line", "gt_class", "pred_class", "conf", "iou", "score"])
        wr.writeheader()
        wr.writerows(sorted(rows, key=lambda r: -float(r["score"])))
    queue = []
    for stem, found in per_image.items():
        cnt = Counter(f["type"] for f in found)
        queue.append([stem, *(cnt[t] for t in TYPES), round(sum(f["score"] for f in found), 3)])
    queue.sort(key=lambda r: -r[-1])
    with open(out / "audit_queue.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["stem", *[f"n_{t}" for t in TYPES], "priority"])
        wr.writerows(queue)

    by_type = Counter(r["type"] for r in rows)
    confusions = Counter(f"{r['gt_class']} -> {r['pred_class']}" for r in rows if r["type"] == "wrong_class")
    missing_cls = Counter(r["pred_class"] for r in rows if r["type"] == "missing_label")
    summary = {"data": str(root), "split": args.split, "images": len(imgs) - no_pred, "skipped_no_pred": no_pred,
               "images_flagged": len(per_image),
               "findings": dict(by_type), "wrong_class_pairs": dict(confusions.most_common()),
               "missing_by_class": dict(missing_cls.most_common()),
               "params": {k: v for k, v in vars(args).items() if k not in ("data",)}}
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.render:
        prev = out / "preview"
        for i, row in enumerate(queue[:args.render]):
            img_path, gt = cache[row[0]]
            render(C.read_image(img_path), gt, per_image[row[0]], prev / f"{i:03d}_{Path(row[0]).name}.jpg")
        print(f"Đã vẽ {min(args.render, len(queue))} ảnh -> {prev} (trái: nhãn, phải: nghi vấn; "
              f"vàng = sót, tím = sai lớp, xanh = biên, đỏ = thừa)")

    if no_pred:
        print(f"\nBỏ qua {no_pred} ảnh không có file dự đoán trong {args.preds} "
              f"(ultralytics không ghi file cho ảnh không có dự đoán nào: dùng --model để soi đủ).")
    print(f"\n{len(imgs) - no_pred} ảnh đã soi, {len(per_image)} ảnh có nghi vấn: {dict(by_type)}")
    if confusions:
        print("Cặp nghi sai lớp nhiều nhất:", dict(confusions.most_common(5)))
    if missing_cls:
        print("Nghi sót nhãn theo lớp:", dict(missing_cls.most_common()))
    print(f"Hàng duyệt: {out / 'audit_queue.csv'} (đưa vào CVAT như review_queue.csv); chi tiết: {out / 'findings.csv'}")
    print("Sau khi duyệt: ghi lại tỉ lệ nghi vấn là lỗi thật (precision) để chỉnh ngưỡng và làm số liệu cho slide.")


if __name__ == "__main__":
    main()
