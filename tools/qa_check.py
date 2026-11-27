"""Cổng chất lượng dữ liệu: MỘT bộ kiểm tra, chạy theo profile (plan/data_pipeline.md §8).

Profile (chọn bằng --profile, hoặc khai báo `profile:` trong file --expect):
  source  một nguồn đã chuyển đổi (Vistas, IDD, COCO, ảnh Việt Nam nháp/đã duyệt...)  -> trước khi đưa vào recipe
  gold    gold val                                                                     -> trước khi khóa phiên bản
  merged  bộ đã gộp images/{train,val} + labels/{train,val}                            -> trước khi đưa lên Kaggle
  auto    (mặc định) merged nếu có images/train hoặc images/val, ngược lại source

Các tầng kiểm tra (mức lỗi/cảnh báo của từng mã phụ thuộc profile, xem bảng CHECKS):
  L0 cấu trúc     ảnh đọc được, đã resize, tên không xung đột, nhãn mồ côi, cấu trúc split
  L1 nhãn         id lớp, số điểm, tọa độ, diện tích, polygon tự cắt, chồng mask, ngưỡng kích thước, rider
  L2 phân phối    độ phủ lớp (theo profile + file --expect), tỉ lệ ảnh nền / ảnh đông, tỉ lệ converter loại ảnh
  L3 chéo tập     ảnh gần trùng trong split, rò rỉ train<->val, gần trùng ảnh tham chiếu (--against)
  L4 toàn vẹn     manifest khớp dữ liệu trên đĩa, zip khớp sha256 và danh sách file (profile merged)
  L5 người xem    mẫu phân tầng qa/sample.csv (ngẫu nhiên + lớp hiếm + ảnh bị cảnh báo); người xem điền cột
                  verdict = ok/bad, chạy lại lệnh để tính tỉ lệ lỗi
  (L6 mô hình soi nhãn: tools/label_audit.py)

Kết quả nằm gọn trong <data>/qa/ (không lẫn với images/, labels/):
  qa/report.json   báo cáo đầy đủ (commit tools, ngưỡng, số liệu từng split, các mã kiểm tra)
  qa/sample.csv    mẫu người xem: CÔNG SỨC CỦA NGƯỜI CHẤM, sao lưu trước khi xóa / chuyển đổi lại thư mục dataset
  qa/preview/      ảnh mẫu đã vẽ nhãn (--render)
  qa/audit/        kết quả label_audit.py (L6)
Mã thoát 1 nếu KHÔNG ĐẠT (còn lỗi chưa được miễn trừ). --archive reports/qa chép report vào repo để tái lập.
Kết luận: FAIL / PASS_AUTO (tự động sạch, mẫu người xem chưa chấm xong) / PASS.

Ví dụ:
    python tools/qa_check.py --data D:/data/converted/vistas --expect configs/qa/vistas.yaml --render
    python tools/qa_check.py --data D:/data/gold_val/gold_val_v1 --expect configs/qa/gold_val.yaml \
        --against D:/data/converted/vn_reviewed/images D:/data/raw/vn_frames
    python tools/qa_check.py --data D:/data/datasets/merged_v1 --render
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

import common as C

OPEN_KERNEL = np.ones((3, 3), np.uint8)
RASTER_LONG = 1280  # raster tối đa cạnh dài 1280 (thang nhỏ hơn làm sai lệch đường biên chung giữa hai vật)
PROFILES = ("source", "gold", "merged")
E, W, O = "error", "warn", "off"

# mã -> (tầng, mô tả, mức theo profile (source, gold, merged))
CHECKS = {
    # L0 cấu trúc
    "img_unreadable":     ("L0", "ảnh hỏng / không đọc được", (E, E, E)),
    "img_not_resized":    ("L0", f"ảnh có cạnh dài > {C.DEFAULT_LONG_SIDE} (chưa resize)", (W, W, W)),
    "stem_conflict":      ("L0", "hai ảnh cùng tên khác đuôi (dùng chung một file nhãn)", (E, E, E)),
    "label_orphan":       ("L0", "nhãn không có ảnh", (W, W, E)),
    "split_layout":       ("L0", "sai cấu trúc images/{train,val} + labels/{train,val}", (O, O, E)),
    # L1 nhãn
    "parse_error":        ("L1", "dòng nhãn không đọc được số", (E, E, E)),
    "class_id":           ("L1", "id lớp ngoài 0-10", (E, E, E)),
    "few_points":         ("L1", "polygon < 3 điểm hoặc số tọa độ lẻ", (E, E, E)),
    "coord_range":        ("L1", "tọa độ ngoài [0, 1]", (E, E, E)),
    "zero_area":          ("L1", "polygon diện tích ~0", (E, E, E)),
    "self_intersect":     ("L1", "polygon tự cắt kiểu hình nơ (diện tích triệt tiêu, vẽ sai thứ tự đỉnh?)", (W, E, W)),
    "small_instance":     ("L1", "vật dưới ngưỡng kích thước (bộ dữ liệu chưa lọc?)", (W, E, W)),
    "rider_tall":         ("L1", "Motorcyclist/Cyclist cao bất thường (mask chỉ có người?)", (W, W, W)),
    "overlap_partial":    ("L1", "hai mask chồng một phần nặng (> overlap_error)", (E, E, E)),
    "overlap_light":      ("L1", "hai mask chồng nhẹ (> overlap_warn)", (W, E, W)),
    "multipart_overlap":  ("L1", "chồng do polygon nhiều mảnh (cạnh nối cắt qua vật khác / vật lớn bị tô kín lỗ): "
                                 "sản phẩm phụ của định dạng YOLO, ít hại", (W, W, W)),
    "contained":          ("L1", "vật nằm trọn trong vật lớn (lỗ polygon, thường chấp nhận được)", (W, W, W)),
    "ped_in_vehicle":     ("L1", "người đi bộ nằm trọn trong xe (người trong xe? guideline: không gán)", (W, E, W)),
    "image_crowded":      ("L1", f"ảnh > {C.MAX_INSTANCES} instance (vượt max_det)", (W, W, W)),
    # L2 phân phối
    "crowded_frac":       ("L2", f"tỉ lệ ảnh > {C.MAX_INSTANCES} instance vượt max_crowded_frac", (E, E, E)),
    "class_missing":      ("L2", "lớp có 0 instance (không học được / không đo được)", (W, E, E)),
    "class_below_min":    ("L2", "lớp dưới mức tối thiểu bắt buộc (min)", (E, E, E)),
    "class_below_target": ("L2", "lớp dưới mục tiêu (target)", (W, W, W)),
    "background_frac":    ("L2", "tỉ lệ ảnh nền (không nhãn) vượt max_background_frac", (W, W, W)),
    "reject_frac":        ("L2", "converter loại quá nhiều ảnh (xem rejected.csv / stats.json)", (W, O, O)),
    "stats_mismatch":     ("L2", "stats.json của converter không khớp dữ liệu trên đĩa (thiếu file?)", (W, O, O)),
    "night_frac":         ("L2", "ít ảnh đêm (ước lượng theo độ sáng) so với min_night_frac", (O, W, O)),
    # L3 chéo tập
    "dup_within":         ("L3", "ảnh gần trùng trong cùng split", (W, E, W)),
    "leak_train_val":     ("L3", "RÒ RỈ: ảnh train trùng / gần trùng ảnh val", (O, O, E)),
    "leak_ref":           ("L3", "gần trùng ảnh tham chiếu (--against)", (W, E, E)),
    "leak_unchecked":     ("L3", "chưa kiểm tra rò rỉ với ảnh train (thiếu --against)", (O, W, O)),
    # L4 toàn vẹn
    "manifest_missing":   ("L4", "không tìm thấy <tên>.manifest.json cạnh thư mục dataset", (O, O, W)),
    "manifest_mismatch":  ("L4", "manifest không khớp dữ liệu trên đĩa", (O, O, E)),
    "zip_mismatch":       ("L4", "zip không khớp (sha256 / danh sách file / cấu trúc)", (O, O, E)),
    # L5 người xem
    "human_audit":        ("L5", "tỉ lệ ảnh lỗi trong mẫu người xem vượt max_sample_error", (E, E, E)),
    "human_audit_pending": ("L5", "mẫu người xem (qa/sample.csv) chưa chấm xong", (W, W, W)),
}

DEFAULTS = {
    "overlap_error": 0.20,        # chồng > 20% diện tích vật nhỏ hơn = lỗi
    "overlap_warn": 0.05,
    "contained_frac": 0.90,       # >= 90% vật nhỏ nằm trong vật lớn = lỗ polygon, không tính là chồng
    "rider_aspect": 3.2,          # cao/rộng của Motorcyclist/Cyclist
    "twist_ratio": 0.7,           # diện tích shoelace < 70% số pixel tô = polygon tự cắt kiểu hình nơ
    "twist_min_px": 400,          # bỏ qua vật nhỏ (sai số biên lớn so với diện tích)
    "max_crowded_frac": 0.05,     # data_pipeline.md §5: ảnh > 100 instance tối đa 5%
    "max_background_frac": 0.10,
    "max_reject_frac": 0.25,
    "dup_threshold": 4,           # Hamming pHash
    "leak_threshold": 4,
    "min_night_frac": 0.10,       # gold val mục tiêu 20% ảnh đêm (labeling_sop.md §3)
    "night_brightness": 60,       # độ sáng trung bình < 60 = ảnh đêm (giống pseudo_label.py)
    "max_sample_error": 0.10,     # > 10% ảnh lỗi trong mẫu người xem = không đạt
    "sample_random": 10,
    "sample_rare": 10,
    "sample_flagged": 10,
}

# mục tiêu số instance mỗi lớp (plan/PLAN.md §2): train >= 1.000 (Cyclist, Other vehicle >= 400), val >= 15
MERGED_TRAIN_TARGET = {c: (400 if c in (3, 7) else 1000) for c in range(len(C.CLASSES))}
VAL_TARGET = 15
QA_DIR = "qa"   # mọi kết quả QA nằm trong <data>/qa/
LEGACY = {"qa_report.json": "report.json", "qa_sample.csv": "sample.csv", "qa_preview": "preview"}
SAMPLE_COLUMNS = ["key", "split", "stem", "reason", "verdict", "error_type", "note"]
VERDICT_OK = {"ok", "dat", "đạt", "pass", "1", "y"}
VERDICT_BAD = {"bad", "loi", "lỗi", "fail", "0", "n"}
REPEAT_RE = re.compile(r"_r\d+$")


# ------------------------------------------------------------------------------------------------ tiện ích
class Issues:
    def __init__(self):
        self.items = defaultdict(list)       # mã -> danh sách mục (chuỗi)
        self.by_image = defaultdict(set)     # "split/stem" -> các mã gặp ở ảnh đó

    def add(self, code, item, key=None):
        assert code in CHECKS, code
        self.items[code].append(item)
        if key is not None:
            self.by_image[key].add(code)


def twisted(poly_raster, filled_px, th):
    """True nếu polygon tự cắt kiểu "vòng xoắn ngược chiều" (hình nơ / số 8).

    Ultralytics và visualize đều tô mask bằng cv2.fillPoly, nên đa số chỗ tự cắt (răng cưa 1 px của approxPolyDP,
    cạnh nối giữa các mảnh) vô hại. Ca có hại: hai vòng ngược chiều làm diện tích hình học (shoelace) triệt tiêu,
    thường do vẽ tay sai thứ tự đỉnh trong CVAT. Đo bằng tỉ lệ diện tích shoelace / số pixel được tô.

    Polygon nhiều mảnh do converter nối (common._bridge_parts) có đỉnh lặp lại ở điểm nối và diện tích shoelace của
    nó cũng bị triệt tiêu dù mask tô ra đúng, nên được bỏ qua.
    """
    if filled_px < th["twist_min_px"]:
        return False
    if len(np.unique(np.round(poly_raster, 3), axis=0)) < len(poly_raster):
        return False
    return abs(cv2.contourArea(poly_raster.astype(np.float32))) < th["twist_ratio"] * filled_px


def git_info():
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "-C", str(here), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", "."], capture_output=True,
                               text=True, timeout=10).stdout.strip()
        return {"commit": commit or None, "tools_dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "tools_dirty": None}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_expect(path):
    """File kỳ vọng của một nguồn (configs/qa/*.yaml). Kiểm tra khóa để bắt lỗi gõ sai."""
    if not path:
        return {}
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    allowed = {"profile", "note", "classes", "val_classes", "absent_ok", "thresholds", "severity", "waivers"}
    bad = set(d) - allowed
    if bad:
        raise SystemExit(f"{path}: khóa không hợp lệ {sorted(bad)} (hợp lệ: {sorted(allowed)})")
    if d.get("profile") not in (None, *PROFILES):
        raise SystemExit(f"{path}: profile phải là một trong {PROFILES}")
    for key in ("classes", "val_classes"):
        for name, rule in (d.get(key) or {}).items():
            if name not in C.NAME2ID:
                raise SystemExit(f"{path}: lớp không tồn tại {name!r} (đúng tên: {C.CLASSES})")
            if not isinstance(rule, dict) or set(rule) - {"min", "target"}:
                raise SystemExit(f"{path}: {key}.{name} phải có dạng {{min: N, target: N}}")
    for name in d.get("absent_ok") or []:
        if name not in C.NAME2ID:
            raise SystemExit(f"{path}: absent_ok có lớp không tồn tại {name!r}")
    for k in d.get("thresholds") or {}:
        if k not in DEFAULTS:
            raise SystemExit(f"{path}: ngưỡng không tồn tại {k!r} (có: {sorted(DEFAULTS)})")
    sev = {code: (O if lvl is False else lvl) for code, lvl in (d.get("severity") or {}).items()}  # YAML: off -> False
    d["severity"] = sev
    for code, lvl in sev.items():
        if code not in CHECKS or lvl not in (E, W, O):
            raise SystemExit(f"{path}: severity.{code}={lvl!r} không hợp lệ (mã trong CHECKS, mức error/warn/off)")
    for wv in d.get("waivers") or []:
        if not isinstance(wv, dict) or wv.get("check") not in CHECKS or not wv.get("item") or not wv.get("reason"):
            raise SystemExit(f"{path}: mỗi waiver cần check (mã hợp lệ), item (mẫu glob) và reason: {wv}")
    return d


def qa_dir(root):
    """<data>/qa/; tự chuyển file của phiên bản cũ (qa_report.json, qa_sample.csv, qa_preview/ ở gốc) vào đây."""
    out = Path(root) / QA_DIR
    out.mkdir(exist_ok=True)
    for old, new in LEGACY.items():
        src, dst = Path(root) / old, out / new
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"Đã chuyển {src.name} -> {QA_DIR}/{new}")
    return out


def find_splits(root):
    images = root / "images"
    subs = {p.name for p in images.iterdir() if p.is_dir()} if images.is_dir() else set()
    named = [s for s in ("train", "val") if s in subs]
    return named or [""]


def class_rules(profile, split, expect, min_instances):
    rules = {c: {"min": 0, "target": 0} for c in range(len(C.CLASSES))}
    if profile == "merged" and split == "train":
        for c, t in MERGED_TRAIN_TARGET.items():
            rules[c]["target"] = t
    if (profile == "merged" and split == "val") or profile == "gold":
        for c in rules:
            rules[c]["target"] = VAL_TARGET
    key = "val_classes" if (profile == "merged" and split == "val") else "classes"
    for name, r in (expect.get(key) or {}).items():
        rules[C.NAME2ID[name]].update({k: int(v) for k, v in r.items()})
    if min_instances is not None and key == "classes":
        for c in rules:
            rules[c]["target"] = min_instances
    return rules


# ------------------------------------------------------------------------------------------------ L1: một ảnh
def check_label(img_path, lab_path, w, h, item_prefix, key, issues, th):
    """Kiểm tra nhãn của một ảnh. -> Counter lớp."""
    classes = Counter()
    if not lab_path.is_file():
        return classes
    s = min(1.0, RASTER_LONG / max(w, h))
    rw, rh = max(1, round(w * s)), max(1, round(h * s))
    count = np.zeros((rh, rw), dtype=np.uint8)
    masks = []
    for ln, line in enumerate(lab_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        where = f"{item_prefix}:{ln}"
        try:
            cls = int(parts[0])
            vals = np.asarray([float(v) for v in parts[1:]], dtype=np.float64)
        except ValueError:
            issues.add("parse_error", where, key)
            continue
        if not 0 <= cls < len(C.CLASSES):
            issues.add("class_id", where, key)
            continue
        if len(vals) % 2 or len(vals) < 6:
            issues.add("few_points", where, key)
            continue
        pts = vals.reshape(-1, 2)
        if (pts < -1e-3).any() or (pts > 1 + 1e-3).any():
            issues.add("coord_range", where, key)
        classes[cls] += 1   # đếm mọi dòng hợp lệ về lớp/điểm, để so khớp được với stats.json / manifest
        px = pts * [w, h]
        if abs(cv2.contourArea(px.astype(np.float32))) < 1.0:
            issues.add("zero_area", where, key)
            continue
        # đỉnh polygon nằm ở tâm pixel biên, nên bề rộng mask = khoảng cách đỉnh + 1 px
        bw, bh = np.ptp(px[:, 0]) + 1, np.ptp(px[:, 1]) + 1
        if min(bw, bh) < C.MIN_SIDE_FRAC * max(w, h) - 0.5:   # 0,5 px: sai số làm tròn tọa độ 6 chữ số
            issues.add("small_instance", where, key)
        if cls in (5, 7) and bh > th["rider_aspect"] * bw:
            issues.add("rider_tall", where, key)
        # raster cục bộ trong bbox (nhanh hơn nhiều so với tô cả ảnh cho từng vật)
        pr = np.clip(pts, 0, 1) * [rw, rh]
        x0, y0 = np.floor(pr.min(0)).astype(int)
        x1, y1 = np.minimum(np.ceil(pr.max(0)).astype(int) + 1, [rw, rh])
        if x1 <= x0 or y1 <= y0:
            continue
        local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillPoly(local, [np.round((pr - [x0, y0]) * 16).astype(np.int32)], 1, shift=4)
        area = int(np.count_nonzero(local))
        if not area:
            continue
        if twisted(pr, area, th):
            issues.add("self_intersect", where, key)
        count[y0:y1, x0:x1] += local
        multi = len(np.unique(np.round(pts, 6), axis=0)) < len(pts)   # đỉnh lặp = điểm nối giữa các mảnh
        masks.append((where, cls, local > 0, (x0, y0, x1, y1), area, multi))
    n = sum(classes.values())
    if n > C.MAX_INSTANCES:
        issues.add("image_crowded", f"{item_prefix} ({n})", key)
    if masks and count.max() > 1:
        check_overlaps(masks, key, issues, th)
    return classes


def check_overlaps(masks, key, issues, th):
    """Xét từng cặp instance chồng nhau.

    - Vật nhỏ nằm gần trọn (>= contained_frac) bên trong vật lớn: thường do vật lớn có LỖ (YOLO-seg không biểu diễn
      được lỗ, ví dụ xe tải phía sau một ô tô) -> cảnh báo. Riêng người đi bộ nằm trọn trong ô tô/xe buýt/xe tải:
      thường là người nhìn qua kính xe, guideline yêu cầu KHÔNG gán -> mã riêng để xóa tay.
    - Chồng một phần > overlap_error diện tích vật nhỏ: lỗi thật (hai mask tranh nhau pixel).
    - Chồng do polygon NHIỀU MẢNH -> multipart_overlap, chỉ cảnh báo. YOLO-seg chỉ có một polygon/vật và không có lỗ,
      nên vật bị che còn vài mẩu phải được nối bằng cạnh đi-về (common._bridge_parts); cạnh nối cắt ngang vật phía
      trước (đường mảnh 1–2 px), hoặc vật lớn có lỗ bị tô kín đè lên mẩu nhìn thấy qua lỗ. Converter dựng mask từ
      bản đồ instance không chồng nhau, nên các ca này là sản phẩm phụ của định dạng, không phải lỗi gán. Khi train,
      Ultralytics vẽ vật nhỏ đè lên vật lớn, nên vật nhỏ vẫn giữ đúng pixel của nó.
    """
    for i in range(len(masks)):
        wi, ci, mi, bi, ai, multi_i = masks[i]
        for j in range(i + 1, len(masks)):
            wj, cj, mj, bj, aj, multi_j = masks[j]
            ix0, iy0 = max(bi[0], bj[0]), max(bi[1], bj[1])
            ix1, iy1 = min(bi[2], bj[2]), min(bi[3], bj[3])
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter_m = (mi[iy0 - bi[1]:iy1 - bi[1], ix0 - bi[0]:ix1 - bi[0]]
                       & mj[iy0 - bj[1]:iy1 - bj[1], ix0 - bj[0]:ix1 - bj[0]])
            inter = int(np.count_nonzero(inter_m))
            if inter <= 20:
                continue
            if ai <= aj:
                (ws, cs, as_, ms), (wb, cb, mb) = (wi, ci, ai, multi_i), (wj, cj, multi_j)
            else:
                (ws, cs, as_, ms), (wb, cb, mb) = (wj, cj, aj, multi_j), (wi, ci, multi_i)
            frac = inter / max(1, as_)
            if frac >= th["contained_frac"]:
                code = "ped_in_vehicle" if cs == 8 and cb in (0, 1, 2) else "contained"
                issues.add(code, f"{ws} trong {wb}", key)
                continue
            if frac <= th["overlap_warn"]:
                continue
            # phần "đặc" của vùng giao: bỏ các đường mảnh < 3 px (cạnh nối giữa các mảnh)
            solid = int(np.count_nonzero(cv2.morphologyEx(inter_m.astype(np.uint8), cv2.MORPH_OPEN, OPEN_KERNEL)))
            sfrac = solid / max(1, as_)
            if ms or (mb and sfrac <= th["overlap_warn"]):
                issues.add("multipart_overlap", f"{ws} với {wb} ({frac:.0%}, phần đặc {sfrac:.0%})", key)
            elif sfrac > th["overlap_error"]:
                issues.add("overlap_partial", f"{ws} với {wb} ({sfrac:.0%})", key)
            else:
                issues.add("overlap_light", f"{ws} với {wb} ({frac:.0%})", key)


# ------------------------------------------------------------------------------------------------ L3: pHash
def near_dups_within(names, hashes, thr):
    """-> list (a, b, d) với b gần trùng a (a đứng trước theo tên)."""
    out = []
    for i in range(len(names) - 1):
        d = C.hamming_to_all(int(hashes[i]), hashes[i + 1:])
        for k in np.nonzero(d <= thr)[0]:
            out.append((names[i], names[i + 1 + int(k)], int(d[k])))
    return out


def near_dups_against(names, hashes, ref_names, ref_hashes, thr):
    out = []
    if not len(ref_hashes):
        return out
    for n, h in zip(names, hashes):
        d = C.hamming_to_all(int(h), ref_hashes)
        j = int(d.argmin())
        if d[j] <= thr:
            out.append((n, ref_names[j], int(d[j])))
    return out


def hash_images(paths, label):
    hashes = np.zeros(len(paths), dtype=np.uint64)
    for i, p in enumerate(paths):
        hashes[i] = np.uint64(C.phash(C.read_image(p)))
        if (i + 1) % 2000 == 0:
            print(f"  băm {label}: {i + 1}/{len(paths)}")
    return hashes


# ------------------------------------------------------------------------------------------------ L4: manifest / zip
def check_manifest(root, split_stats, issues, skip_sha):
    mpath = root.parent / f"{root.name}.manifest.json"
    if not mpath.is_file():
        issues.add("manifest_missing", str(mpath))
        return
    man = json.loads(mpath.read_text(encoding="utf-8"))
    for split, st in split_stats.items():
        want = (man.get("images") or {}).get(split)
        if want != st["images"]:
            issues.add("manifest_mismatch", f"{split}: manifest {want} ảnh, trên đĩa {st['images']}")
        want_inst = (man.get("instances_per_class") or {}).get(split) or {}
        for c, name in enumerate(C.CLASSES):
            if want_inst.get(name, 0) != st["instances"][c]:
                issues.add("manifest_mismatch", f"{split}/{name}: manifest {want_inst.get(name, 0)}, "
                                                f"trên đĩa {st['instances'][c]}")
    if not man.get("zip"):
        return
    zpath = root.parent / man["zip"]
    if not zpath.is_file():
        issues.add("zip_mismatch", f"không thấy {zpath}")
        return
    if not skip_sha and man.get("zip_sha256") and sha256_file(zpath) != man["zip_sha256"]:
        issues.add("zip_mismatch", f"{zpath.name}: sha256 khác manifest (zip bị sửa sau khi dựng?)")
    with zipfile.ZipFile(zpath) as zf:
        in_zip = {n for n in zf.namelist() if not n.endswith("/")}
    on_disk = {p.relative_to(root).as_posix() for sub in ("images", "labels")
               for p in (root / sub).rglob("*") if p.is_file()}
    tops = {n.split("/")[0] for n in in_zip}
    if not tops <= {"images", "labels"}:
        issues.add("zip_mismatch", f"zip có thư mục gốc lạ {sorted(tops - {'images', 'labels'})}")
    for n in sorted(in_zip - on_disk)[:20]:
        issues.add("zip_mismatch", f"chỉ có trong zip: {n}")
    for n in sorted(on_disk - in_zip)[:20]:
        issues.add("zip_mismatch", f"chỉ có trên đĩa: {n}")


# ------------------------------------------------------------------------------------------------ L2: stats.json
def check_converter_stats(root, split_stats, issues, th):
    spath = root / "stats.json"
    if not spath.is_file():
        return None
    st = json.loads(spath.read_text(encoding="utf-8"))
    rejected = st.get("rejected") or st.get("dropped") or {}
    n_rej = int(sum(rejected.values()))
    n_ok = int(st.get("images_ok", st.get("n_images", 0)))
    total = int(st.get("n_input") or (n_ok + n_rej))
    frac = n_rej / total if total else 0.0
    if frac > th["max_reject_frac"]:
        issues.add("reject_frac", f"loại {n_rej}/{total} ảnh ({frac:.0%}): {dict(rejected)}")
    disk = split_stats[next(iter(split_stats))]
    if n_ok and n_ok != disk["images"]:
        issues.add("stats_mismatch", f"stats.json {n_ok} ảnh, trên đĩa {disk['images']}")
    for name, n in (st.get("instances_per_class") or {}).items():
        if name in C.NAME2ID and n != disk["instances"][C.NAME2ID[name]]:
            issues.add("stats_mismatch", f"{name}: stats.json {n}, trên đĩa {disk['instances'][C.NAME2ID[name]]}")
    return {"rejected": n_rej, "total": total, "reject_frac": round(frac, 4), "reasons": rejected}


# ------------------------------------------------------------------------------------------------ L5: mẫu người xem
def build_sample(records, issues, levels, th, seed):
    """Mẫu phân tầng: ngẫu nhiên + có lớp hiếm/rider + bị cảnh báo (xoay vòng qua các mã để đa dạng)."""
    rng = random.Random(seed)
    chosen, rows = set(), []

    def take(pool, n, reason):
        for r in pool:
            if n <= 0:
                break
            if r["key"] not in chosen:
                chosen.add(r["key"])
                rows.append({"key": r["key"], "split": r["split"], "stem": r["stem"], "reason": reason})
                n -= 1

    pool = [r for r in records if not r["repeat"]]
    by_code = defaultdict(list)
    for r in pool:
        for code in sorted(issues.by_image.get(r["key"], ())):
            if levels.get(code) in (E, W):
                by_code[code].append(r)
    flagged = []
    for code in by_code:
        rng.shuffle(by_code[code])
    while any(by_code.values()) and len(flagged) < 4 * th["sample_flagged"]:
        for code in sorted(by_code):
            if by_code[code]:
                flagged.append((code, by_code[code].pop()))
    n = th["sample_flagged"]
    for code, r in flagged:
        if n <= 0:
            break
        if r["key"] not in chosen:
            take([r], 1, f"flag:{code}")
            n -= 1
    rare = [r for r in pool if r["classes"] & (C.RARE_CLASSES | {5})]
    rng.shuffle(rare)
    take(rare, th["sample_rare"], "rare")
    rnd = list(pool)
    rng.shuffle(rnd)
    take(rnd, th["sample_random"], "random")
    return rows


def backup_verdicts(path):
    """--resample: giữ bản đã chấm (sample.<giờ>.csv) thay vì ghi đè công sức của người xem."""
    if not path.is_file():
        return
    with open(path, newline="", encoding="utf-8-sig") as fh:
        if not any(norm_verdict(r.get("verdict")) for r in csv.DictReader(fh)):
            return
    dst = path.with_name(f"sample.{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    shutil.copyfile(path, dst)
    print(f"Đã lưu bản mẫu đã chấm cũ -> {dst}")


def norm_verdict(v):
    v = (v or "").strip().lower()
    return "ok" if v in VERDICT_OK else "bad" if v in VERDICT_BAD else ""


def human_audit(root, records, issues, levels, th, seed, resample, render):
    path = qa_dir(root) / "sample.csv"
    keys = {r["key"] for r in records}
    rows = None
    if path.is_file() and not resample:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh)) or None   # file rỗng -> tạo mẫu mới
    if rows:
        stale = [r["key"] for r in rows if r.get("key") not in keys]
        if stale:
            issues.add("human_audit_pending", f"{QA_DIR}/sample.csv có {len(stale)} ảnh không còn trong dữ liệu "
                                              f"(dữ liệu đã đổi): chạy lại với --resample")
    if rows is None:
        backup_verdicts(path)
        rows = build_sample(records, issues, levels, th, seed)
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:  # utf-8-sig: Excel mở đúng tiếng Việt
            wr = csv.DictWriter(fh, fieldnames=SAMPLE_COLUMNS)
            wr.writeheader()
            for r in rows:
                wr.writerow({**{k: "" for k in SAMPLE_COLUMNS}, **r})
        print(f"Đã tạo mẫu người xem {path} ({len(rows)} ảnh): điền cột verdict = ok/bad, "
              f"error_type = sot|sai_lop|bien|rider|khac, rồi chạy lại lệnh này.")
    if render:
        render_sample(root, rows, records)
    verdicts = [norm_verdict(r.get("verdict")) for r in rows]
    n_ok, n_bad = verdicts.count("ok"), verdicts.count("bad")
    pending = len(rows) - n_ok - n_bad
    result = {"sample_size": len(rows), "ok": n_ok, "bad": n_bad, "pending": pending,
              "error_rate": round(n_bad / (n_ok + n_bad), 4) if n_ok + n_bad else None,
              "error_types": dict(Counter((r.get("error_type") or "?").strip() for r, v in zip(rows, verdicts)
                                          if v == "bad"))}
    if pending:
        issues.add("human_audit_pending", f"{pending}/{len(rows)} ảnh chưa chấm ({path.name})")
    elif rows and result["error_rate"] > th["max_sample_error"]:
        issues.add("human_audit", f"{n_bad}/{len(rows)} ảnh lỗi ({result['error_rate']:.0%}) > "
                                  f"{th['max_sample_error']:.0%}: {result['error_types']}")
    return result


def render_sample(root, rows, records):
    import visualize  # noqa: PLC0415  (cùng thư mục tools/)

    by_key = {r["key"]: r for r in records}
    out = qa_dir(root) / "preview"
    if out.is_dir():
        shutil.rmtree(out)   # vẽ lại cho khớp mẫu hiện tại
    out.mkdir()
    for i, row in enumerate(rows):
        rec = by_key.get(row["key"])
        if rec is None:
            continue
        img = C.read_image(rec["img"])
        vis = np.concatenate([img, visualize.draw(img, C.read_label(rec["lab"]))], axis=1)
        cv2.putText(vis, f"{row['reason']}  {row['key']}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4,
                    cv2.LINE_AA)
        cv2.putText(vis, f"{row['reason']}  {row['key']}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                    cv2.LINE_AA)
        safe = re.sub(r"[^\w.-]+", "_", f"{i:02d}_{row['reason']}_{row['stem']}")
        C.save_image(out / f"{safe}.jpg", vis)
    print(f"Đã vẽ {len(rows)} ảnh mẫu (trái: ảnh gốc, phải: nhãn) -> {out}")


# ------------------------------------------------------------------------------------------------ chạy
def run(args):
    root = Path(args.data)
    if not (root / "images").is_dir():
        raise SystemExit(f"Không thấy {root / 'images'}")
    expect = load_expect(args.expect)
    splits = find_splits(root)
    profile = args.profile
    if profile == "auto":
        profile = expect.get("profile") or ("merged" if splits != [""] else "source")
    th = {**DEFAULTS, **(expect.get("thresholds") or {})}
    pidx = PROFILES.index(profile)
    levels = {code: (expect.get("severity") or {}).get(code, spec[2][pidx]) for code, spec in CHECKS.items()}
    issues = Issues()
    do_hash = not args.no_hash

    if profile == "merged":
        if splits != ["train", "val"]:
            issues.add("split_layout", f"có split {splits}, cần đủ train và val")
        loose = [p.name for p in (root / "images").iterdir() if p.is_file()]
        if loose:
            issues.add("split_layout", f"{len(loose)} ảnh nằm thẳng trong images/ (ngoài train/val)")
        for s in splits:
            if s and not (root / "labels" / s).is_dir():
                issues.add("split_layout", f"thiếu labels/{s}")
    elif splits != [""]:
        print(f"Lưu ý: thư mục có split {splits} nhưng profile là {profile}: kiểm tra từng split riêng.")

    records, split_stats, split_hashes = [], {}, {}
    for split in splits:
        sname = split or "all"
        images_root = root / "images" / split if split else root / "images"
        labels_root = root / "labels" / split if split else root / "labels"
        imgs = C.list_images(images_root)
        stems = {p.relative_to(images_root).with_suffix("").as_posix() for p in imgs}
        base_names = {p.stem for p in imgs}
        st = {"images": 0, "background": 0, "crowded": 0, "night": 0, "instances": Counter(),
              "images_per_class": Counter()}
        seen_rel = Counter(p.relative_to(images_root).with_suffix("").as_posix() for p in imgs)
        for rel, k in seen_rel.items():
            if k > 1:
                issues.add("stem_conflict", f"{sname}/{rel} ({k} ảnh)")
        hash_names, hash_vals = [], []
        for k, img_path in enumerate(imgs, 1):
            lab_path = C.label_path_for(img_path, images_root, labels_root)
            rel = lab_path.relative_to(labels_root).as_posix()
            key = f"{sname}/{img_path.relative_to(images_root).with_suffix('').as_posix()}"
            img = None
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    im.verify()
                if do_hash:
                    img = C.read_image(img_path)
            except Exception as e:  # noqa: BLE001  ảnh hỏng thuộc nhiều loại lỗi khác nhau
                issues.add("img_unreadable", f"{sname}/{img_path.name}: {type(e).__name__}", key)
                continue
            st["images"] += 1
            if max(w, h) > C.DEFAULT_LONG_SIDE:
                issues.add("img_not_resized", f"{sname}/{img_path.name} ({w}x{h})", key)
            m = REPEAT_RE.search(img_path.stem)
            repeat = bool(m) and img_path.stem[:m.start()] in base_names
            classes = check_label(img_path, lab_path, w, h, f"{sname}/{rel}", key, issues, th)
            if not repeat:
                n = sum(classes.values())
                st["background"] += n == 0
                st["crowded"] += n > C.MAX_INSTANCES
            st["instances"].update(classes)
            for c in classes:
                st["images_per_class"][c] += 1
            if img is not None and not repeat:
                hash_names.append(key)
                hash_vals.append(C.phash(img))
                if profile == "gold":
                    st["night"] += float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()) < th["night_brightness"]
            records.append({"key": key, "split": sname, "stem": img_path.stem, "img": img_path, "lab": lab_path,
                            "classes": set(classes), "repeat": repeat})
            if k % 1000 == 0:
                print(f"  {sname}: {k}/{len(imgs)} ảnh")
        if labels_root.is_dir():
            for p in labels_root.rglob("*.txt"):
                if p.relative_to(labels_root).with_suffix("").as_posix() not in stems:
                    issues.add("label_orphan", f"{sname}/{p.relative_to(labels_root).as_posix()}")
        split_stats[sname] = st
        split_hashes[sname] = (hash_names, np.array(hash_vals, dtype=np.uint64))

        # L2 theo split
        n_img = max(1, st["images"] - sum(1 for r in records if r["split"] == sname and r["repeat"]))
        if st["crowded"] / n_img > th["max_crowded_frac"]:
            issues.add("crowded_frac", f"{sname}: {st['crowded']}/{n_img} ảnh ({st['crowded'] / n_img:.1%})")
        if st["background"] / n_img > th["max_background_frac"]:
            issues.add("background_frac", f"{sname}: {st['background']}/{n_img} ảnh ({st['background'] / n_img:.1%})")
        absent_ok = {C.NAME2ID[n] for n in expect.get("absent_ok") or []}
        for c, rule in class_rules(profile, split, expect, args.min_instances).items():
            n = st["instances"][c]
            if n == 0 and c not in absent_ok:
                issues.add("class_missing", f"{sname}/{C.CLASSES[c]}")
            if n < rule["min"]:
                issues.add("class_below_min", f"{sname}/{C.CLASSES[c]}: {n} < min {rule['min']}")
            elif 0 < n < rule["target"]:
                issues.add("class_below_target", f"{sname}/{C.CLASSES[c]}: {n} < {rule['target']}")
        if profile == "gold" and do_hash and st["images"]:
            frac = st["night"] / st["images"]
            if frac < th["min_night_frac"]:
                issues.add("night_frac", f"{st['night']}/{st['images']} ảnh đêm ({frac:.0%})")

    # L2 converter
    converter = check_converter_stats(root, split_stats, issues, th) if profile == "source" else None

    # L3
    if do_hash:
        for sname, (names, hv) in split_hashes.items():
            for a, b, d in near_dups_within(names, hv, th["dup_threshold"]):
                issues.add("dup_within", f"{b} ~ {a} (d={d})", b)
        if "train" in split_hashes and "val" in split_hashes:
            tn, th_ = split_hashes["train"]
            vn, vh = split_hashes["val"]
            for v, t, d in near_dups_against(vn, vh, tn, th_, th["leak_threshold"]):
                issues.add("leak_train_val", f"{v} ~ {t} (d={d})", v)
        for ref in args.against:
            ref_paths = C.list_images(ref)
            ref_h = hash_images(ref_paths, Path(ref).name)
            ref_n = [str(p) for p in ref_paths]
            for sname, (names, hv) in split_hashes.items():
                for a, b, d in near_dups_against(names, hv, ref_n, ref_h, th["leak_threshold"]):
                    issues.add("leak_ref", f"{a} ~ {b} (d={d})", a)
    if "train" in split_hashes and "val" in split_hashes:   # trùng tên (kể cả khi --no-hash)
        tbase = {REPEAT_RE.sub("", r["stem"]) for r in records if r["split"] == "train"}
        for r in records:
            if r["split"] == "val" and r["stem"] in tbase:
                issues.add("leak_train_val", f"{r['key']}: cùng tên với ảnh train", r["key"])
    if profile == "gold" and not args.against:
        issues.add("leak_unchecked", "thêm --against <ảnh train Việt Nam> để chắc gold val không lẫn vào train")

    # L4
    if profile == "merged":
        check_manifest(root, split_stats, issues, args.skip_zip_sha)

    # L5
    audit = None
    if not args.no_sample:
        audit = human_audit(root, records, issues, levels, th, args.seed, args.resample, args.render)

    return finish(root, profile, args, expect, th, levels, issues, split_stats, converter, audit)


def finish(root, profile, args, expect, th, levels, issues, split_stats, converter, audit):
    waivers = expect.get("waivers") or []
    checks, n_err, n_warn = {}, 0, 0
    for code, items in issues.items.items():
        lvl = levels[code]
        if lvl == O:
            continue
        waived = [it for it in items if any(wv["check"] == code and fnmatch.fnmatch(it, wv["item"]) for wv in waivers)]
        active = [it for it in items if it not in set(waived)]
        checks[code] = {"level": lvl, "layer": CHECKS[code][0], "message": CHECKS[code][1], "count": len(active),
                        "waived": len(waived), "examples": active[:50]}
        n_err += len(active) if lvl == E else 0
        n_warn += len(active) if lvl == W else 0
    done = audit is not None and audit["sample_size"] and not audit["pending"]
    verdict = "FAIL" if n_err else ("PASS" if done else "PASS_AUTO")

    report = {
        "data": str(root), "profile": profile, "verdict": verdict, "errors": n_err, "warnings": n_warn,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": git_info(),
        "expect": args.expect, "thresholds": th, "against": args.against, "hashed": not args.no_hash,
        "splits": {s: {"images": st["images"], "background_images": st["background"],
                       "crowded_images": st["crowded"],
                       "instances_per_class": {C.CLASSES[c]: st["instances"][c] for c in range(len(C.CLASSES))},
                       "images_per_class": {C.CLASSES[c]: st["images_per_class"][c] for c in range(len(C.CLASSES))}}
                   for s, st in split_stats.items()},
        "converter": converter, "human_audit": audit,
        "checks": dict(sorted(checks.items(), key=lambda kv: (kv[1]["level"] != E, kv[1]["layer"], kv[0]))),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    (qa_dir(root) / "report.json").write_text(text, encoding="utf-8")
    if args.archive:
        arch = Path(args.archive)
        arch.mkdir(parents=True, exist_ok=True)
        (arch / f"{root.name}.json").write_text(text, encoding="utf-8")
        print(f"Đã lưu bản sao báo cáo -> {arch / (root.name + '.json')}")
    print_report(report, args.show)
    return report


def print_report(r, show):
    print(f"\n{r['data']}  [profile {r['profile']}]")
    names = list(r["splits"])
    print(f"{'id':>2}  {'lớp':<14}" + "".join(f" {n + ' inst':>11} {'ảnh':>6}" for n in names))
    for c, cname in enumerate(C.CLASSES):
        row = "".join(f" {r['splits'][n]['instances_per_class'][cname]:>11} {r['splits'][n]['images_per_class'][cname]:>6}"
                      for n in names)
        print(f"{c:>2}  {cname:<14}{row}")
    for n in names:
        s = r["splits"][n]
        print(f"  {n}: {s['images']} ảnh, {s['background_images']} ảnh nền, {s['crowded_images']} ảnh > "
              f"{C.MAX_INSTANCES} instance")
    for code, ck in r["checks"].items():
        tag = "LỖI" if ck["level"] == E else "CẢNH BÁO"
        waived = f" (+{ck['waived']} miễn trừ)" if ck["waived"] else ""
        if ck["count"] or ck["waived"]:
            print(f"[{tag}][{ck['layer']}] {code}: {ck['count']}{waived}  {ck['message']}")
            for ex in ck["examples"][:show]:
                print(f"      {ex}")
    a = r["human_audit"]
    audit_txt = ("chưa tạo" if a is None else
                 "mẫu rỗng" if not a["sample_size"] else
                 f"đã chấm {a['ok'] + a['bad']}/{a['sample_size']} ({a['bad']} lỗi)" if a["pending"] else
                 f"{a['bad']}/{a['sample_size']} lỗi ({a['error_rate']:.0%})")
    print(f"\nBáo cáo: {Path(r['data']) / QA_DIR / 'report.json'}")
    print(f"KẾT LUẬN: {r['verdict']}  ({r['errors']} lỗi, {r['warnings']} cảnh báo; mẫu người xem: {audit_txt})")
    n_img = sum(s["images"] for s in r["splits"].values())
    print(f"Dòng cho experiment_log.md: | {Path(r['data']).name} | | {n_img} | {r['errors']} lỗi / "
          f"{r['warnings']} cảnh báo | {audit_txt} | {r['verdict']} (commit {r['tool']['commit']}) |")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="thư mục có images/ và labels/ (có thể có split train/val)")
    ap.add_argument("--profile", choices=("auto", *PROFILES), default="auto")
    ap.add_argument("--expect", help="file kỳ vọng của nguồn: configs/qa/<nguồn>.yaml")
    ap.add_argument("--against", nargs="*", default=[], help="thư mục ảnh tham chiếu để kiểm tra rò rỉ")
    ap.add_argument("--min-instances", type=int, help="ghi đè target số instance mỗi lớp (split chính)")
    ap.add_argument("--no-hash", action="store_true", help="bỏ L3 pHash (nhanh hơn, không kiểm tra trùng/rò rỉ ảnh)")
    ap.add_argument("--skip-zip-sha", action="store_true", help="không băm lại zip (zip lớn)")
    ap.add_argument("--no-sample", action="store_true", help="không tạo / đọc mẫu người xem")
    ap.add_argument("--resample", action="store_true",
                    help="tạo lại qa/sample.csv (bản đã chấm cũ được lưu thành qa/sample.<giờ>.csv)")
    ap.add_argument("--render", action="store_true", help="vẽ ảnh mẫu người xem ra <data>/qa/preview")
    ap.add_argument("--archive", help="chép report vào thư mục này dưới tên <dataset>.json (ví dụ reports/qa, rồi commit)")
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--show", type=int, default=5, help="số ví dụ in ra cho mỗi mã")
    return ap.parse_args(argv)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = run(parse_args(argv))
    if report["verdict"] == "FAIL":
        print("KHÔNG ĐẠT: sửa các lỗi trên (hoặc ghi miễn trừ có lý do trong file --expect) trước khi đi tiếp.")
        sys.exit(1)


if __name__ == "__main__":
    main()
