"""Hàm dùng chung cho mọi converter và công cụ dữ liệu của đội.

Ý tưởng chính: mỗi ảnh được dựng thành một ``Scene`` = một bản đồ instance (``id_map``, int32, 0 = nền) +
vai trò (role) của từng instance. Vật vẽ sau ghi đè vật vẽ trước, nên mask tự động KHÔNG chồng nhau
(đúng guideline mục 2.2). Sau đó ``finalize`` làm theo đúng thứ tự:

    1. loại ảnh có vùng đám đông (crowd) lớn
    2. ghép rider + xe -> Motorcyclist / Cyclist / Other vehicle (guideline mục 3)
    3. bỏ vật nhỏ: cạnh ngắn bbox nhìn thấy < 1,25% cạnh dài ảnh (guideline mục 2.3)
    4. mask -> polygon (nhiều mảnh nối thành một dòng nhãn), chuẩn hóa về [0, 1]

Role hợp lệ:
    car, bus, truck, other, motorcycle, bicycle, person, sign, light  -> xuất thành lớp
    rider     -> phải ghép được với xe (motorcycle / bicycle / other)
    crowd     -> vùng đám đông / nhóm xe không tách được (không xuất, dùng để loại ảnh)
    occluder  -> vật không thuộc 11 lớp nhưng che khuất vật khác (cột, cây...): vẽ để che, không xuất
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

# ------------------------------------------------------------------------------------------------ hằng số
CLASSES = ["Car", "Bus", "Truck", "Other vehicle", "Motorcycle", "Motorcyclist",
           "Bicycle", "Cyclist", "Pedestrian", "Sign", "Traffic light"]
NAME2ID = {n: i for i, n in enumerate(CLASSES)}
RARE_CLASSES = {1, 3, 6, 7}  # Bus, Other vehicle, Bicycle, Cyclist: ưu tiên khi chọn ảnh / duyệt
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ROLE_CLASS = {"car": 0, "bus": 1, "truck": 2, "other": 3, "motorcycle": 4, "motorcyclist": 5,
              "bicycle": 6, "cyclist": 7, "person": 8, "sign": 9, "light": 10}
CLASS_ROLE = {v: k for k, v in ROLE_CLASS.items()}  # lớp đích -> role (cho nhãn đã đúng 11 lớp)
RIDER_TARGET_CLASS = {"motorcycle": 5, "bicycle": 7, "other": 3}  # xe có người ngồi -> lớp mới
RIDER, CROWD, OCCLUDER = "rider", "crowd", "occluder"
VALID_ROLES = set(ROLE_CLASS) | {RIDER, CROWD, OCCLUDER}

DEFAULT_LONG_SIDE = 1280
MIN_SIDE_FRAC = 0.0125      # guideline mục 2.3
MAX_INSTANCES = 100         # max_det của bộ chấm
JPEG_QUALITY = 92

CLASS_COLORS = [  # BGR, dùng cho visualize
    (230, 120, 40), (40, 160, 240), (60, 60, 230), (200, 80, 160), (160, 190, 30), (150, 60, 240),
    (60, 200, 90), (40, 130, 250), (220, 200, 60), (40, 200, 230), (90, 230, 60)]


# ------------------------------------------------------------------------------------------------ ảnh
def list_images(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()
                  and not any(part.startswith(".") or part == "__MACOSX" for part in p.relative_to(root).parts))


def read_image(path):
    """cv2.imread không đọc được đường dẫn Unicode trên Windows, nên đọc qua numpy."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không đọc được ảnh {path}")
    return img


def save_image(path, img, quality=JPEG_QUALITY):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError(f"Không mã hóa được ảnh {path}")
    buf.tofile(str(path))


def resize_scale(h, w, long_side=DEFAULT_LONG_SIDE):
    """Hệ số thu nhỏ để cạnh dài = long_side (không phóng to ảnh nhỏ)."""
    return min(1.0, long_side / max(h, w)) if long_side else 1.0


def resize_image(img, scale):
    if scale >= 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def resize_id_map(id_map, size_wh):
    if (id_map.shape[1], id_map.shape[0]) == tuple(size_wh):
        return id_map
    return cv2.resize(id_map, size_wh, interpolation=cv2.INTER_NEAREST)


# ------------------------------------------------------------------------------------------------ scene
class Scene:
    """Bản đồ instance của một ảnh. Vẽ theo thứ tự từ SAU ra TRƯỚC: vật vẽ sau che vật vẽ trước."""

    def __init__(self, h, w):
        self.h, self.w = int(h), int(w)
        self.id_map = np.zeros((self.h, self.w), dtype=np.int32)
        self.roles = [None]          # roles[i] = role của instance id i (id 0 = nền)
        self.meta = [None]           # thông tin thêm tùy ý (ví dụ score của nhãn nháp)

    def _new_id(self, role, meta):
        if role not in VALID_ROLES:
            raise ValueError(f"role không hợp lệ: {role!r}")
        self.roles.append(role)
        self.meta.append(meta)
        return len(self.roles) - 1

    def paint_polygons(self, role, polygons, meta=None):
        """polygons: list các mảng Nx2 tọa độ pixel (float). Nhiều mảng = nhiều mảnh của cùng một vật."""
        polys = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polygons]
        polys = [p for p in polys if len(p) >= 3]
        if not polys:
            return None
        allp = np.concatenate(polys)
        x0, y0 = np.floor(allp.min(0)).astype(int)
        x1, y1 = np.ceil(allp.max(0)).astype(int) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, self.w), min(y1, self.h)
        if x1 <= x0 or y1 <= y0:
            return None
        local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        pts = [np.round((p - [x0, y0]) * 16).astype(np.int32) for p in polys]  # shift=4: độ chính xác 1/16 px
        cv2.fillPoly(local, pts, 1, lineType=cv2.LINE_8, shift=4)
        if not local.any():
            return None
        iid = self._new_id(role, meta)
        self.id_map[y0:y1, x0:x1][local > 0] = iid
        return iid

    def paint_mask(self, role, mask, meta=None):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.id_map.shape:
            raise ValueError("mask phải cùng kích thước với ảnh")
        if not mask.any():
            return None
        iid = self._new_id(role, meta)
        self.id_map[mask] = iid
        return iid

    @classmethod
    def from_id_map(cls, id_map, roles, meta=None):
        """id_map đã có sẵn (ví dụ Vistas): giá trị pixel 1..n ứng với roles[1..n]."""
        s = cls(*id_map.shape)
        s.id_map = id_map.astype(np.int32, copy=False)
        s.roles = [None] + list(roles)
        s.meta = [None] + (list(meta) if meta is not None else [None] * len(roles))
        return s


# ------------------------------------------------------------------------------------------------ ghép rider
def _bbox(sl):
    sy, sx = sl
    return sx.start, sy.start, sx.stop, sy.stop


def _rider_pairs(scene, slices, areas, dilate_px=7, min_score=0.05, expand=0.3):
    """Mọi cặp (điểm, rider, xe) hợp lệ. Điểm = |rider giãn 7 px ∩ xe| / min(diện tích rider, diện tích xe)."""
    roles = scene.roles
    riders = [i for i in range(1, len(roles)) if roles[i] == RIDER and areas[i] > 0]
    vehicles = [i for i in range(1, len(roles)) if roles[i] in RIDER_TARGET_CLASS and areas[i] > 0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    pairs = []
    for r in riders:
        rx0, ry0, rx1, ry1 = _bbox(slices[r - 1])
        foot_x, foot_y = (rx0 + rx1) / 2, ry1
        for v in vehicles:
            vx0, vy0, vx1, vy1 = _bbox(slices[v - 1])
            ew, eh = (vx1 - vx0) * expand, (vy1 - vy0) * expand
            if not (vx0 - ew <= foot_x <= vx1 + ew and vy0 - eh <= foot_y <= vy1 + eh):
                continue
            x0, y0 = max(min(rx0, vx0) - dilate_px, 0), max(min(ry0, vy0) - dilate_px, 0)
            x1, y1 = min(max(rx1, vx1) + dilate_px, scene.w), min(max(ry1, vy1) + dilate_px, scene.h)
            sub = scene.id_map[y0:y1, x0:x1]
            grown = cv2.dilate((sub == r).astype(np.uint8), kernel) > 0
            inter = int(np.count_nonzero(grown & (sub == v)))
            score = inter / max(1, min(areas[r], areas[v]))
            if score >= min_score:
                pairs.append((score, r, v))
    return riders, pairs


def merge_riders(scene, unmatched="reject", **kw):
    """Gộp rider vào xe gần nhất (tham lam theo điểm). Một xe nhận nhiều rider, một rider chỉ vào một xe.

    unmatched: "reject" (trả về lỗi để loại ảnh), "person" (coi là Pedestrian), "drop" (giữ pixel nhưng không xuất).
    Trả về (danh sách id xe đã có người ngồi, số rider không ghép được).
    """
    areas = np.bincount(scene.id_map.ravel(), minlength=len(scene.roles))
    slices = ndimage.find_objects(scene.id_map, max_label=len(scene.roles) - 1)
    riders, pairs = _rider_pairs(scene, slices, areas, **kw)
    assigned, ridden = {}, set()
    for score, r, v in sorted(pairs, reverse=True):
        if r not in assigned:
            assigned[r] = v
            ridden.add(v)
    for r, v in assigned.items():
        sl = slices[r - 1]
        sub = scene.id_map[sl]
        sub[sub == r] = v
        scene.roles[r] = None
    n_unmatched = 0
    for r in riders:
        if r in assigned:
            continue
        n_unmatched += 1
        if unmatched == "person":
            scene.roles[r] = "person"
        elif unmatched == "drop":
            scene.roles[r] = OCCLUDER
    return ridden, n_unmatched


def riders_from_boxes(person_boxes, vehicle_boxes, min_overlap=0.25):
    """Dataset/mô hình không phân biệt rider với người đi bộ (COCO): đoán người nào đang NGỒI trên xe 2 bánh.

    box = (x0, y0, x1, y1). Heuristic: hộp người chồng lên hộp xe ≥ 25% diện tích người, tâm ngang của người nằm
    trong hộp xe (nới 10%), chân người nằm ở nửa dưới xe, đầu người cao hơn phần trên xe.
    Người đứng cạnh dắt xe thường lệch ngang nên bị loại. Trả về tập chỉ số người là rider.
    LƯU Ý: chỉ là heuristic, phải xem ảnh mẫu bằng visualize.py trước khi dùng cả bộ.
    """
    riders = set()
    for i, (px0, py0, px1, py1) in enumerate(person_boxes):
        parea = max(1e-6, (px1 - px0) * (py1 - py0))
        cx = (px0 + px1) / 2
        for vx0, vy0, vx1, vy1 in vehicle_boxes:
            vw, vh = vx1 - vx0, vy1 - vy0
            iw = max(0.0, min(px1, vx1) - max(px0, vx0))
            ih = max(0.0, min(py1, vy1) - max(py0, vy0))
            if iw * ih / parea < min_overlap:
                continue
            if not (vx0 - 0.1 * vw <= cx <= vx1 + 0.1 * vw):
                continue
            if not (vy0 + 0.3 * vh <= py1 <= vy1 + 0.15 * vh):
                continue
            if py0 > vy0 + 0.2 * vh:
                continue
            riders.add(i)
            break
    return riders


# ------------------------------------------------------------------------------------------------ mask -> polygon
def _bridge_parts(parts):
    """Nối nhiều mảnh polygon của CÙNG một vật thành một đường khép kín (đi sang mảnh kế bằng cạnh nối rồi quay lại).

    Cùng mục đích với ultralytics.data.converter.merge_multi_segment: YOLO-seg chỉ cho một polygon mỗi dòng.
    """
    if len(parts) == 1:
        return parts[0]
    parts = sorted(parts, key=lambda p: -abs(cv2.contourArea(p.astype(np.float32))))
    chain, rest, links = [parts[0]], parts[1:], []
    while rest:
        cur, best = chain[-1], None
        for j, p in enumerate(rest):
            d = ((cur[:, None, :] - p[None, :, :]) ** 2).sum(-1)
            k = int(d.argmin())
            if best is None or d.flat[k] < best[0]:
                best = (d.flat[k], j, *divmod(k, d.shape[1]))
        _, j, a, b = best
        links.append((a, b))
        chain.append(rest.pop(j))

    def walk(i, start):
        p, out = chain[i], []
        link = links[i][0] if i < len(links) else None
        for k in range(len(p) + 1):
            idx = (start + k) % len(p)
            out.append(p[idx])
            if link is not None and idx == link:
                out.extend(walk(i + 1, links[i][1]))
                out.append(p[idx])
                link = None
        return out

    return np.asarray(walk(0, 0), dtype=np.float64)


def mask_to_polygon(mask, offset=(0, 0), eps=0.7, min_part_area=20.0, min_part_frac=0.0015, max_parts=12):
    """mask (bool, đã cắt theo bbox) -> một polygon Nx2 tọa độ pixel ảnh gốc, hoặc None."""
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    bbox_area = mask.shape[0] * mask.shape[1]
    keep = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < max(min_part_area, min_part_frac * bbox_area) and len(contours) > 1:
            continue
        a = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64)
        if len(a) >= 3:
            keep.append((area, a))
    if not keep:  # vật rất nhỏ / mảnh: giữ contour lớn nhất nếu đủ điểm và đủ diện tích
        c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        if len(c) < 3 or abs(cv2.contourArea(c.astype(np.float32))) < 1.5:
            return None
        keep = [(0.0, c)]
    keep = [a for _, a in sorted(keep, key=lambda t: -t[0])[:max_parts]]
    poly = _bridge_parts(keep)
    if abs(cv2.contourArea(poly.astype(np.float32))) < 1.5:
        return None
    # +0.5: đổi chỉ số pixel sang tọa độ liên tục (tâm pixel)
    return poly + np.asarray(offset, dtype=np.float64) + 0.5


def yolo_line(cls_id, poly_px, w, h):
    xs = np.clip(poly_px[:, 0] / w, 0.0, 1.0)
    ys = np.clip(poly_px[:, 1] / h, 0.0, 1.0)
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
    return f"{int(cls_id)} {coords}"


# ------------------------------------------------------------------------------------------------ finalize
@dataclass
class SceneResult:
    lines: list = field(default_factory=list)
    classes: list = field(default_factory=list)
    reject: str | None = None          # lý do loại ảnh (None = dùng được)
    dropped_small: int = 0
    unmatched_riders: int = 0
    flags: list = field(default_factory=list)


def finalize(scene, unmatched_rider="reject", crowd_max_frac=0.01, min_side_frac=MIN_SIDE_FRAC,
             eps=0.7, max_instances=MAX_INSTANCES):
    """Scene -> các dòng nhãn YOLO-seg theo đúng quy ước của cuộc thi."""
    res = SceneResult()
    n = len(scene.roles)
    areas = np.bincount(scene.id_map.ravel(), minlength=n)
    crowd = sum(int(areas[i]) for i in range(1, n) if scene.roles[i] == CROWD)
    if crowd > crowd_max_frac * scene.h * scene.w:
        res.reject = "crowd"
        return res

    ridden, res.unmatched_riders = merge_riders(scene, unmatched=unmatched_rider)
    if res.unmatched_riders and unmatched_rider == "reject":
        res.reject = "unmatched_rider"
        return res

    slices = ndimage.find_objects(scene.id_map, max_label=n - 1)
    min_side = min_side_frac * max(scene.h, scene.w)
    for iid in range(1, n):
        role, sl = scene.roles[iid], slices[iid - 1]
        if sl is None or role not in ROLE_CLASS:
            continue
        cls_id = RIDER_TARGET_CLASS[role] if iid in ridden else ROLE_CLASS[role]
        x0, y0, x1, y1 = _bbox(sl)
        if min(x1 - x0, y1 - y0) < min_side:
            res.dropped_small += 1
            continue
        poly = mask_to_polygon(scene.id_map[sl] == iid, offset=(x0, y0), eps=eps)
        if poly is None:
            res.dropped_small += 1
            continue
        res.lines.append(yolo_line(cls_id, poly, scene.w, scene.h))
        res.classes.append(cls_id)
    if len(res.lines) > max_instances:
        res.flags.append("many_instances")
    return res


# ------------------------------------------------------------------------------------------------ xuất và thống kê
class Stats:
    """Thống kê của một lần chuyển đổi, ghi ra stats.json + rejected.csv."""

    def __init__(self, source):
        self.source = source
        self.images_ok = 0
        self.images_empty = 0
        self.rejected = Counter()
        self.instances = Counter()
        self.dropped_small = 0
        self.flags = Counter()
        self.rejected_rows = []

    def add(self, stem, res: SceneResult):
        if res.reject:
            self.rejected[res.reject] += 1
            self.rejected_rows.append((stem, res.reject))
            return
        self.images_ok += 1
        self.images_empty += int(not res.lines)
        self.instances.update(res.classes)
        self.dropped_small += res.dropped_small
        self.flags.update(res.flags)

    def merge(self, other: "Stats"):
        self.images_ok += other.images_ok
        self.images_empty += other.images_empty
        self.rejected.update(other.rejected)
        self.instances.update(other.instances)
        self.dropped_small += other.dropped_small
        self.flags.update(other.flags)
        self.rejected_rows += other.rejected_rows

    def to_dict(self):
        return {"source": self.source, "images_ok": self.images_ok, "images_empty": self.images_empty,
                "rejected": dict(self.rejected), "dropped_small_instances": self.dropped_small,
                "flags": dict(self.flags),
                "instances_per_class": {CLASSES[c]: self.instances.get(c, 0) for c in range(len(CLASSES))}}

    def write(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "stats.json").write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        with open(out_dir / "rejected.csv", "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows([("stem", "reason"), *self.rejected_rows])

    def print(self):
        d = self.to_dict()
        print(f"[{self.source}] ảnh dùng được: {d['images_ok']} (rỗng: {d['images_empty']}), bị loại: {d['rejected']}, "
              f"vật nhỏ bị bỏ: {d['dropped_small_instances']}, cờ: {d['flags']}")
        for name, cnt in d["instances_per_class"].items():
            print(f"    {name:<14} {cnt:>8}")


def export_scene(stem, img, scene, out_dir, stats=None, **finalize_kw):
    """finalize + ghi ảnh/nhãn ra out_dir/images, out_dir/labels. Trả về SceneResult."""
    res = finalize(scene, **finalize_kw)
    if stats is not None:
        stats.add(stem, res)
    if res.reject:
        return res
    out_dir = Path(out_dir)
    save_image(out_dir / "images" / f"{stem}.jpg", img)
    write_label(out_dir / "labels" / f"{stem}.txt", res.lines)
    return res


# ------------------------------------------------------------------------------------------------ đọc/ghi nhãn YOLO
def write_label(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_label(path):
    """-> list (cls_id, mảng Nx2 tọa độ chuẩn hóa). File không tồn tại = ảnh nền (list rỗng)."""
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        vals = [float(v) for v in parts[1:]]
        out.append((int(float(parts[0])), np.asarray(vals, dtype=np.float64).reshape(-1, 2) if len(vals) % 2 == 0
                    else np.asarray(vals[:-1], dtype=np.float64).reshape(-1, 2)))
    return out


def label_path_for(img_path, images_root, labels_root):
    rel = Path(img_path).relative_to(images_root)
    return Path(labels_root) / rel.with_suffix(".txt")


def iter_dataset(root):
    """Duyệt dataset kiểu YOLO: root/images/**.jpg + root/labels/**.txt. Trả về (ảnh, nhãn)."""
    root = Path(root)
    images_root, labels_root = root / "images", root / "labels"
    for img in list_images(images_root):
        yield img, label_path_for(img, images_root, labels_root)


def class_counts(label_path):
    return Counter(c for c, _ in read_label(label_path))


# ------------------------------------------------------------------------------------------------ perceptual hash
def phash(img_bgr):
    """pHash 64 bit (DCT 32x32 -> 8x8, so với trung vị). Trả về số nguyên Python."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8].ravel()
    bits = d > np.median(d[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming_to_all(h, hashes):
    """Khoảng Hamming từ hash h tới mảng hashes (np.uint64)."""
    x = np.bitwise_xor(hashes, np.uint64(h))
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
