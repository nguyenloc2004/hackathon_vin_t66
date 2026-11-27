"""Xem nhanh một hoặc nhiều bộ dữ liệu (zip hoặc thư mục) mà KHÔNG cần giải nén: cấu trúc, số ảnh, đã có nhãn chưa, số instance mỗi lớp.

Dùng để trả lời "bộ này dùng được để train / làm val chưa?" trước khi đưa vào build_dataset.py hoặc notebook.

    python tools/inspect_data.py D:/data/converted/*.zip D:/data/gold_val/*.zip
    python tools/inspect_data.py D:/data/converted/vistas          # thư mục cũng được
"""
from __future__ import annotations

import argparse
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

CLASSES = ["Car", "Bus", "Truck", "Other vehicle", "Motorcycle", "Motorcyclist",
           "Bicycle", "Cyclist", "Pedestrian", "Sign", "Traffic light"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Source:
    """Liệt kê file và đọc file text, chung cho zip và thư mục."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_file() and self.path.suffix.lower() == ".zip":
            self.zf = zipfile.ZipFile(self.path)
            self.names = [n for n in self.zf.namelist() if not n.endswith("/")]
        else:
            self.zf = None
            self.names = [p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()]

    def read_text(self, name):
        if self.zf:
            return self.zf.read(name).decode("utf-8", errors="replace")
        return (self.path / name).read_text(encoding="utf-8", errors="replace")


def split_of(parts):
    for p in parts:
        if p.lower() in ("train", "val", "valid", "validation", "test"):
            return {"valid": "val", "validation": "val"}.get(p.lower(), p.lower())
    return "(không chia)"


def label_key(name):
    """images/a/x.jpg -> labels/a/x ; x.jpg -> x (nhãn cạnh ảnh)."""
    parts = list(PurePosixPath(name).with_suffix("").parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
    return "/".join(parts)


def inspect(path):
    src = Source(path)
    imgs = [n for n in src.names if PurePosixPath(n).suffix.lower() in IMG_EXTS
            and "qa" not in PurePosixPath(n).parts[:-1]]   # qa/preview/*.jpg là ảnh xem trước của qa_check, không phải dữ liệu
    txts = {str(PurePosixPath(n).with_suffix("")): n for n in src.names if n.endswith(".txt") and "/qa/" not in f"/{n}"}
    extras = sorted({PurePosixPath(n).name for n in src.names if PurePosixPath(n).suffix.lower() in {".json", ".yaml", ".yml", ".csv", ".md"}})
    top = sorted({PurePosixPath(n).parts[0] for n in src.names})

    per_split = {}
    inst, img_with = Counter(), Counter()
    n_box, n_bad, n_poly = 0, 0, 0
    for n in imgs:
        sp = split_of(PurePosixPath(n).parts[:-1])
        d = per_split.setdefault(sp, Counter())
        d["images"] += 1
        key = label_key(n)
        lab = txts.get(key) or txts.get(str(PurePosixPath(n).with_suffix("")))
        if lab is None:
            d["no_label_file"] += 1
            continue
        lines = [ln.split() for ln in src.read_text(lab).splitlines() if ln.strip()]
        if not lines:
            d["empty_label"] += 1
            continue
        d["labeled"] += 1
        seen = set()
        for parts in lines:
            try:
                c = int(parts[0])
            except ValueError:
                n_bad += 1
                continue
            if not 0 <= c < len(CLASSES):
                n_bad += 1
                continue
            if len(parts) == 5:
                n_box += 1
            elif len(parts) >= 7 and len(parts) % 2 == 1:
                n_poly += 1
            else:
                n_bad += 1
                continue
            inst[c] += 1
            d["instances"] += 1
            seen.add(c)
        for c in seen:
            img_with[c] += 1

    print("=" * 78)
    print(f"{src.path.name}  ({'zip' if src.zf else 'thư mục'}, {src.path.stat().st_size / 2**20:.0f} MB)" if src.zf else f"{src.path}  (thư mục)")
    print(f"  Thư mục gốc: {', '.join(top[:8])}{' ...' if len(top) > 8 else ''}")
    if extras:
        print(f"  File kèm: {', '.join(extras[:10])}")
    total = Counter()
    for sp, d in sorted(per_split.items()):
        total.update(d)
        print(f"  [{sp}] {d['images']} ảnh: {d['labeled']} có nhãn, {d['empty_label']} nhãn rỗng (ảnh nền), "
              f"{d['no_label_file']} KHÔNG có file nhãn, {d['instances']} instance")
    if not imgs:
        print("  !! Không có ảnh nào")
        return
    print(f"  Dòng nhãn: {n_poly} đa giác, {n_box} hộp 5 cột, {n_bad} lỗi")
    print(f"  {'id':>2}  {'lớp':<14}{'instance':>9}{'số ảnh':>8}")
    for i, name in enumerate(CLASSES):
        print(f"  {i:>2}  {name:<14}{inst[i]:>9}{img_with[i]:>8}")

    # Kết luận
    verdict = []
    if total["labeled"] == 0:
        verdict.append("CHƯA CÓ NHÃN: chỉ là ảnh (vd ứng viên chờ gán nhãn) -> chưa dùng để train hay làm val được")
    elif total["no_label_file"] > 0.5 * total["images"]:
        verdict.append("Phần lớn ảnh chưa có nhãn -> kiểm tra lại đã xuất nhãn từ CVAT chưa")
    if n_box:
        verdict.append("Có nhãn HỘP 5 cột -> notebook từ chối, cần nhãn đa giác (YOLO-seg)")
    if n_bad:
        verdict.append("Có dòng nhãn lỗi -> chạy qa_check.py để xem chi tiết")
    if set(per_split) == {"(không chia)"}:
        verdict.append("Chưa chia train/val: là MỘT nguồn -> đưa vào recipe của build_dataset.py "
                       "(notebook nhận trực tiếp thì sẽ tự chia ngẫu nhiên, val không phải gold val)")
    elif {"train", "val"} <= set(per_split):
        verdict.append("Đã có images/{train,val}: notebook nhận trực tiếp")
    missing = [CLASSES[i] for i in range(len(CLASSES)) if inst[i] == 0]
    if total["labeled"] and missing:
        verdict.append("Thiếu hẳn lớp: " + ", ".join(missing))
    print("  => " + "\n  => ".join(verdict or ["Có nhãn đa giác, không lỗi định dạng"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    for p in ap.parse_args().paths:
        if not Path(p).exists():
            print(f"Không thấy {p}")
            continue
        inspect(p)


if __name__ == "__main__":
    main()
