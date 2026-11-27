"""Chia THỬ train/val khi chưa có gold val: đọc thẳng các zip đã convert (không cần giải nén), lấy một tập val cố định từ mỗi nguồn,
phần còn lại làm train, ghi ra MỘT zip images/{train,val} + labels/{train,val} mà notebook nhận trực tiếp.

Val được chọn theo seed + tên nguồn, KHÔNG phụ thuộc nguồn nào được đưa vào train => các lượt khác nhau (A0 chỉ Vistas, A1 Vistas + IDD)
có cùng một tập val và so sánh được với nhau. Ảnh val không bao giờ nằm trong train.

Lưu ý: val này là ảnh nước ngoài (Vistas/IDD), chỉ để thử pipeline và so sánh tương đối. Khi có gold val Việt Nam thì dùng
build_dataset.py với val = gold val. Ảnh IDD lấy từ video nên có thể có khung gần giống nhau giữa train và val: Bước 2 của notebook
(REMOVE_LEAKAGE_FROM_TRAIN = True) tự bỏ các ảnh train gần trùng val.

Ví dụ (A1 = Vistas + IDD, A0 = chỉ Vistas; cùng --sources và --val-per-source thì cùng val):
    python tools/split_trial.py --sources vistas=E:/data/vistas_yolo.zip idd=E:/data/idd_yolo_train.zip \
        --val-per-source 250 --train vistas idd --name trial_a1_vistas_idd --out E:/data/datasets
    python tools/split_trial.py --sources vistas=E:/data/vistas_yolo.zip idd=E:/data/idd_yolo_train.zip \
        --val-per-source 250 --train vistas --name trial_a0_vistas --out E:/data/datasets
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

CLASSES = ["Car", "Bus", "Truck", "Other vehicle", "Motorcycle", "Motorcyclist",
           "Bicycle", "Cyclist", "Pedestrian", "Sign", "Traffic light"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Source:
    """Một nguồn dạng zip hoặc thư mục, có images/ và labels/ (có thể nằm trong một thư mục con). Bỏ qua qa/."""

    def __init__(self, name, path):
        self.name, self.path = name, Path(path)
        if self.path.is_file():
            self.zf = zipfile.ZipFile(self.path)
            files = [n for n in self.zf.namelist() if not n.endswith("/")]
        else:
            self.zf = None
            files = [p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()]
        files = [f for f in files if "qa" not in PurePosixPath(f).parts[:-1]]
        labels = {self._key(f): f for f in files if f.endswith(".txt")}
        self.items = []  # (ảnh, nhãn hoặc None)
        for f in sorted(files):
            p = PurePosixPath(f)
            if p.suffix.lower() in IMG_EXTS and "images" in p.parts:
                self.items.append((f, labels.get(self._key(f))))
        if not self.items:
            raise SystemExit(f"[{name}] không thấy ảnh trong thư mục images/ của {path}")

    @staticmethod
    def _key(f):
        parts = list(PurePosixPath(f).with_suffix("").parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] in ("images", "labels"):
                parts[i] = "*"
                break
        return "/".join(parts)

    def read(self, member):
        return self.zf.read(member) if self.zf else (self.path / member).read_bytes()


def class_counts(text):
    return Counter(int(ln.split()[0]) for ln in text.splitlines() if ln.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", required=True, help="ten=đường_dẫn_zip_hoặc_thư_mục, ví dụ vistas=E:/x/vistas_yolo.zip")
    ap.add_argument("--val-per-source", type=int, default=250, help="số ảnh val lấy từ MỖI nguồn (mặc định 250)")
    ap.add_argument("--train", nargs="+", required=True, help="tên các nguồn đưa vào train (val luôn lấy từ mọi nguồn trong --sources)")
    ap.add_argument("--train-max", nargs="*", default=[], help="giới hạn ảnh train theo nguồn, ví dụ vistas=3000 idd=3000")
    ap.add_argument("--name", required=True, help="tên bộ dữ liệu, ví dụ trial_a1_vistas_idd")
    ap.add_argument("--out", required=True, help="thư mục ghi kết quả")
    ap.add_argument("--seed", type=int, default=71)
    args = ap.parse_args()

    srcs = {}
    for s in args.sources:
        name, _, path = s.partition("=")
        srcs[name] = Source(name, path)
    unknown = set(args.train) - set(srcs)
    if unknown:
        raise SystemExit(f"--train có nguồn không nằm trong --sources: {unknown}")
    train_max = {k: int(v) for k, v in (t.split("=") for t in args.train_max)}

    out_root = Path(args.out) / args.name
    zip_path = out_root.with_suffix(".zip")
    if out_root.exists() or zip_path.exists():
        raise SystemExit(f"{out_root} hoặc {zip_path} đã tồn tại: đổi --name (không ghi đè bộ cũ)")
    out_root.parent.mkdir(parents=True, exist_ok=True)

    plan, manifest_src = [], {}
    for name, src in srcs.items():
        rng = random.Random(f"{args.seed}-{name}")          # val chỉ phụ thuộc seed + tên nguồn
        idx = list(range(len(src.items)))
        rng.shuffle(idx)
        k = min(args.val_per_source, max(1, len(idx) // 5))   # val tối đa 20% của nguồn, ít nhất 1 ảnh
        val_idx, rest = sorted(idx[:k]), idx[k:]
        plan += [("val", name, i) for i in val_idx]
        n_train = 0
        if name in args.train:
            if name in train_max and train_max[name] < len(rest):
                rest = random.Random(f"{args.seed}-{name}-train").sample(rest, train_max[name])
            plan += [("train", name, i) for i in sorted(rest)]
            n_train = len(rest)
        manifest_src[name] = {"path": str(src.path), "images": len(src.items), "val": len(val_idx), "train": n_train}
        print(f"  {name:<8} {len(src.items):>6} ảnh -> val {len(val_idx):>4}, train {n_train:>5}")

    totals, n_img, seen = {"train": Counter(), "val": Counter()}, Counter(), set()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for k, (split, name, i) in enumerate(plan, 1):
            img, lab = srcs[name].items[i]
            stem = PurePosixPath(img).stem
            if stem in seen:
                raise SystemExit(f"Trùng tên ảnh giữa các nguồn: {stem}")
            seen.add(stem)
            text = srcs[name].read(lab).decode("utf-8") if lab else ""
            totals[split].update(class_counts(text))
            n_img[split] += 1
            for rel, data in ((f"images/{split}/{stem}{PurePosixPath(img).suffix.lower()}", srcs[name].read(img)),
                              (f"labels/{split}/{stem}.txt", text.encode("utf-8"))):
                dst = out_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
                zf.writestr(zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0)), data)
            if k % 1000 == 0:
                print(f"  ... {k}/{len(plan)}")

    h = hashlib.sha256()
    with open(zip_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    manifest = {"name": args.name, "seed": args.seed, "kind": "trial split (val = Vistas/IDD, chưa phải gold val)",
                "args": vars(args), "sources": manifest_src, "images": dict(n_img),
                "instances_per_class": {s: {CLASSES[c]: totals[s][c] for c in range(len(CLASSES))} for s in totals},
                "zip": zip_path.name, "zip_sha256": h.hexdigest()}
    (out_root.parent / f"{args.name}.manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{args.name}: train {n_img['train']} ảnh, val {n_img['val']} ảnh")
    print(f"{'lớp':<14}{'train':>8}{'val':>6}")
    for c, cname in enumerate(CLASSES):
        flag = "  <-- val < 15" if totals["val"][c] < 15 else ""
        print(f"{cname:<14}{totals['train'][c]:>8}{totals['val'][c]:>6}{flag}")
    print(f"\nZip: {zip_path} ({zip_path.stat().st_size / 2**30:.2f} GB), sha256 {manifest['zip_sha256'][:16]}…")
    print("Bước tiếp: tải zip lên Kaggle Datasets / Google Drive, đặt DATASET_PATH trong notebooks/train_t66.ipynb")


if __name__ == "__main__":
    main()
