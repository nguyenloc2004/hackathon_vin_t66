"""Dựng một phiên bản dataset từ recipe YAML -> thư mục + zip + manifest (plan/data_pipeline.md §7).

Zip có cấu trúc images/{train,val}, labels/{train,val}: notebook của cuộc thi nhận trực tiếp (dùng val có sẵn,
không chia ngẫu nhiên). Mọi lựa chọn ngẫu nhiên dùng seed của recipe (mặc định 71) nên dựng lại ra đúng kết quả.

Ví dụ recipe: configs/recipes/example.yaml

    python tools/build_dataset.py --recipe configs/recipes/merged_v1.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

import common as C


def scan(source_dir):
    """-> list (ảnh, nhãn, Counter lớp)."""
    items = [(img, lab, C.class_counts(lab)) for img, lab in C.iter_dataset(source_dir)]
    if not items:
        raise SystemExit(f"Nguồn rỗng hoặc sai cấu trúc (cần images/ và labels/): {source_dir}")
    return items


def priority_select(items, k, rng, random_frac=0.2):
    """Chọn k ảnh: 80% theo điểm ưu tiên lớp hiếm, 20% ngẫu nhiên trong phần còn lại."""
    if k >= len(items):
        return list(items)
    totals = Counter()
    for _, _, cnt in items:
        totals.update(cnt)
    n_max = max(totals.values()) if totals else 1
    w = {c: (n_max / max(1, totals[c])) ** 0.5 for c in range(len(C.CLASSES))}

    def score(cnt):
        s = sum(w[c] * min(n, 10) for c, n in cnt.items())
        s += 2 * bool(cnt.get(5) or cnt.get(7)) + 1 * bool(cnt.get(9) or cnt.get(10))
        return s

    order = sorted(range(len(items)), key=lambda i: (-score(items[i][2]), items[i][0].name))
    n_top = int(round(k * (1 - random_frac)))
    chosen = order[:n_top]
    rest = order[n_top:]
    chosen += rng.sample(rest, k - n_top)
    return [items[i] for i in sorted(chosen)]


def phashes(paths):
    return np.array([C.phash(C.read_image(p)) for p in paths], dtype=np.uint64)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    recipe = yaml.safe_load(Path(args.recipe).read_text(encoding="utf-8"))
    name, seed = recipe["name"], int(recipe.get("seed", 71))
    sources = {k: Path(v) for k, v in recipe["sources"].items()}
    out_root = Path(recipe.get("out_dir", "data/datasets")) / name
    if out_root.exists():
        raise SystemExit(f"{out_root} đã tồn tại: đổi tên phiên bản (không ghi đè phiên bản cũ)")
    rng = random.Random(seed)

    # ---- val
    val_items = scan(sources[recipe["val"]["source"]])
    leak_thr = int(recipe.get("exclude_near_duplicates_of_val", 4))
    val_hash = phashes([i[0] for i in val_items]) if leak_thr > 0 else None

    # ---- train
    plan, manifest_sources, seen = [], {}, set()
    for entry in recipe["train"]:
        src = entry["source"]
        items = scan(sources[src])
        k = int(entry.get("max_images", 0)) or len(items)
        sel = priority_select(items, k, rng) if entry.get("select", "random") == "priority" else \
            sorted(rng.sample(items, min(k, len(items))), key=lambda t: t[0].name)
        leaked = 0
        if val_hash is not None:
            keep = []
            for it in sel:
                if C.hamming_to_all(C.phash(C.read_image(it[0])), val_hash).min() <= leak_thr:
                    leaked += 1
                else:
                    keep.append(it)
            sel = keep
        repeat = int(entry.get("repeat", 1))
        cnt = Counter()
        for img, lab, c in sel:
            if img.stem in seen:
                raise SystemExit(f"Trùng tên ảnh giữa các nguồn: {img.stem} (đặt --prefix khác nhau khi chuyển đổi)")
            seen.add(img.stem)
            cnt.update({cc: n * repeat for cc, n in c.items()})
            for r in range(repeat):
                plan.append(("train", img, lab, img.stem if r == 0 else f"{img.stem}_r{r}"))
        manifest_sources[src] = {"available": len(items), "selected": len(sel), "repeat": repeat,
                                 "removed_val_leak": leaked, "select": entry.get("select", "random"),
                                 "instances_per_class": {C.CLASSES[c]: cnt.get(c, 0) for c in range(len(C.CLASSES))}}
        print(f"  {src:<22} chọn {len(sel):>6}/{len(items):<6} ×{repeat}  (loại {leaked} ảnh gần trùng val)")
    for img, lab, _ in val_items:
        plan.append(("val", img, lab, img.stem))

    # ---- ghi
    totals = {"train": Counter(), "val": Counter()}
    n_images = Counter()
    for split, img, lab, stem in plan:
        dst_img = out_root / "images" / split / f"{stem}{img.suffix.lower()}"
        dst_lab = out_root / "labels" / split / f"{stem}.txt"
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lab.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(img, dst_img)
        if lab.is_file():
            shutil.copyfile(lab, dst_lab)
        else:
            dst_lab.write_text("", encoding="utf-8")
        totals[split].update(C.class_counts(dst_lab))
        n_images[split] += 1
    manifest = {"name": name, "seed": seed, "recipe": recipe, "sources": manifest_sources,
                "images": dict(n_images),
                "instances_per_class": {s: {C.CLASSES[c]: totals[s].get(c, 0) for c in range(len(C.CLASSES))} for s in totals}}

    if not args.no_zip:
        zip_path = out_root.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for p in sorted(out_root.rglob("*")):
                if p.is_file():
                    info = zipfile.ZipInfo(p.relative_to(out_root).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                    zf.writestr(info, p.read_bytes())
        manifest["zip"] = zip_path.name
        manifest["zip_sha256"] = sha256_file(zip_path)
    (out_root.parent / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{name}: train {n_images['train']} ảnh, val {n_images['val']} ảnh -> {out_root}")
    print(f"{'lớp':<14} {'train':>8} {'val':>6}")
    for c, cname in enumerate(C.CLASSES):
        flag = "  <-- ít" if totals["train"][c] < 400 or totals["val"][c] < 15 else ""
        print(f"{cname:<14} {totals['train'][c]:>8} {totals['val'][c]:>6}{flag}")
    if "zip_sha256" in manifest:
        print(f"zip sha256: {manifest['zip_sha256']}")
    print("Bước tiếp: qa_check.py trên thư mục này, rồi đưa zip lên Drive/Kaggle và ghi vào plan/experiment_log.md")


if __name__ == "__main__":
    main()
