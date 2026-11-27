"""Tìm ảnh trùng / gần trùng bằng perceptual hash.

Hai chế độ:
  1. Trong một thư mục: giữ ảnh đầu tiên (theo tên), đánh dấu các ảnh cách nó <= --threshold bit.
  2. So với tập tham chiếu (--against, ví dụ gold val): đánh dấu mọi ảnh trong --dir gần trùng ảnh tham chiếu
     (chống rò rỉ train/val).

Mặc định chỉ ghi báo cáo dedup_report.csv. Thêm --move-to để chuyển ảnh trùng (kèm nhãn nếu có) ra chỗ khác.

Ví dụ:
    python tools/dedup.py --dir D:/data/raw/vn_frames --threshold 6 --move-to D:/data/raw/vn_frames_dup
    python tools/dedup.py --dir D:/data/raw/vn_frames --against D:/data/gold_val/images --threshold 4 --move-to D:/data/raw/leak
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

import common as C


def hash_dir(root):
    paths = C.list_images(root)
    hashes = np.zeros(len(paths), dtype=np.uint64)
    for i, p in enumerate(paths):
        hashes[i] = np.uint64(C.phash(C.read_image(p)))
        if (i + 1) % 1000 == 0:
            print(f"  đã băm {i + 1}/{len(paths)}")
    return paths, hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="thư mục ảnh (có thể là <dataset>/images)")
    ap.add_argument("--against", help="thư mục ảnh tham chiếu (gold val)")
    ap.add_argument("--threshold", type=int, default=6)
    ap.add_argument("--move-to", help="chuyển ảnh trùng (và nhãn cùng tên trong ../labels) sang đây")
    args = ap.parse_args()

    paths, hashes = hash_dir(args.dir)
    dup_of = {}
    if args.against:
        ref_paths, ref_hashes = hash_dir(args.against)
        for i, h in enumerate(hashes):
            d = C.hamming_to_all(int(h), ref_hashes)
            j = int(d.argmin()) if len(d) else -1
            if j >= 0 and d[j] <= args.threshold:
                dup_of[i] = (str(ref_paths[j]), int(d[j]))
    else:
        removed = np.zeros(len(paths), dtype=bool)
        for i in range(len(paths)):
            if removed[i]:
                continue
            d = C.hamming_to_all(int(hashes[i]), hashes[i + 1:])
            for k in np.nonzero(d <= args.threshold)[0]:
                j = i + 1 + int(k)
                if not removed[j]:
                    removed[j] = True
                    dup_of[j] = (str(paths[i]), int(d[k]))

    report = Path(args.dir) / "dedup_report.csv"
    with open(report, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["duplicate", "of", "hamming"])
        for j, (ref, d) in sorted(dup_of.items()):
            w.writerow([str(paths[j]), ref, d])
    print(f"{len(dup_of)}/{len(paths)} ảnh trùng hoặc gần trùng. Báo cáo: {report}")

    if args.move_to:
        dst, root = Path(args.move_to), Path(args.dir)
        for j in dup_of:
            p = paths[j]
            rel = p.relative_to(root)
            (dst / "images" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst / "images" / rel))
            # nếu --dir là <dataset>/images thì nhãn nằm ở <dataset>/labels
            lab = root.parent / "labels" / rel.with_suffix(".txt") if root.name == "images" else None
            if lab is not None and lab.is_file():
                (dst / "labels" / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(lab), str(dst / "labels" / rel.with_suffix(".txt")))
        print(f"Đã chuyển {len(dup_of)} ảnh sang {dst}")


if __name__ == "__main__":
    main()
