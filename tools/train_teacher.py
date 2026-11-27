"""Huấn luyện teacher (yolo26l/m-seg @ 1024) trên đúng 11 lớp để gán nháp ảnh Việt Nam và soi nhãn.

Teacher KHÔNG được nộp (hệ thống chấm chỉ nhận yolo26n-seg): nó chỉ dùng cho tools/pseudo_label.py và tools/label_audit.py.
Lưu ý (tools/README.md, L6): teacher không được train trên dữ liệu mà nó sẽ soi; muốn soi gold val thì gold val chỉ làm val.

Đầu vào: thư mục hoặc zip do tools/build_dataset.py tạo (images/{train,val}, labels/{train,val}).
Đầu ra: <out>/<name>/weights/best.pt, và bản sao <out>/teacher.pt + teacher_report.json.
Cần: pip install ultralytics==8.4.214 (GPU). Kaggle T4: imgsz 1024 + l-seg chậm 4-6 lần cỡ n -> dùng --model yolo26m-seg.pt
hoặc giảm --epochs nếu không có GPU mạnh (plan/training_submission.md §2).

Ví dụ:
    python tools/train_teacher.py --data /kaggle/input/merged-v1 --out /kaggle/working/teacher
    python tools/train_teacher.py --data merged_v1.zip --model yolo26m-seg.pt --epochs 25 --batch 8 --out runs/teacher
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

CLASSES = ["Car", "Bus", "Truck", "Other vehicle", "Motorcycle", "Motorcyclist",
           "Bicycle", "Cyclist", "Pedestrian", "Sign", "Traffic light"]
TEAM_SEED = 71


def find_root(path: Path, work: Path) -> Path:
    """Thư mục chứa images/train và images/val (giải nén nếu là zip)."""
    if path.is_file() and path.suffix == ".zip":
        dst = work / "_data"
        if not dst.exists():
            with zipfile.ZipFile(path) as zf:
                zf.extractall(dst)
        path = dst
    for cand in [path, *sorted(p for p in path.rglob("images") if p.is_dir())]:
        root = cand if (cand / "images").is_dir() else cand.parent
        if (root / "images" / "train").is_dir() and (root / "images" / "val").is_dir():
            return root.resolve()
    raise SystemExit(f"Không tìm thấy images/train và images/val trong {path}")


def write_data_yaml(root: Path, work: Path) -> Path:
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    y = work / "teacher_data.yaml"
    y.write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    return y


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="thư mục hoặc zip dataset (images/{train,val})")
    ap.add_argument("--out", required=True, type=Path, help="thư mục output")
    ap.add_argument("--model", default="yolo26l-seg.pt", help="yolo26l-seg.pt (mặc định) hoặc yolo26m-seg.pt / yolo26x-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8, help="T4 + l-seg @1024: 4-8; A100: 16-32; -1 = tự chọn")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", default="false", choices=["false", "ram", "disk"])
    ap.add_argument("--name", default="teacher")
    ap.add_argument("--resume", action="store_true", help="tiếp tục từ <out>/<name>/weights/last.pt (khi phiên GPU bị ngắt)")
    ap.add_argument("--copy-paste", type=float, default=0.2, help="augmentation cho lớp hiếm (teacher không bị ràng buộc như mô hình nộp)")
    args = ap.parse_args()

    from ultralytics import YOLO  # import muộn để --help chạy được khi chưa cài

    args.out.mkdir(parents=True, exist_ok=True)
    last = args.out / args.name / "weights" / "last.pt"
    t0 = time.time()
    if args.resume:
        if not last.exists():
            raise SystemExit(f"Không có {last} để resume")
        model = YOLO(str(last))
        model.train(resume=True)
        data_yaml = None
    else:
        root = find_root(args.data, args.out)
        data_yaml = write_data_yaml(root, args.out)
        n_train = sum(1 for _ in (root / "images" / "train").iterdir())
        n_val = sum(1 for _ in (root / "images" / "val").iterdir())
        print(f"Teacher {args.model} @ {args.imgsz}: {n_train} ảnh train, {n_val} ảnh val, {args.epochs} epoch")
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, patience=args.patience,
            workers=args.workers, cache=False if args.cache == "false" else args.cache,
            seed=TEAM_SEED, deterministic=False,  # teacher không cần tái lập từng bit, ưu tiên tốc độ
            cos_lr=True, close_mosaic=10, copy_paste=args.copy_paste,
            project=str(args.out), name=args.name, exist_ok=True, plots=True,
        )

    best = Path(model.trainer.best)
    names = [model.names[i] for i in sorted(model.names)]
    if names != CLASSES:
        raise SystemExit(f"Tên lớp của teacher {names} khác 11 lớp cuộc thi: kiểm tra data.yaml")
    dst = args.out / "teacher.pt"
    shutil.copyfile(best, dst)

    metrics = YOLO(str(dst)).val(data=str(data_yaml or model.trainer.args.data), imgsz=args.imgsz, batch=args.batch, plots=False,
                                 project=str(args.out), name=f"{args.name}_val", exist_ok=True)
    seg = metrics.seg
    report = {
        "model": args.model, "imgsz": args.imgsz, "epochs": args.epochs, "data": str(args.data),
        "best_pt": str(best), "teacher_pt": str(dst), "sha256": sha256(dst),
        "mask_map50_95": float(seg.map), "mask_map50": float(seg.map50),
        "per_class_mask_ap": {CLASSES[int(c)]: float(a) for c, a in zip(metrics.ap_class_index, seg.maps[metrics.ap_class_index])},
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (args.out / "teacher_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Teacher xong: {dst} | mask mAP50-95 (val, ultralytics) = {100 * report['mask_map50_95']:.2f}")
    print("Bước tiếp: python tools/pseudo_label.py --model teacher.pt --images <vn_frames> --out <vn_pseudo_t> --augment")


if __name__ == "__main__":
    main()
