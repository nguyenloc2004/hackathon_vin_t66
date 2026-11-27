"""Sinh notebooks/train_t66.ipynb từ notebook của BTC (docs/train_and_export.ipynb).

Chỉ thay 2 ô được phép sửa (plan/training_submission.md §1):
  * ô "Tham số của đội"  -> preset smoke/ablation/final, TEAM_ID=71, tự đặt thư mục output theo RUN_NAME;
  * ô Bước 3 (huấn luyện) -> TRAIN_ARGS.update(EXTRA_TRAIN_ARGS), name=RUN_NAME;
và thêm 1 ô cuối gom output (ảnh lỗi, báo cáo, biểu đồ, dòng cho experiment_log.md) về thư mục lượt chạy.
Bước 0-2 và 4-6 giữ nguyên từng ký tự. Khi BTC ra notebook mới: thay docs/train_and_export.ipynb rồi chạy lại script.

    python tools/make_team_notebook.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "train_and_export.ipynb"
DST = ROOT / "notebooks" / "train_t66.ipynb"

PARAMS_MARK = "#@title Tham số của đội"
STEP3_MARK = "TRAIN_ARGS = dict("

PARAMS_CELL = r'''#@title Tham số của đội: điền ở đây, mọi ô sau đọc từ đây
# Notebook đội 71, sinh bởi tools/make_team_notebook.py từ notebook BTC. Chỉ ô này và Bước 3 khác bản gốc.
# Mỗi lượt chạy chỉ cần sửa 4 dòng: RUN_NAME, DATASET_PATH, PRESET, EXTRA_TRAIN_ARGS.

# ---- BẮT BUỘC ----
TEAM_ID = 71  #@param {type:"integer"}
RUN_NAME = "R01_merged_v1_baseline"  #@param {type:"string"}
# Quy ước tên: R{NN}_{dataset}_{ghi-chú} (ablation) hoặc F{N}_{dataset}_{ghi-chú} (lượt dài). Ghi đúng tên này vào plan/experiment_log.md.
DATASET_PATH = "/kaggle/input/merged-v1"  #@param {type:"string"}
# Zip/thư mục do tools/build_dataset.py tạo (đã có images/train + images/val = gold val). Kaggle: /kaggle/input/<ten-dataset>.

# ---- Preset (plan/training_submission.md §2, §4) ----
PRESET = "ablation"  #@param ["smoke", "ablation", "final"]
#   smoke    : 3 epoch, kiểm tra pipeline Bước 0-6 (G0), không dùng để nộp thật
#   ablation : 30 epoch, mặc định notebook; mỗi lượt chỉ đổi MỘT yếu tố so với lượt gốc
#   final    : 60 epoch, patience 30, cos_lr, close_mosaic 10 (F1-F3); bật thêm copy_paste/hsv_v nếu A7/A8 thắng
PRESETS = {
    "smoke":    dict(EPOCHS=3,  PATIENCE=3,  extra={}),
    "ablation": dict(EPOCHS=30, PATIENCE=20, extra={}),
    "final":    dict(EPOCHS=60, PATIENCE=30, extra=dict(cos_lr=True, close_mosaic=10)),
}
EXTRA_TRAIN_ARGS = {
    # Thay đổi của lượt này, ghép đè lên preset. Ví dụ:
    # "copy_paste": 0.2,                      # A7
    # "close_mosaic": 10, "hsv_v": 0.5,       # A8
}
EPOCHS_OVERRIDE = None  # số nguyên để đè số epoch của preset (ví dụ F2 = 80), None = theo preset

# ---- Mô hình và huấn luyện ----
MODEL_SIZE = "n"  # cố định: cuộc thi chỉ nhận YOLO26n-seg
MODEL_WEIGHTS = f"yolo26{MODEL_SIZE}-seg.pt"
EPOCHS = EPOCHS_OVERRIDE or PRESETS[PRESET]["EPOCHS"]
BATCH = 16  #@param {type:"integer"}
# 16 chạy tốt trên T4 với cỡ n; nếu báo hết bộ nhớ GPU (CUDA out of memory) thì giảm còn 8.
PATIENCE = PRESETS[PRESET]["PATIENCE"]
TRAIN_IMGSZ = 640  # KHÔNG đổi
IN_KAGGLE = Path("/kaggle/working").exists()
WORKERS = 4 if IN_KAGGLE else 2
CACHE = "ram" if IN_KAGGLE else False   # Kaggle 29 GB RAM đủ cho ~8k ảnh ở 640; hết RAM thì đổi về False
PLOTS = True
TRAIN_EXTRA = {**PRESETS[PRESET]["extra"], **EXTRA_TRAIN_ARGS}

# ---- Dữ liệu và báo cáo ----
VAL_RATIO = 0.2                  # không dùng khi dữ liệu đã chia sẵn (val = gold val)
MIN_TRAIN_INSTANCES = 100
MAX_IMBALANCE_RATIO = 30
DUP_HAMMING_MAX = 2
REMOVE_LEAKAGE_FROM_TRAIN = True  # gold val phải sạch: bỏ khỏi train mọi ảnh trùng/gần giống val

# ---- Tự kiểm tra sau khi xuất ----
SELFCHECK_MAX_IMAGES = 300
CONF_SHOW = 0.25
N_ERROR_IMAGES = 6

# ---- Hệ thống ----
INSTALL_PACKAGES = True
REQUIRE_GPU = True
# Thư mục lưu best.pt, submission.zip, run_report.json của lượt chạy (Bước 3 và 6 tự sao chép vào đây).
# Kaggle: /kaggle/working/{RUN_NAME} (nằm trong Output khi Save Version). Colab: Drive/runs/{RUN_NAME}.
DRIVE_OUTPUT_DIR = (f"/kaggle/working/{RUN_NAME}" if IN_KAGGLE
                    else f"/content/drive/MyDrive/t66/runs/{RUN_NAME}" if IN_COLAB else f"runs/{RUN_NAME}")
WORK_DIR = "/content/work" if IN_COLAB else "/kaggle/temp/work" if IN_KAGGLE else "work"
# Kaggle: làm việc ở /kaggle/temp để Output chỉ chứa thư mục lượt chạy, không chứa bản sao dataset.

assert TEAM_ID == 71 and MODEL_SIZE == "n" and TRAIN_IMGSZ == 640, "Không đổi TEAM_ID / MODEL_SIZE / TRAIN_IMGSZ"
print(f"Lượt {RUN_NAME} | preset {PRESET} | {EPOCHS} epoch | patience {PATIENCE} | thêm {TRAIN_EXTRA or '{}'}")
print(f"Dữ liệu: {DATASET_PATH} | output: {DRIVE_OUTPUT_DIR}")

PARAMS = dict(TEAM_ID=TEAM_ID, MODEL_SIZE=MODEL_SIZE, MODEL_WEIGHTS=MODEL_WEIGHTS, EPOCHS=EPOCHS, BATCH=BATCH, PATIENCE=PATIENCE,
              TRAIN_IMGSZ=TRAIN_IMGSZ, WORKERS=WORKERS, CACHE=CACHE, VAL_RATIO=VAL_RATIO, MIN_TRAIN_INSTANCES=MIN_TRAIN_INSTANCES,
              MAX_IMBALANCE_RATIO=MAX_IMBALANCE_RATIO, DUP_HAMMING_MAX=DUP_HAMMING_MAX, REMOVE_LEAKAGE_FROM_TRAIN=REMOVE_LEAKAGE_FROM_TRAIN,
              SELFCHECK_MAX_IMAGES=SELFCHECK_MAX_IMAGES, CONF_SHOW=CONF_SHOW,
              RUN_NAME=RUN_NAME, PRESET=PRESET, TRAIN_EXTRA=TRAIN_EXTRA)'''

# Bước 3: giữ khung của BTC, chỉ đổi name=RUN_NAME, ghép TRAIN_EXTRA, và mkdir trước khi chép best.pt.
STEP3_REPLACEMENTS = [
    ('    name="train",\n', '    name=RUN_NAME,\n'),
    ('if TRAIN_ARGS["imgsz"] != IMGSZ:',
     'TRAIN_ARGS.update(TRAIN_EXTRA)  # preset + thay đổi của lượt (ô tham số)\n'
     'print("TRAIN_ARGS thêm:", TRAIN_EXTRA)\n'
     'if TRAIN_ARGS["imgsz"] != IMGSZ:'),
]

COLLECT_MD = """## Bước 7 (đội 71). Gom output của lượt chạy
Sao chép ảnh lỗi Bước 5, ảnh minh họa nhãn, biểu đồ huấn luyện, `results.csv` vào `DRIVE_OUTPUT_DIR` và in **dòng cho `plan/experiment_log.md`**.
Kaggle: bấm *Save Version* (Save & Run All) để thư mục `/kaggle/working/{RUN_NAME}` nằm trong Output và tải về được."""

COLLECT_CELL = r'''import csv as _csv

OUT = Path(DRIVE_OUTPUT_DIR)
OUT.mkdir(parents=True, exist_ok=True)
copied = []
for src_dir, dst_name in ((WORK / "report", "report"), (Path(TRAIN_ARGS["project"]) / RUN_NAME, "train")):
    if src_dir.exists():
        for p in src_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".csv", ".yaml", ".json"}:
                dst = OUT / dst_name / p.relative_to(src_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dst)
                copied.append(dst)
print(f"Đã gom {len(copied)} tệp vào {OUT}")

# Dòng cho bảng "Các lượt huấn luyện" trong plan/experiment_log.md
ap = VAL_RESULT["per_class_ap"]
ap_items = sorted(ap.items(), key=lambda kv: kv[1]) if isinstance(ap, dict) else sorted(zip(CLASSES, ap), key=lambda kv: kv[1])
weak = ", ".join(f"{k} {100 * v:.1f}" for k, v in ap_items[:3])
epochs_done = "?"
res_csv = Path(TRAIN_ARGS["project"]) / RUN_NAME / "results.csv"
if res_csv.exists():
    with open(res_csv) as f:
        epochs_done = sum(1 for _ in _csv.DictReader(f))
line = (f"| {RUN_NAME.split('_')[0]} | | {DATASET_PATH.rstrip('/').split('/')[-1]} | {PRESET} {TRAIN_EXTRA or ''} | "
        f"{epochs_done}/{EPOCHS} | {VAL_RESULT['score']:.2f} | {weak} | | | | |")
(OUT / "experiment_log_line.md").write_text(line + "\n", encoding="utf-8")
print("Dòng cho experiment_log.md:\n" + line)
print("\nTrong thư mục lượt chạy:", sorted(p.name for p in OUT.iterdir()))'''


def src_lines(text):
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]
    i_params = [i for i, c in enumerate(cells) if "".join(c["source"]).startswith(PARAMS_MARK)]
    i_step3 = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and STEP3_MARK in "".join(c["source"])]
    assert len(i_params) == 1 and len(i_step3) == 1, "Không tìm thấy đúng ô tham số / Bước 3: notebook BTC đã đổi cấu trúc"

    cells[i_params[0]]["source"] = src_lines(PARAMS_CELL)

    step3 = "".join(cells[i_step3[0]]["source"])
    for old, new in STEP3_REPLACEMENTS:
        assert step3.count(old) == 1, f"Bước 3 của BTC đã đổi, không tìm thấy: {old!r}"
        step3 = step3.replace(old, new)
    cells[i_step3[0]]["source"] = src_lines(step3)

    cells.append({"cell_type": "markdown", "metadata": {}, "source": src_lines(COLLECT_MD)})
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src_lines(COLLECT_CELL)})

    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    meta = copy.deepcopy(nb.get("metadata", {}))
    meta.setdefault("colab", {})["name"] = DST.name
    nb["metadata"] = meta

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Đã ghi {DST} ({len(cells)} ô; sửa ô {i_params[0]} và {i_step3[0]}, thêm 2 ô cuối)")


if __name__ == "__main__":
    main()
