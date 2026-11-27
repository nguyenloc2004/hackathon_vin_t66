"""Sinh notebooks/train_t66.ipynb (notebook của đội 71) từ notebook của BTC: docs/train_and_export.ipynb.

Notebook BTC chứa bộ kiểm tra + bộ chấm của hệ thống (Bước 4-5) và hợp đồng xuất ONNX, nên đội KHÔNG viết lại mà chỉ:
  * thay ô "Tham số của đội"  -> TEAM_ID=71, RUN_NAME, preset smoke/ablation/final, EXTRA_TRAIN_ARGS, thư mục output tự đặt;
  * thay ô Bước 3 (huấn luyện, BTC cho phép sửa tự do) -> preset, fine-tune từ best.pt cũ, resume, lưu last.pt mỗi epoch;
  * thêm Bước 7 ở cuối -> IoU mask của các cặp đúng theo lớp, gom output, dòng cho experiment_log.md.
Bước 0-2 và 4-6 giữ nguyên từng ký tự (script kiểm tra điều này). Khi BTC ra bản notebook mới: thay file trong docs/ rồi chạy lại.

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

INTRO_MD = r'''
# Đội 71 · notebook huấn luyện (dựng từ notebook BTC)

Sinh bởi `tools/make_team_notebook.py` từ `docs/train_and_export.ipynb`. **Chỉ ô Tham số, Bước 3 và Bước 7 khác bản BTC**;
Bước 0–2 và 4–6 (cài đúng phiên bản, kiểm tra dữ liệu, **xuất ONNX + kiểm tra như hệ thống chấm**, chạy thử bộ chấm) giữ nguyên.

Mỗi lượt chạy: sửa `RUN_NAME`, `DATASET_PATH`, `PRESET`, `EXTRA_TRAIN_ARGS` ở ô Tham số → Kaggle *Save Version → Save & Run All* (GPU + Internet).
Kết quả nằm trong `/kaggle/working/{RUN_NAME}` (Colab: Drive `MyDrive/t66/runs/{RUN_NAME}`): `best.pt`, `last.pt`, `submission.zip`, `run_report.json`,
ảnh lỗi, biểu đồ, `experiment_log_line.md`.
'''

PARAMS_CELL = r'''#@title Tham số của đội: điền ở đây, mọi ô sau đọc từ đây
# Đội 71. Mỗi lượt chạy chỉ sửa khối "Lượt chạy". Tên biến của BTC giữ nguyên vì các ô sau dùng chúng.

# ==== Lượt chạy ====
TEAM_ID = 71  #@param {type:"integer"}
RUN_NAME = "R01_merged_v1_baseline"  #@param {type:"string"}
# R{NN}_{dataset}_{ghi-chú} (ablation) hoặc F{N}_{dataset}_{ghi-chú} (lượt dài). Ghi đúng tên này vào plan/experiment_log.md.
DATASET_PATH = "/kaggle/input/merged-v1"  #@param {type:"string"}
# Zip/thư mục do tools/build_dataset.py tạo: images/{train,val} + labels/{train,val}, val = gold val.
PRESET = "ablation"  #@param ["smoke", "ablation", "final"]
EXTRA_TRAIN_ARGS = {
    # Thay đổi của lượt này, ghép đè lên preset. Ablation: mỗi lượt chỉ đổi MỘT nhóm (lý do ở markdown Bước 3).
    # "copy_paste": 0.3, "copy_paste_mode": "flip",   # A7: dán instance lớp hiếm
    # "hsv_v": 0.5,                                    # A8: ánh sáng / đêm (close_mosaic=10 là mặc định sẵn của Ultralytics)
    # "mask_ratio": 2,                                 # mask GT phân giải cao hơn khi train -> biên sát hơn
    # "scale": 0.7,                                    # biến thiên tỉ lệ mạnh hơn
}
EPOCHS_OVERRIDE = None   # vd 80 cho F2; None = theo preset
INIT_WEIGHTS = None      # None = pretrained COCO (MODEL_WEIGHTS); hoặc đường dẫn best.pt của lượt trước để fine-tune tiếp
RESUME = False           # True = chạy tiếp từ last.pt trong thư mục output (khi phiên GPU bị ngắt; giữ nguyên mọi tham số khác)

# ==== Preset (plan/training_submission.md §2-4) ====
PRESETS = {
    "smoke":    dict(epochs=3,  patience=3,  save_period=-1, extra={}),               # thử pipeline Bước 0-6, không để nộp thật
    "ablation": dict(epochs=30, patience=20, save_period=-1, extra={}),               # ngân sách cố định, so sánh công bằng
    "final":    dict(epochs=60, patience=30, save_period=10, extra=dict(cos_lr=True)),  # F1-F3; checkpoint mỗi 10 epoch
}
_P = PRESETS[PRESET]
TRAIN_EXTRA = {**_P["extra"], **EXTRA_TRAIN_ARGS}
SAVE_PERIOD = _P["save_period"]

# ---- Mô hình và huấn luyện (tên biến của BTC) ----
MODEL_SIZE = "n"  # cố định: cuộc thi chỉ nhận YOLO26n-seg
MODEL_WEIGHTS = f"yolo26{MODEL_SIZE}-seg.pt"
EPOCHS = EPOCHS_OVERRIDE or _P["epochs"]
BATCH = 16  #@param {type:"integer"}
# 16 chạy tốt trên T4 với cỡ n; nếu báo hết bộ nhớ GPU (CUDA out of memory) thì giảm còn 8.
PATIENCE = _P["patience"]
TRAIN_IMGSZ = 640  # KHÔNG đổi
IN_KAGGLE = Path("/kaggle/working").exists()
WORKERS = 4 if IN_KAGGLE else 2
CACHE = "ram" if IN_KAGGLE else False   # Kaggle ~29 GB RAM đủ cho ~8-9k ảnh @640; thiếu RAM thì đặt False
PLOTS = True

# ---- Dữ liệu và báo cáo ----
VAL_RATIO = 0.2                  # chỉ dùng khi dữ liệu CHƯA chia sẵn (lượt thật luôn dùng gold val)
MIN_TRAIN_INSTANCES = 400        # plan: mỗi lớp >= 400-1000 instance train
MAX_IMBALANCE_RATIO = 30
DUP_HAMMING_MAX = 2
REMOVE_LEAKAGE_FROM_TRAIN = True  # gold val phải sạch: bỏ khỏi train ảnh trùng/gần giống val

# ---- Tự kiểm tra sau khi xuất ----
SELFCHECK_MAX_IMAGES = 300
CONF_SHOW = 0.25
N_ERROR_IMAGES = 6

# ---- Hệ thống ----
INSTALL_PACKAGES = True
REQUIRE_GPU = True
DRIVE_OUTPUT_DIR = (f"/kaggle/working/{RUN_NAME}" if IN_KAGGLE
                    else f"/content/drive/MyDrive/t66/runs/{RUN_NAME}" if IN_COLAB else str(Path("runs", RUN_NAME).resolve()))
WORK_DIR = "/content/work" if IN_COLAB else "/tmp/t66_work" if IN_KAGGLE else "work"
# Kaggle: làm việc ở /tmp để Output chỉ chứa thư mục lượt chạy, không chứa bản sao dataset.

assert TEAM_ID == 71 and MODEL_SIZE == "n" and TRAIN_IMGSZ == 640 and "imgsz" not in TRAIN_EXTRA, "Không đổi TEAM_ID / cỡ mô hình / imgsz"
print(f"Lượt {RUN_NAME} | preset {PRESET} | {EPOCHS} epoch | patience {PATIENCE} | thêm {TRAIN_EXTRA or '{}'}"
      + (f" | khởi tạo từ {INIT_WEIGHTS}" if INIT_WEIGHTS else "") + (" | RESUME" if RESUME else ""))
print(f"Dữ liệu: {DATASET_PATH}\nOutput: {DRIVE_OUTPUT_DIR}")

PARAMS = dict(TEAM_ID=TEAM_ID, MODEL_SIZE=MODEL_SIZE, MODEL_WEIGHTS=MODEL_WEIGHTS, EPOCHS=EPOCHS, BATCH=BATCH, PATIENCE=PATIENCE,
              TRAIN_IMGSZ=TRAIN_IMGSZ, WORKERS=WORKERS, CACHE=CACHE, VAL_RATIO=VAL_RATIO, MIN_TRAIN_INSTANCES=MIN_TRAIN_INSTANCES,
              MAX_IMBALANCE_RATIO=MAX_IMBALANCE_RATIO, DUP_HAMMING_MAX=DUP_HAMMING_MAX, REMOVE_LEAKAGE_FROM_TRAIN=REMOVE_LEAKAGE_FROM_TRAIN,
              SELFCHECK_MAX_IMAGES=SELFCHECK_MAX_IMAGES, CONF_SHOW=CONF_SHOW,
              RUN_NAME=RUN_NAME, PRESET=PRESET, TRAIN_EXTRA=TRAIN_EXTRA, INIT_WEIGHTS=INIT_WEIGHTS, RESUME=RESUME)'''

STEP3_MD = r'''
## Bước 3. Huấn luyện (đội 71: preset + thay đổi của lượt)

Mặc định Ultralytics đã là công thức tốt cho YOLO26 (optimizer `auto`, mosaic, `close_mosaic=10`, `hsv_v=0.4`, `fliplr=0.5`, EMA). Preset chỉ đổi epoch / patience,
`final` thêm `cos_lr` và lưu checkpoint mỗi 10 epoch. `last.pt` được chép sang thư mục output **sau mỗi epoch** để `RESUME = True` chạy tiếp được khi phiên bị ngắt.

| Thử qua `EXTRA_TRAIN_ARGS` | Vì sao có thể giúp | Rủi ro |
|---|---|---|
| `copy_paste=0.3, copy_paste_mode="flip"` | Có polygon nên dán được instance lớp hiếm (Bus, Other vehicle, Bicycle, Cyclist); điểm là trung bình đều 11 lớp | Vị trí dán phi thực tế |
| `mask_ratio=2` | Mask GT khi train ít bị thu nhỏ (mặc định 4) → biên sát hơn, có ích ở IoU 0,75–0,95 | Chậm hơn, tốn VRAM |
| `hsv_v=0.5` | Đa dạng độ sáng (đêm, ngược sáng) | Ít |
| `scale=0.7` | Xe máy xa/gần khác cỡ nhiều | Vật nhỏ bị thu quá mức |

Không nên: `imgsz` khác 640, `freeze`, `half`, `rect=True`. Fine-tune 2 giai đoạn: lượt 2 đặt `INIT_WEIGHTS` = `best.pt` lượt 1, dữ liệu nặng ảnh Việt Nam,
`EXTRA_TRAIN_ARGS={"lr0": 0.002, "warmup_epochs": 0}`, 15–20 epoch. Ước lượng thời gian là thô (T4); tốc độ thật xem ở epoch đầu.
'''

STEP3_CELL = r'''from ultralytics import YOLO

TRAIN_ARGS = dict(
    data=DATASET["data_yaml"],
    imgsz=TRAIN_IMGSZ,      # giữ 640
    epochs=EPOCHS,
    batch=BATCH,
    seed=SEED,              # seed của đội
    deterministic=True,
    device=DEVICE,
    workers=WORKERS,
    patience=PATIENCE,
    cache=CACHE,
    plots=PLOTS,
    project=str(WORK / "runs"),
    name=RUN_NAME,
    exist_ok=True,
    save_period=SAVE_PERIOD,
)
TRAIN_ARGS.update(TRAIN_EXTRA)  # preset + thay đổi của lượt (ô Tham số)
print("Thêm so với mặc định:", TRAIN_EXTRA or "không")
if TRAIN_ARGS["imgsz"] != IMGSZ:
    print(f"CẢNH BÁO: imgsz huấn luyện = {TRAIN_ARGS['imgsz']} khác {IMGSZ}. Hệ thống chấm chạy mô hình ở {IMGSZ}, điểm có thể thấp hơn.")
n_train, n_val = DATASET["counts"]["train"]["images"], DATASET["counts"]["val"]["images"]
print(f"Ước lượng thô (chưa đo): {n_train} ảnh train x {EPOCHS} epoch, cỡ {MODEL_SIZE} -> khoảng {fmt_minutes(estimate_train_minutes(n_train, n_val, EPOCHS, MODEL_SIZE))} trên T4. "
      "Nếu epoch đầu chậm hơn nhiều, giảm EPOCHS.")

OUT_DIR = Path(DRIVE_OUTPUT_DIR)
if str(OUT_DIR).startswith("/content/drive"):
    mount_drive_if_needed(DRIVE_OUTPUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_last(trainer):
    """Chép last.pt sang thư mục output sau mỗi epoch (~6 MB) để RESUME được khi phiên GPU bị ngắt."""
    if Path(trainer.last).exists():
        shutil.copyfile(trainer.last, OUT_DIR / "last.pt")


import csv, time

seed_everything(SEED)
t_start = time.time()
if RESUME:
    resume_from = OUT_DIR / "last.pt"
    if not resume_from.exists():
        raise NotebookError(f"RESUME = True nhưng không có {resume_from}", "TRAIN")
    run_dir = WORK / "runs" / RUN_NAME / "weights"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(resume_from, run_dir / "last.pt")
    model = YOLO(str(run_dir / "last.pt"))
    model.add_callback("on_fit_epoch_end", _save_last)
    model.train(resume=True)
else:
    model = YOLO(INIT_WEIGHTS or MODEL_WEIGHTS)  # .pt: pretrained (tự tải) hoặc best.pt của lượt trước
    if model.task != "segment":
        raise NotebookError(f"{INIT_WEIGHTS or MODEL_WEIGHTS} không phải mô hình segmentation", "TRAIN")
    model.add_callback("on_fit_epoch_end", _save_last)
    model.train(**TRAIN_ARGS)
BEST_WEIGHTS = Path(model.trainer.best)
_best = YOLO(str(BEST_WEIGHTS))
trained_names = [_best.names[i] for i in sorted(_best.names)]
if trained_names != list(CLASSES):
    raise NotebookError(f"Tên lớp của mô hình {trained_names} khác danh sách lớp của cuộc thi {list(CLASSES)}. Hãy huấn luyện bằng data.yaml do Bước 1 tạo.", "TRAIN")
if sum(p.numel() for p in _best.model.parameters()) >= 5_000_000:
    raise NotebookError("Mô hình có quá nhiều tham số: không phải YOLO26n-seg", "TRAIN")

with open(Path(model.trainer.save_dir) / "results.csv", newline="") as f:
    _res = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
_col = "metrics/mAP50-95(M)"
_best_row = max(_res, key=lambda r: float(r[_col])) if _res else {}
TRAIN_INFO = {"best_weights": str(BEST_WEIGHTS), "best_sha256": sha256_file(BEST_WEIGHTS), "best_tensors_sha256": weights_fingerprint(BEST_WEIGHTS),
              "train_args": {k: v for k, v in TRAIN_ARGS.items() if k != "project"}, "device": DEVICE, "gpu": GPU_NAME,
              "run_name": RUN_NAME, "preset": PRESET, "init_weights": INIT_WEIGHTS or MODEL_WEIGHTS, "resume": RESUME,
              "epochs_done": len(_res), "best_epoch": int(float(_best_row.get("epoch", 0))),
              "best_val_mask_map50_95_ultralytics": float(_best_row.get(_col, 0.0)), "minutes": round((time.time() - t_start) / 60, 1)}
print(f"Xong {TRAIN_INFO['epochs_done']}/{EPOCHS} epoch trong {TRAIN_INFO['minutes']} phút; epoch tốt nhất {TRAIN_INFO['best_epoch']} "
      f"(mask mAP50-95 val theo Ultralytics = {100 * TRAIN_INFO['best_val_mask_map50_95_ultralytics']:.2f})")
print("Trọng số tốt nhất:", BEST_WEIGHTS)
shutil.copyfile(BEST_WEIGHTS, OUT_DIR / "best.pt")
print(f"Đã sao chép best.pt sang {OUT_DIR}")
if (Path(model.trainer.save_dir) / "results.png").exists():
    show_image(Path(model.trainer.save_dir) / "results.png")'''

STEP7_MD = r'''
## Bước 7 (đội 71). Độ sát biên mask theo lớp, gom output, dòng nhật ký

Ghép mỗi nhãn val với dự đoán cùng lớp (conf ≥ `CONF_SHOW`, IoU mask ≥ 0,5) rồi tính **IoU trung bình của các cặp đúng** theo lớp.
Lớp có AP thấp nhưng IoU cặp đúng cao → thiếu dữ liệu / bỏ sót; IoU cặp đúng thấp (< 0,8) → **biên mask lỏng**: vẽ polygon sát hơn hoặc thử `mask_ratio=2`.
Sau đó gom ảnh lỗi Bước 5, biểu đồ huấn luyện vào thư mục output và in dòng cho `plan/experiment_log.md`.
'''

STEP7_CELL = r'''import csv as _csv
import cv2

VAL_IMG_DIR = WORK / "dataset" / "images" / "val"
VAL_LBL_DIR = WORK / "dataset" / "labels" / "val"
_m = YOLO(str(BEST_WEIGHTS), task="segment")
_tp, _iou_sum, _miss = Counter(), Counter(), Counter()
for img_path in VAL_IMAGES:
    r = _m.predict(str(img_path), imgsz=IMGSZ, conf=CONF_SHOW, max_det=MAX_DET, retina_masks=True, device=DEVICE, verbose=False)[0]
    H, W = r.orig_shape
    s = 640 / max(H, W)
    w2, h2 = max(1, round(W * s)), max(1, round(H * s))
    gts = []
    lp = VAL_LBL_DIR / f"{Path(img_path).stem}.txt"
    for line in (lp.read_text(encoding="utf-8").splitlines() if lp.exists() else []):
        p = line.split()
        if len(p) >= 7:
            m = np.zeros((h2, w2), np.uint8)
            cv2.fillPoly(m, [(np.asarray(p[1:], float).reshape(-1, 2) * (w2, h2)).astype(np.int32)], 1)
            gts.append((int(p[0]), m.reshape(-1).astype(np.float32)))
    if r.masks is None or not len(r.boxes) or not gts:
        for c, _ in gts:
            _miss[c] += 1
        continue
    pc, pf = r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()
    pm = np.stack([cv2.resize(x.astype(np.uint8), (w2, h2), interpolation=cv2.INTER_NEAREST).reshape(-1) for x in r.masks.data.cpu().numpy()]).astype(np.float32)
    gm = np.stack([g for _, g in gts])
    inter = gm @ pm.T
    iou = inter / np.maximum(gm.sum(1)[:, None] + pm.sum(1)[None, :] - inter, 1)
    used = np.zeros(len(gts), bool)
    for j in np.argsort(-pf):
        cand = [i for i in range(len(gts)) if not used[i] and gts[i][0] == pc[j] and iou[i, j] >= 0.5]
        if cand:
            i = max(cand, key=lambda i: iou[i, j])
            used[i] = True
            _tp[gts[i][0]] += 1
            _iou_sum[gts[i][0]] += iou[i, j]
    for i in np.where(~used)[0]:
        _miss[gts[i][0]] += 1

print(f"{'lớp':<14}{'AP':>7}{'đúng':>6}{'không khớp':>11}{'IoU TB cặp đúng':>17}")
BOUNDARY = {}
for i, n in enumerate(CLASSES):
    miou = _iou_sum[i] / _tp[i] if _tp[i] else float("nan")
    ap = VAL_RESULT["per_class_ap"].get(n)
    BOUNDARY[n] = {"tp": _tp[i], "unmatched_gt": _miss[i], "mean_iou_tp": None if np.isnan(miou) else round(float(miou), 3)}
    print(f"{n:<14}{'' if ap is None else f'{ap:.3f}':>7}{_tp[i]:>6}{_miss[i]:>11}{miou:>17.3f}" + ("  <-- biên lỏng" if _tp[i] and miou < 0.8 else ""))

OUT_DIR = Path(DRIVE_OUTPUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)
for src, dst_name in ((WORK / "report", "report"), (Path(model.trainer.save_dir), "train")):
    if src.exists():
        for p in src.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".csv", ".yaml", ".json"}:
                d = OUT_DIR / dst_name / p.relative_to(src)
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, d)
(OUT_DIR / "boundary_iou.json").write_text(json.dumps(BOUNDARY, ensure_ascii=False, indent=1), encoding="utf-8")

ap = VAL_RESULT["per_class_ap"]
weak = ", ".join(f"{k} {100 * v:.1f}" for k, v in sorted(ap.items(), key=lambda kv: kv[1])[:3])
change = ", ".join(f"{k}={v}" for k, v in TRAIN_EXTRA.items()) or "mặc định"
line = (f"| {RUN_NAME.split('_')[0]} | | {Path(str(DATASET_PATH)).name} | {PRESET}: {change} | "
        f"{TRAIN_INFO['epochs_done']}/{EPOCHS} (best {TRAIN_INFO['best_epoch']}) | {VAL_RESULT['score']:.2f} | {weak} | | | | |")
(OUT_DIR / "experiment_log_line.md").write_text(line + "\n", encoding="utf-8")
print("\nDòng cho plan/experiment_log.md:\n" + line)
print(f"\nThư mục lượt chạy {OUT_DIR}:", sorted(p.name for p in OUT_DIR.iterdir()))'''


def src_lines(text):
    text = text.strip("\n")
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": src_lines(text)}


def code_cell(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src_lines(text)}


def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    orig = copy.deepcopy(nb["cells"])
    cells = nb["cells"]
    i_params = [i for i, c in enumerate(cells) if "".join(c["source"]).startswith(PARAMS_MARK)]
    i_step3 = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and STEP3_MARK in "".join(c["source"])]
    assert len(i_params) == 1 and len(i_step3) == 1, "Không tìm thấy đúng ô tham số / Bước 3: notebook BTC đã đổi cấu trúc"
    i_step3_md = i_step3[0] - 1
    assert cells[i_step3_md]["cell_type"] == "markdown" and "Bước 3" in "".join(cells[i_step3_md]["source"])

    cells[i_params[0]] = code_cell(PARAMS_CELL)
    cells[i_step3_md] = md_cell(STEP3_MD)
    cells[i_step3[0]] = code_cell(STEP3_CELL)
    changed = {i_params[0], i_step3_md, i_step3[0]}
    for i, (a, b) in enumerate(zip(orig, cells)):
        assert i in changed or a["source"] == b["source"], f"ô {i} bị đổi ngoài ý muốn"

    cells.insert(0, md_cell(INTRO_MD))
    cells += [md_cell(STEP7_MD), code_cell(STEP7_CELL)]
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    nb.setdefault("metadata", {}).setdefault("colab", {})["name"] = DST.name

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Đã ghi {DST}: {len(cells)} ô; thay ô tham số + Bước 3 (markdown, code), thêm ô giới thiệu đầu và Bước 7 cuối; các ô khác giữ nguyên.")


if __name__ == "__main__":
    main()
