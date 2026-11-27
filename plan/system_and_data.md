# Hệ thống và dữ liệu của đội: đang có gì, train được chưa

Tài liệu giải thích (1) dữ liệu nhóm đang có trên Drive, (2) hệ thống từ dữ liệu thô tới bài nộp, (3) các bước để chạy lượt train đầu tiên.
Số liệu lấy từ `plan/experiment_log.md`, `plan/assignments.md`, `tools/README.md` (nhánh `main`). Chỗ nào **chưa xác minh được** thì ghi rõ.

---

## 1. Trả lời nhanh: train được chưa?

**Được, cho lượt baseline (A0/A1).** Dữ liệu train đã đủ (Vistas + IDD), nhưng **chưa đưa thẳng các zip vào notebook được**. Còn 3 bước:

1. Giải nén từng zip vào một thư mục riêng.
2. Gộp bằng `tools/build_dataset.py`: train = Vistas (+ IDD), val = `gold_val_v0`. Kết quả là **một** `merged_v1.zip`.
3. Đưa `merged_v1.zip` lên Kaggle Datasets, đặt `DATASET_PATH` trong `notebooks/train_t66.ipynb`, chạy.

Lý do không dùng thẳng: notebook chỉ nhận **một** đường dẫn dữ liệu. Nếu đưa một zip chưa chia train/val (như `vistas_yolo.zip`), notebook tự cắt ngẫu nhiên 20% làm val. Khi đó val là ảnh Vistas nước ngoài, không phải ảnh Việt Nam, nên điểm val không nói lên điểm thật.

Hai điều cần nhớ:

- `gold_val_v0` chỉ có ~60–80 ảnh, nên AP của các lớp ít ví dụ (Bus, Other vehicle, Cyclist...) sẽ **dao động mạnh**. Dùng được để so sánh baseline, nhưng chưa nên dựa vào nó để chốt quyết định cuối. Các quyết định quan trọng chờ `gold_val_v1` (~150 ảnh).
- `gold_val_candidates_223.zip` **tuyệt đối không đưa vào train**: đó là ảnh dành cho gold val. Nếu lọt vào train, điểm val sẽ cao ảo.

---

## 2. Giải thích từng file dữ liệu

### Bảng tổng quan

| File | Là gì | Ảnh nước nào | Vai trò | Số liệu đã biết |
|---|---|---|---|---|
| `vistas_yolo.zip` (đã mở, 1,06 GB) | Mapillary Vistas v2 đã chuyển sang YOLO-seg 11 lớp: **gộp chung** 1.722 ảnh split training + 1.877 ảnh split validation của Vistas trong một thư mục | Toàn thế giới | **Train** (lượt thử: cắt 250 ảnh làm val tạm) | 3.599 ảnh, 51.220 instance, 49 ảnh nền; không lưu được ảnh nào thuộc split nào |
| `idd_yolo_train.zip` (đã mở, 0,93 GB) | India Driving Dataset, split train, đã chuyển | Ấn Độ (giao thông hỗn hợp, gần Việt Nam nhất) | **Train** (lượt thử: cắt 250 ảnh làm val tạm) | 5.006 ảnh, 54.367 instance, QA **ĐẠT**; 152 ảnh nền; rất ít Bicycle/Cyclist/Traffic light (~220 mỗi lớp) |
| `gold_val/gold_val_v0.zip` | Ảnh giao thông **Việt Nam** do đội gán nhãn và duyệt chéo 100% | Việt Nam | **Val** (thước đo), không bao giờ train | Kế hoạch 60–80 ảnh; chưa ghi sha256/số liệu vào log |
| `gold_val/gold_val_candidates_223.zip` | 223 ảnh Việt Nam **ứng viên** cho gold val bản sau | Việt Nam | Nguồn để làm `gold_val_v1` | Không có trong tài liệu repo: **cần hỏi Bình Anh** đã gán nhãn/duyệt chưa |

### "converted" nghĩa là gì

Ảnh và nhãn gốc của mỗi public dataset có định dạng riêng (Vistas dùng ảnh PNG instance, IDD dùng polygon JSON...). Các script `tools/convert_*.py`
chuyển tất cả về **cùng một chuẩn** của cuộc thi, dùng chung bộ quy tắc trong `tools/common.py`:

- Định dạng YOLO-seg: mỗi ảnh có một file `.txt`, mỗi dòng là `id_lớp x1 y1 x2 y2 ...` (đa giác, tọa độ 0–1). File rỗng nghĩa là ảnh nền.
- 11 lớp đúng thứ tự cuộc thi; nhãn gốc được ánh xạ theo `plan/class_mapping.md`.
- **Người + xe máy/xe đạp được ghép thành một instance** Motorcyclist/Cyclist (đúng guideline).
  Ảnh có người lái không ghép được với xe bị **loại bỏ**: IDD mất 1.987 ảnh (28%) vì lý do này.
- Mask không chồng nhau, bỏ vật quá nhỏ (cạnh ngắn < 1,25% cạnh dài ảnh), ảnh resize cạnh dài 1280.
- Tên file có tiền tố nguồn (`vistas_...`, `idd_...`) để gộp không bị trùng tên.

Bên trong hai zip (đã mở bằng `tools/inspect_data.py`):

```
vistas_yolo.zip                          idd_yolo_train.zip
└── vistas_yolo/                         └── idd/
    ├── images/   3.599 vistas_*.jpg         ├── images/   5.006 idd_*.jpg
    ├── labels/   3.599 .txt                 ├── labels/   5.006 .txt
    ├── stats_train.json, stats_val.json     ├── stats.json, rejected.csv
    └── qa/  (report, sample.csv, 70 ảnh     └── qa_report.json
              preview: KHÔNG phải dữ liệu)
```

Không zip nào có sẵn `images/{train,val}`: mỗi zip là **một nguồn**, phải chia hoặc gộp trước khi đưa vào notebook.

### Val của đội phải là ảnh Việt Nam

Bộ test ẩn là ảnh giao thông Việt Nam, nên val chuẩn là **gold val**. Khi gold val chưa sẵn sàng (phương án dự phòng trong `milestones.md` M3),
tạm cắt một phần Vistas + IDD làm val để chạy thử (mục 4a). Điểm trên val tạm chỉ dùng để **so sánh tương đối** giữa các lượt, và phải ghi rõ vào log.

### Gold val: thước đo của đội

- Là thứ quyết định chọn `best.pt` khi train, và là điểm ta so sánh giữa các lượt (`plan/training_submission.md` §3).
- Phải giống bộ test nhất: ảnh Việt Nam, có đêm/mưa/đông đúc, mỗi lớp ≥ 15 instance (`labeling_sop.md`).
- Không được trùng hoặc gần trùng với ảnh train (`build_dataset.py` tự loại ảnh train gần giống val với pHash ≤ 4; notebook kiểm tra thêm một lần).
- `v0` (~60–80 ảnh) để chạy sớm. `v1` (~150 ảnh) sẽ **khóa sau G2** (21:00 27/11): từ đó không sửa nữa để điểm giữa các lượt so sánh được.

---

## 3. Hệ thống: dữ liệu đi từ đâu tới bài nộp

```
 Public dataset gốc        Ảnh Việt Nam (video)
 (Vistas, IDD, COCO)       extract_frames → dedup
        │                           │
        ▼                           ▼
 convert_*.py              pseudo_label.py (mô hình lớn gán nháp)
 → converted/<nguồn>/              │
        │                           ▼
        │                  CVAT: người duyệt/sửa (labeling_sop.md)
        │                  → vn_reviewed/, gold_val/
        ▼                           │
 qa_check.py  ◄─────────────────────┘   cổng chất lượng: chỉ PASS mới được dùng
        │
        ▼
 build_dataset.py + recipe.yaml  →  merged_vN.zip (images/{train,val}) + manifest (sha256)
        │
        ▼  upload Kaggle Datasets (Private)
 notebooks/train_t66.ipynb  (dựng từ notebook BTC)
   Bước 0  cài đúng phiên bản ghim, GPU, seed 71
   Bước 1  nạp dữ liệu, kiểm tra nhãn, tạo data.yaml
   Bước 2  báo cáo dữ liệu: instance/lớp, vật nhỏ, ảnh trùng, rò rỉ train/val
   Bước 3  huấn luyện YOLO26n-seg @640 (preset smoke / ablation / final)   ← đội sửa
   Bước 4  xuất ONNX + kiểm tra bằng mã hệ thống chấm + submission.zip    ← KHÔNG sửa
   Bước 5  chạy thử bộ chấm trên val: mask mAP, AP từng lớp, ảnh lỗi
   Bước 6  run_report.json (bản kê tái lập)
   Bước 7  độ sát biên mask theo lớp, gom output, dòng cho experiment_log  ← đội thêm
        │
        ▼
 submission.zip (đúng 1 file model.onnx) → website (20 lượt) → leaderboard
        │
        ▼
 Phân tích lỗi → biết lớp nào yếu → thu thêm / sửa nhãn → vòng sau
```

### Các thành phần

| Thành phần | File | Ai dùng | Ghi chú |
|---|---|---|---|
| Quy tắc chung | `tools/common.py` | mọi script | 11 lớp, ghép rider, polygon, lọc vật nhỏ |
| Chuyển đổi public | `convert_vistas.py`, `convert_idd.py`, `convert_coco.py`, `import_yolo.py` | Thắng, Long | ghi `stats.json`, `rejected.csv` |
| Ảnh Việt Nam | `extract_frames.py`, `dedup.py`, `pseudo_label.py` | Đạt | gán nháp cần GPU |
| Kiểm tra chất lượng | `qa_check.py`, `label_audit.py`, `visualize.py`, `viewer/` | người phụ trách nguồn, Bình Anh | quy trình: `plan/qa_process.md` |
| **Xem nhanh một bộ dữ liệu** | `tools/inspect_data.py` | ai cũng dùng | đọc thẳng trong zip: cấu trúc, đã có nhãn chưa, instance/lớp |
| Gộp dataset | `build_dataset.py` + `configs/recipes/*.yaml` | Lộc | chọn ảnh ưu tiên lớp hiếm, nhân bản ảnh VN, loại ảnh gần trùng val |
| Huấn luyện + nộp | `notebooks/train_t66.ipynb` (sinh bởi `tools/make_team_notebook.py`) | Hưng | dựng trên notebook BTC `docs/train_and_export.ipynb` |
| Teacher | `tools/train_teacher.py` | Hưng | `yolo26l/m-seg` @1024 để gán nháp tốt hơn; **không nộp** |

### Hệ thống chấm của cuộc thi (tóm tắt)

- Mô hình cố định: **YOLO26n-seg**, `imgsz = 640`. Đội chỉ được cải thiện **dữ liệu** (và cách train).
- Điểm = 100 × mask mAP@[.5:.95], **trung bình đều 11 lớp**: một lớp gần 0 điểm kéo tụt cả bài.
- Nộp `.zip` chứa đúng `model.onnx` (fp32, ≤ 42 MB), xuất bằng **Bước 4 của notebook BTC**. Bài sai định dạng bị từ chối và không tốn lượt.
- 20 lượt nộp; xếp hạng cuối = 50% điểm private + 50% thuyết trình.

---

## 4a. Lượt THỬ khi chưa có gold val (đã tạo dữ liệu, 27/11)

Trưởng nhóm muốn train thử trước khi có gold val. `tools/split_trial.py` đọc thẳng hai zip, lấy **250 ảnh val từ mỗi nguồn** (cố định theo seed 71),
phần còn lại làm train. Val không phụ thuộc lượt nào, nên A0 và A1 có **cùng 500 ảnh val** và so sánh được với nhau.

```bash
SRC="vistas=E:/Download_thay_o_C/vistas/vistas_yolo.zip idd=E:/Download_thay_o_C/idd/idd_yolo_train.zip"
python tools/split_trial.py --sources $SRC --val-per-source 250 --train vistas --name trial_a0_vistas --out E:/Download_thay_o_C/datasets
python tools/split_trial.py --sources $SRC --val-per-source 250 --train vistas idd --train-max idd=3000 --name trial_a1_vistas_idd --out E:/Download_thay_o_C/datasets
```

| Bộ (trong `E:\Download_thay_o_C\datasets\`) | Train | Val | Zip | Ước tính 30 epoch trên T4 | Chạy ở |
|---|---|---|---|---|---|
| `trial_a0_vistas.zip` | 3.349 ảnh Vistas | 250 Vistas + 250 IDD | 1,07 GB | ~1 giờ 20 phút | Kaggle |
| `trial_a1_vistas_idd.zip` | 3.349 Vistas + 3.000 IDD | 250 Vistas + 250 IDD | 1,63 GB | ~2 giờ 20 phút | Colab (hoặc Kaggle tài khoản 2) |

Val có ≥ 72 instance cho mọi lớp (ít nhất: Cyclist 72, Bicycle 75), đủ để AP từng lớp không quá nhiễu.

Tiếp theo:

1. **Kaggle (A0):** Datasets → New Dataset → tải `trial_a0_vistas.zip` (Private). Mở `notebooks/train_t66.ipynb`, bật GPU T4 + Internet, *Add Input* dataset đó.
   Ô Tham số: `RUN_NAME = "R01_trial_a0_vistas"`, `DATASET_PATH = "/kaggle/input/<tên-dataset>"`, `PRESET = "smoke"` chạy thử trước, rồi `"ablation"`.
2. **Colab (A1):** tải `trial_a1_vistas_idd.zip` lên Google Drive. Mở notebook trên Colab, chọn T4 GPU.
   `RUN_NAME = "R02_trial_a1_vistas_idd"`, `DATASET_PATH = "/content/drive/MyDrive/<thư-mục>/trial_a1_vistas_idd.zip"`.
   Giữ tab mở; bị ngắt thì đặt `RESUME = True` rồi chạy lại từ đầu notebook (`last.pt` đã lưu trên Drive mỗi epoch).
3. Ghi cả hai vào `plan/experiment_log.md` với ghi chú **"val tạm = 250 Vistas + 250 IDD"**.

## 4. Các bước để có lượt train đầu tiên (khi đã có gold val)

### Bước 0: kiểm tra từng zip (2 phút, máy có dữ liệu)

```bash
python tools/inspect_data.py E:/Download_thay_o_C/vistas/vistas_yolo.zip E:/Download_thay_o_C/idd/idd_yolo_train.zip <đường dẫn>/gold_val/*.zip
```

Đọc dòng `=>` cuối mỗi bộ:

| Bộ | Mong đợi | Nếu khác |
|---|---|---|
| `vistas_yolo_*`, `idd_yolo_train` | "Có nhãn đa giác", "Chưa chia train/val" | Có nhãn hộp hoặc lỗi → báo Thắng/Long |
| `gold_val_v0` | Có nhãn; lớp nào < 15 instance thì ghi lại | Thiếu nhiều lớp → AP các lớp đó không đo được |
| `gold_val_candidates_223` | Biết được đã có nhãn hay "CHƯA CÓ NHÃN" | Nếu chưa có nhãn: chỉ là ảnh chờ gán, không dùng được |

### Bước 1: giải nén mỗi zip vào một thư mục

```powershell
Expand-Archive E:\Download_thay_o_C\vistas\vistas_yolo.zip  E:\data\converted   # -> E:\data\converted\vistas_yolo\images, labels
Expand-Archive E:\Download_thay_o_C\idd\idd_yolo_train.zip E:\data\converted   # -> E:\data\converted\idd\images, labels
Expand-Archive <đường dẫn>\gold_val_v0.zip                 E:\data\gold_val\gold_val_v0
```

Đường dẫn trong recipe phải trỏ tới thư mục **chứa trực tiếp** `images/` và `labels/`. Nếu zip có thêm một lớp thư mục bên trong, trỏ sâu thêm một cấp.

### Bước 2: recipe cho A0 và A1 (theo `training_submission.md` §3)

Ablation nhanh nên giữ ~3.000–6.000 ảnh train (T4 ≈ 22 ảnh/s: 6.000 ảnh × 30 epoch ≈ 2 giờ 15 phút).

`configs/recipes/a0_vistas.yaml` (mốc khởi điểm):

```yaml
name: a0_vistas
seed: 71
out_dir: E:/data/datasets
sources:
  vistas:   E:/data/converted/vistas_yolo
  gold_val: E:/data/gold_val/gold_val_v0
val:
  source: gold_val
train:
  - {source: vistas, select: priority}   # cả 3.599 ảnh; thêm max_images để giới hạn (priority = ưu tiên ảnh có lớp hiếm)
exclude_near_duplicates_of_val: 4
```

`configs/recipes/a1_vistas_idd.yaml`: giống A0, chỉ thêm IDD (mỗi ablation **chỉ đổi một yếu tố**):

```yaml
name: a1_vistas_idd
# ... sources như A0, thêm:  idd: E:/data/converted/idd
train:
  - {source: vistas, select: priority}
  - {source: idd,          max_images: 3000, select: priority}
```

Khi có gold val, **toàn bộ** Vistas và IDD đều vào train (không cắt val tạm nữa). Lượt dài cuối cùng thêm ảnh Việt Nam đã duyệt (`repeat: 2`) và ảnh nháp tin cậy.

```bash
python tools/build_dataset.py --recipe configs/recipes/a0_vistas.yaml
python tools/qa_check.py --data E:/data/datasets/a0_vistas --render      # profile merged: rò rỉ, mục tiêu lớp
```

Kết quả là `E:/data/datasets/a0_vistas.zip` (đã có `images/{train,val}`) và `a0_vistas.manifest.json`. Bảng cuối lệnh đánh dấu lớp `<-- ít`.

### Bước 3: Kaggle

1. Kaggle → Datasets → New Dataset → tải `a0_vistas.zip` lên (Private) → tên ví dụ `a0-vistas`.
2. Mở `notebooks/train_t66.ipynb` trên Kaggle: Settings → Accelerator **GPU T4**, **Internet on**, *Add Input* dataset vừa tạo.
3. Sửa ô Tham số:

```python
RUN_NAME = "R01_a0_vistas_baseline"
DATASET_PATH = "/kaggle/input/a0-vistas"     # thư mục Kaggle tự giải nén, hoặc đường dẫn tới file .zip
PRESET = "smoke"                              # lần đầu: smoke (vài phút) để chắc pipeline chạy; sau đó "ablation"
```

4. *Save Version → Save & Run All*. Xong thì vào Output: `/kaggle/working/R01_a0_vistas_baseline/` có `submission.zip`, `run_report.json`, `experiment_log_line.md`.
5. Chép dòng trong `experiment_log_line.md` vào `plan/experiment_log.md`, rồi nộp `submission.zip` lên website (lượt #1, xác nhận pipeline).

---

## 5. Thuật ngữ

| Từ | Nghĩa |
|---|---|
| YOLO-seg | Định dạng nhãn: mỗi dòng một đối tượng, `id x1 y1 x2 y2 ...` (đa giác chuẩn hóa 0–1) |
| instance | Một đối tượng cụ thể (một chiếc xe), có mask riêng |
| split train / val | Ảnh để mô hình học / ảnh để chấm thử. Val không được dùng để học |
| gold val | Val do đội tự gán nhãn trên ảnh Việt Nam, duyệt kỹ: thước đo chính |
| rò rỉ (leakage) | Ảnh val (hoặc ảnh gần giống) lọt vào train → điểm val cao ảo |
| recipe | File YAML mô tả lấy bao nhiêu ảnh từ nguồn nào → dựng lại được đúng dataset |
| ablation | Mỗi lượt chỉ đổi một yếu tố để biết yếu tố đó có giúp không |
| teacher | Mô hình lớn hơn (l/m-seg) chỉ dùng để gán nhãn nháp, không nộp |
| mAP@[.5:.95] | Trung bình độ chính xác ở 10 ngưỡng IoU từ 0,50 tới 0,95: mask càng sát biên, điểm càng cao |
