# Huấn luyện, ablation và chiến lược nộp bài

Người phụ trách: **Quang Hưng** (chạy), **Lộc** (quyết định, nộp bài).

## 1. Ràng buộc không được phá

- Notebook của đội: [`notebooks/train_t66.ipynb`](../notebooks/train_t66.ipynb), sinh bằng [`tools/make_team_notebook.py`](../tools/make_team_notebook.py) từ notebook BTC [`docs/train_and_export.ipynb`](../docs/train_and_export.ipynb). Chỉ **ô tham số**, **Bước 3** và **Bước 7** (thêm) khác bản BTC; Bước 4 (xuất ONNX, kiểm tra, đóng gói) **giữ nguyên tuyệt đối** (script tự kiểm tra). Sửa code ở script rồi chạy lại, không sửa tay file `.ipynb`.
- Mỗi lượt chỉ sửa ô tham số: `RUN_NAME`, `DATASET_PATH`, `PRESET` (`smoke` / `ablation` / `final`), `EXTRA_TRAIN_ARGS`; khi cần: `EPOCHS_OVERRIDE`, `INIT_WEIGHTS` (fine-tune từ `best.pt` cũ), `RESUME`.
- `TEAM_ID = 71`; `TRAIN_IMGSZ = 640`; `MODEL_SIZE = "n"` (có `assert`).
- Kaggle: bật GPU và **Internet** (Bước 0 cài đúng phiên bản ghim; `onnxruntime==1.32.0` cần Python ≥ 3.11 nên không chạy được trên máy Python 3.10). Output tự nằm trong `/kaggle/working/{RUN_NAME}`; Colab ghi vào Drive `MyDrive/t66/runs/{RUN_NAME}`.
- Mỗi lượt chạy lưu: `best.pt`, `last.pt` (cập nhật mỗi epoch, để `RESUME`), `submission.zip`, `run_report.json`, `report/` (ảnh lỗi Bước 5), `train/` (biểu đồ), `boundary_iou.json`, `experiment_log_line.md`.

## 2. Ngân sách thời gian huấn luyện

Notebook ước lượng thô T4 ≈ **22 ảnh/giây** cho cỡ n (chưa đo thật; đo lại ở epoch đầu tiên và cập nhật bảng).

| Loại lượt | Ảnh train | Epoch | Ảnh × epoch | Thời gian ước tính (T4) |
|---|---|---|---|---|
| Ablation nhanh | ~3.000 | 30 | 90k | ~1 giờ 10 phút |
| Ablation đầy đủ | ~6.000 | 30 | 180k | ~2 giờ 15 phút |
| Lượt dài (cuối) | ~8.000 | 60 | 480k | ~6 giờ |
| Teacher `yolo26l-seg` @ 1024 | ~8.000 | 40 | 320k | chậm hơn 4–6 lần → cần GPU mạnh hoặc giảm còn `m` / 25 epoch |

Hệ quả:

- Lượt dài phải **khởi chạy trước 02:00 28/11** để xong trước 08:00.
- Kích thước train cuối nên giữ khoảng **6.000–9.000 ảnh**; nhiều hơn mà không đủ epoch thường không tốt hơn.
- `CACHE = "ram"` nếu máy đủ RAM (Kaggle 29 GB thường đủ cho 8k ảnh 1280px ở 640) để giảm nghẽn dataloader; `WORKERS = 4` trên Kaggle.
- Ai có GPU mạnh hơn (Colab Pro A100/L4, GPU thuê) → báo Lộc ngay; dành cho teacher và lượt dài.

## 3. Thiết kế ablation

Nguyên tắc: **mỗi lần chỉ đổi một yếu tố**, cùng gold val, cùng ngân sách (ablation nhanh), so sánh **mAP gold val** và **AP từng lớp**. Thay đổi được giữ nếu tăng ≥ 0,5 điểm (100 × mAP) hoặc tăng rõ một lớp yếu mà không làm tụt lớp khác.

| ID | Thay đổi | So với | Câu hỏi trả lời |
|---|---|---|---|
| A0 | Notebook mặc định + Vistas 3k | — | Mốc khởi điểm |
| A1 | + IDD | A0 | IDD có giúp miền Việt Nam không? |
| A2 | + COCO đường phố (đã gán nháp Sign) | tốt nhất A0/A1 | COCO có làm nhiễu không? |
| A3 | Chọn ảnh theo điểm ưu tiên thay vì ngẫu nhiên | tốt nhất | Chọn lọc có tốt hơn số lượng? |
| A4 | + ảnh Việt Nam nhãn nháp tin cậy | tốt nhất | Nhãn nháp có giúp không? |
| A5 | + ảnh Việt Nam đã duyệt (×2) | A4 | Giá trị của người duyệt |
| A6 | + ảnh nền khó (200 ảnh) | tốt nhất | Giảm báo nhầm? |
| A7 | `copy_paste=0.2` | tốt nhất | Augmentation cho lớp hiếm |
| A8 | `close_mosaic=10`, `hsv_v=0.5` | tốt nhất | Augmentation ánh sáng (đêm) |

A0–A2 chạy song song trên 3 tài khoản Kaggle ngay khi có dữ liệu. A7–A8 chỉ chạy nếu còn GPU rảnh; không ưu tiên hơn việc làm dữ liệu.

## 4. Công thức huấn luyện

**Baseline (A0)**: giữ nguyên `TRAIN_ARGS` mặc định của notebook (`EPOCHS` theo bảng §2, `BATCH=16`, `PATIENCE=20`).

**Ứng viên cho lượt dài** (chỉ dùng phần đã được ablation xác nhận):

```python
TRAIN_ARGS.update(
    epochs=60, patience=30,
    close_mosaic=10,
    copy_paste=0.2,        # nếu A7 tốt
    hsv_v=0.5,             # nếu A8 tốt
    cos_lr=True,
)
```

Ba lượt dài song song (khởi chạy 01:30–02:00):

| Lượt | Dữ liệu | Công thức | Mục đích |
|---|---|---|---|
| F1 | `merged_final` | Tốt nhất theo ablation | Ứng viên chính |
| F2 | `merged_final` | Như F1, epoch 80 nếu đủ thời gian | Kiểm tra lợi ích huấn luyện lâu hơn |
| F3 | `merged_final_vnheavy` (ảnh Việt Nam ×3) | Như F1 | Phương án tối ưu miền Việt Nam (đa dạng hóa 2 bài chọn) |

## 5. Phân tích lỗi sau mỗi lượt (≤ 20 phút)

Dùng output Bước 5 của notebook:

1. Bảng AP từng lớp: 3 lớp yếu nhất → nhiệm vụ cho vòng dữ liệu sau.
2. Ma trận nhầm lẫn: cặp nhầm nhiều nhất (thường Motorcycle↔Motorcyclist, Bicycle↔Cyclist, Truck↔Other vehicle, Car↔Truck) → kiểm tra lại quy ước và nhãn của cặp đó.
3. 6 ảnh lỗi nhất: phân loại nguyên nhân: **nhãn sai** (sửa dữ liệu) / **thiếu dữ liệu** (thu thêm) / **giới hạn mô hình** (vật quá nhỏ, chấp nhận).
4. Ghi 3 dòng kết luận vào `experiment_log.md` và **lưu ảnh** cho slide.

## 6. Chiến lược 20 lượt nộp

| Lượt | Khung giờ | Nội dung |
|---|---|---|
| #1 | ~15:00 27/11 | Baseline A0: xác nhận pipeline và mốc điểm |
| #2–#3 | 19:30–21:00 | Tốt nhất A1/A2: kiểm tra gold val có tương quan với leaderboard |
| #4–#6 | 21:00–01:00 | Tốt nhất A3–A6 |
| #7 | ~01:30 | `merged_final` với ablation nhanh: kiểm tra trước khi đặt cược vào lượt dài |
| #8–#14 | 07:00–09:30 | F1, F2, F3 và checkpoint trung gian của chúng |
| #15–#17 | 09:30–10:00 | Dự phòng / bản sửa lỗi |
| #18–#20 | — | Không dùng nếu không cần (bài lỗi định dạng không bị tính lượt) |

Quy tắc:

- **Trước mỗi lần nộp**: đã chạy Bước 4–5, có điểm gold val, đã ghi vào `experiment_log.md`.
- Chỉ nộp lượt **có khả năng** vượt bài tốt nhất hiện có trên gold val, hoặc để trả lời một câu hỏi cụ thể.
- Kiểm tra **tương quan gold val ↔ public**: sau #3, nếu xếp hạng giữa các bài trên gold val ngược với public → xem lại gold val (có thể quá nhỏ hoặc lệch miền).
- Hàng đợi chỉ nhận một bài mỗi lần → nộp ngay khi có zip, không gom.

## 7. Chọn 2 bài cuối (G5, trước 10:15)

1. **Bài A**: điểm gold val cao nhất trong các bài có điểm public thuộc nhóm đầu.
2. **Bài B**: điểm public cao nhất, **nếu khác A**. Nếu A cũng là public cao nhất → chọn bài tốt thứ hai đến từ **công thức dữ liệu khác** (ví dụ F3 nặng ảnh Việt Nam) để đa dạng hóa rủi ro trên tập private.
3. Không chọn bài chỉ hơn ≤ 0,3 điểm public mà kém rõ trên gold val (dễ là nhiễu của tập public).
4. Ghi lý do chọn vào `experiment_log.md`; lưu đủ hồ sơ tái lập cho cả 2 bài.
