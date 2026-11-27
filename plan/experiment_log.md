# Nhật ký dữ liệu, huấn luyện và nộp bài

`TEAM_ID = 71` · Còn lại: **20 / 20** lượt nộp · Đóng nộp **10:30 28/11** · Quy ước tên lượt chạy: `R{NN}_{dataset}_{ghi-chú}`

## Cổng quyết định

| Cổng | Hạn | Trạng thái | Quyết định / ghi chú |
|---|---|---|---|
| G0 Pipeline | 13:00 27/11 | ☐ | |
| G1 Baseline | 15:30 27/11 | ☐ | |
| G2 Công thức dữ liệu | 21:00 27/11 | ☐ | |
| G3 Đóng băng dữ liệu | 01:30 28/11 | ☐ | |
| G4 Nộp cuối | 09:30 28/11 | ☐ | |
| G5 Chọn 2 bài | 10:15 28/11 | ☐ | |

## Cổng chất lượng (`qa_check.py`, [`qa_process.md`](qa_process.md))

Chép dòng `Dòng cho experiment_log.md` mà `qa_check.py` in ra; chỉ `PASS` mới được vào recipe / lên Kaggle. Cột miễn trừ: liệt kê mã đã miễn trừ và lý do.

| Dữ liệu | Người phụ trách | Số ảnh | Lỗi / cảnh báo | Mẫu người xem (bad / đã chấm) | Kết luận (commit) | Miễn trừ |
|---|---|---|---|---|---|---|
| Vistas v2 | Thắng | | | | | |
| IDD | Long | 5.006 (54.367 inst) | `qa_check` bản cũ: 41 chồng > 20%, 0 lỗi format · bản mới: 0 lỗi / 1.177 cảnh báo (chồng nặng là `multipart_overlap`) | Xem tay: đạt (30 mẫu + 20 ảnh rider) | ĐẠT (chấm tay, `qa_check` bản cũ) | Chồng > 20%: chấp nhận (sản phẩm phụ polygon nhiều mảnh). Lưu ý `reject_frac`: loại 1.987 ảnh `unmatched_rider` (28%) |
| COCO → coco_filled | Long | | | | | |
| Việt Nam nháp | Đạt | | | | | |
| Việt Nam đã duyệt (mỗi lô) | Bình Anh | | | | | |
| gold_val_v0 / v1 | Bình Anh | | | | | |
| merged_vN | Lộc | | | | | |

### Mô hình soi nhãn (`label_audit.py`)

| Dữ liệu | Mô hình | Ảnh có nghi vấn | Sót / sai lớp / biên / thừa | Đã duyệt | Tỉ lệ là lỗi thật | Ghi chú |
|---|---|---|---|---|---|---|
| | | | | | | |

## Gold val

| Phiên bản | Số ảnh | Instance/lớp thấp nhất | sha256 | Ghi chú |
|---|---|---|---|---|
| gold_val_v0 | | | | |
| gold_val_v1 | | | | khóa sau G2 |

## Phiên bản dataset

| Phiên bản | Recipe | Ảnh train | Instance/lớp thấp nhất (lớp) | sha256 | Drive |
|---|---|---|---|---|---|
| merged_v1 | | | | | |

## Các lượt huấn luyện

| Lượt | Ablation | Dataset | Thay đổi | Epoch | mAP gold val (×100) | 3 lớp yếu nhất (AP) | Nộp # | Public | Người chạy | Kết luận |
|---|---|---|---|---|---|---|---|---|---|---|
| R01 | A0 | | baseline Vistas | | | | | | | |
| R02 | A1 | | + IDD | | | | | | | |
| R03 | A2 | | + COCO | | | | | | | |
| R04 | A3 | | chọn ảnh ưu tiên | | | | | | | |
| R05 | A4 | | + Việt Nam nháp | | | | | | | |
| R06 | A5 | | + Việt Nam duyệt ×2 | | | | | | | |
| R07 | A6 | | + ảnh nền khó | | | | | | | |
| R08 | A7 | | copy_paste 0.2 | | | | | | | |
| R09 | A8 | | close_mosaic, hsv_v | | | | | | | |
| F1 | — | merged_final | công thức tốt nhất, 60 ep | | | | | | | |
| F2 | — | merged_final | 80 ep | | | | | | | |
| F3 | — | merged_final_vnheavy | Việt Nam ×3 | | | | | | | |

## Các lượt nộp

| # | Giờ | Lượt | mAP gold val | Public | Ghi chú |
|---|---|---|---|---|---|
| 1 | | | | | |

## Nhật ký ca đêm

| Giờ | Người trực | Lượt | Epoch | mAP val | Sự cố / xử lý |
|---|---|---|---|---|---|
| | | | | | |

## 2 bài chọn cho private

1. Bài A (tốt nhất gold val): lượt … · lý do: …
2. Bài B (tốt nhất public / đa dạng hóa): lượt … · lý do: …
