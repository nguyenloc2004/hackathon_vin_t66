# configs/qa: file kỳ vọng cho `tools/qa_check.py --expect`

Mỗi nguồn dữ liệu có một file nói rõ **nguồn đó phải trông như thế nào**. Nhờ vậy cùng một bộ kiểm tra cho ra kết luận đúng với từng nguồn: COCO thô được phép không có Other vehicle, nhưng `coco_filled` mà thiếu Sign thì phải chặn.

| Khóa | Ý nghĩa |
|---|---|
| `profile` | `source` / `gold` / `merged` (mặc định đoán theo cấu trúc thư mục) |
| `classes` | `{Tên lớp: {min: N, target: N}}`: dưới `min` là **lỗi**, dưới `target` là cảnh báo |
| `val_classes` | như trên, cho split val của bộ gộp |
| `absent_ok` | các lớp được phép có 0 instance (không báo `class_missing`) |
| `thresholds` | ghi đè ngưỡng trong `DEFAULTS` của `qa_check.py` |
| `severity` | đổi mức của một mã kiểm tra: `error` / `warn` / `off` |
| `waivers` | miễn trừ có chủ đích: `{check: mã, item: mẫu glob, reason: lý do}`; vẫn hiện trong báo cáo dưới dạng "miễn trừ" |

Mã kiểm tra và mức mặc định theo profile: bảng `CHECKS` trong [`tools/qa_check.py`](../../tools/qa_check.py), tóm tắt ở phụ lục của [`plan/qa_process.md`](../../plan/qa_process.md).
Bộ gộp (`merged_vN`) thường không cần file kỳ vọng: profile `merged` đã có sẵn mục tiêu train ≥ 1.000 (Cyclist, Other vehicle ≥ 400) và val ≥ 15 instance mỗi lớp.
