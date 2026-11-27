# Lịch làm việc theo giờ

Giả định bắt đầu chính thức lúc **11:00 27/11**. Nếu trễ, dời tất cả các mốc trước G3 tương ứng; **không dời G4 và G5**.

Ký hiệu: 🎯 cổng quyết định · 📤 lượt nộp · 😴 ngủ · 🔁 duyệt nhãn

## Họp nhanh (10 phút, đứng, theo giờ cố định)

`12:00` · `15:00` · `18:00` · `21:00` · `00:30` · `07:00` · `09:30`

Mỗi người trả lời 3 câu: **Đã xong gì? Sắp làm gì? Đang bị chặn bởi gì?** Lộc ghi quyết định vào `experiment_log.md`.

## Ngày 27/11

| Giờ | Lộc | Minh Thắng | Bảo Long | Đạt | Bình Anh | Quang Hưng |
|---|---|---|---|---|---|---|
| **11:00–12:00** | Repo, Drive, sheet; viết `tools/common.py` (lớp, polygon, lọc kích thước) | Đăng ký + tải Vistas v2 (val trước) | Đăng ký IDD, Cityscapes; tải COCO | Lập `vn_sources.csv`, tìm video/bộ ảnh Việt Nam; cài `yt-dlp` | Dựng CVAT, nạp 11 lớp; đọc guideline | Chạy notebook mẫu trên Kaggle với dữ liệu mẫu |
| **12:00–13:00** | `qa_check.py`, `visualize.py` | Đọc định dạng Vistas, viết khung converter | Converter COCO (đơn giản nhất, làm trước) | Tải 3–5 video **riêng cho gold val**, trích khung, giao cho Bình Anh; script trích khung + phash | Chọn 80 khung từ video gold val, gán nháp bằng COCO x-seg | 🎯 **G0 (13:00)**: notebook chạy trọn, có `submission.zip` |
| **13:00–15:00** | Hỗ trợ ghép rider (thuật toán chung), review code | Converter Vistas + ghép rider; xuất 2.000 ảnh val | Converter IDD (nếu có quyền), nếu chưa: lọc COCO đường phố | Tải 20–30 video ban ngày cho train, trích khung, lọc trùng (≤ 150 khung/video) | 🔁 Gán gold val v0 (60–80 ảnh) | Dựng baseline từ dữ liệu COCO/Vistas sẵn có |
| **15:00–15:30** | 🎯 **G1** | | | | | 📤 **#1 baseline** |
| **15:30–18:00** | `build_dataset.py` + recipe; dựng `merged_v1` | Mở rộng Vistas lên 3–5k ảnh có chọn lọc | IDD toàn bộ + thống kê; Roboflow (xem nhanh) | Trích khung, lọc trùng; gán nháp đợt 1 (COCO x-seg + SAM 3) | 🔁 Gold val v1 (tới 150 ảnh), cùng Long/Thắng khi rảnh | Ablation A1 (Vistas) và A2 (Vistas+IDD) song song trên 2 tài khoản |
| **18:00–19:30** | Phân tích lỗi A1/A2 | 🔁 Duyệt gold val (duyệt chéo) | Tìm và tải video **đêm / mưa**, trích khung | Tải bộ ảnh Việt Nam công khai (Kaggle/Roboflow/HF), lọc ảnh đúng góc nhìn | 🔁 Gold val | Ablation A3 (+COCO); bắt đầu huấn luyện **teacher** |
| **19:30–21:00** | Quyết định công thức public | 🔁 Duyệt ảnh Việt Nam đợt 1 | Trích khung đêm, đưa vào gán nháp | Đưa ảnh Việt Nam đợt 1 vào CVAT (hàng duyệt) | Phân việc duyệt, thống kê tiến độ | 📤 #2–#3 (A1–A3 tốt nhất) |
| **21:00** | 🎯 **G2**: chốt nguồn public + trọng số | | | | | |
| **21:00–00:30** | Dựng `merged_v2` (+ảnh Việt Nam đã duyệt); phân tích lỗi | 🔁 Duyệt | 🔁 Duyệt | Teacher gán nháp đợt 2 (TTA) cho toàn bộ ảnh Việt Nam | 🔁 Duyệt + QA chéo 10% | Ablation A4 (+Việt Nam nháp) / A5 (+Việt Nam duyệt); 📤 #4–#6 |

## Đêm 27/11 → sáng 28/11

| Giờ | Lộc | Minh Thắng | Bảo Long | Đạt | Bình Anh | Quang Hưng |
|---|---|---|---|---|---|---|
| **00:30–01:30** | Chốt recipe cuối; QA `merged_final` | 😴 | 😴 | 🔁 Duyệt lô cuối | 🔁 Duyệt lô cuối, chốt thống kê | Chuẩn bị 2–3 cấu hình lượt dài |
| **01:30** | 🎯 **G3: đóng băng dữ liệu** | 😴 | 😴 | | | |
| **01:30–03:00** | Khởi chạy lượt dài F1 (Kaggle A); 📤 #7 (merged_final, lượt ngắn) | 😴 | 😴 | 😴 (từ 02:00) | 😴 (từ 02:00) | Khởi chạy F2, F3 (Kaggle B, C) rồi 😴 |
| **03:00–06:30** | 😴 | **Trực ca**: theo dõi các lượt chạy, lưu checkpoint, xử lý ngắt kết nối; viết phần slide “dữ liệu public” | **Trực ca**: như trên; vẽ biểu đồ thống kê lớp cho slide | 😴 | 😴 | 😴 |
| **06:30–07:00** | Dậy, đọc báo cáo trực ca | Bàn giao | Bàn giao → 😴 ngắn | Dậy | Dậy | Dậy |
| **07:00–09:30** | Đánh giá F1–F3 trên gold val; 📤 #8–#14 | 😴 (07:00–09:00) rồi hỗ trợ slide | Hỗ trợ slide (ảnh trước/sau) | Slide: quy trình thu thập + gán nháp | Slide: guideline, quy trình duyệt, số liệu QA | Phân tích lỗi bản tốt nhất; biên tập slide |
| **09:30** | 🎯 **G4**: không nộp lượt mới sau 09:45 trừ khi hàng đợi trống | | | | | |
| **09:30–10:15** | Thêm lượt nộp dự phòng nếu cần; 🎯 **G5**: chọn 2 bài | Rà slide | Rà slide | Rà slide | Rà slide | Hoàn thiện slide |
| **10:15–10:30** | Xác nhận 2 bài trên website, lưu toàn bộ hồ sơ tái lập lên Drive | | | | | |
| **10:30–11:30** | Tập thuyết trình (2 lần), chốt slide, chuẩn bị câu hỏi phản biện | ← cả đội → | | | | |

## Nguyên tắc ca đêm

- Luôn có **ít nhất 1 người thức** theo dõi các lượt huấn luyện từ 01:30 đến 07:00.
- Người trực ghi lại trong `experiment_log.md`: giờ, epoch hiện tại, mAP val theo epoch, sự cố.
- Nếu lượt chạy chết: khởi động lại với `resume=True` từ `last.pt` trên Drive; nếu không kịp, bỏ lượt đó, **không** mở lượt mới sau 04:00 (sẽ không xong trước 08:00).
- Không ai duyệt nhãn sau 02:00: nhãn ban đêm khuya là nguồn lỗi lớn.
