# Thuyết trình (50% điểm chung cuộc)

Biên tập: **Quang Hưng** · Người trình bày chính: **Lộc** · Mỗi phần có một “chủ” cung cấp nội dung.

## 1. Thông điệp chính

> “Mô hình cố định, nên chúng tôi coi dữ liệu là sản phẩm: xây một **bánh đà dữ liệu** (public dataset ánh xạ chuẩn → ảnh Việt Nam trích từ video sẵn có → teacher gán nháp → người duyệt theo quy ước → đo trên gold val Việt Nam), và **mỗi vòng đều có số đo** chứng minh thay đổi nào có ích.”

Ban giám khảo đánh giá giải pháp, nên slide phải trả lời được: **Làm gì? Vì sao? Có chứng cứ không? Tái lập được không?**

## 2. Dàn slide (10–12 slide)

| # | Slide | Nội dung | Chủ nội dung |
|---|---|---|---|
| 1 | Tiêu đề | Tên đội, thành viên, điểm public/private (nếu đã có) | Hưng |
| 2 | Bài toán và thách thức | Đặc thù giao thông Việt Nam; mô hình khóa; mAP trung bình lớp nên lớp hiếm quan trọng | Lộc |
| 3 | Chiến lược tổng thể | Sơ đồ bánh đà dữ liệu 4 tầng | Lộc |
| 4 | Dữ liệu public | **Bảng bắt buộc**: tên, link, license, số ảnh dùng, cách ánh xạ về 11 lớp; ví dụ ghép rider + xe thành Motorcyclist (ảnh trước/sau) | Thắng, Long |
| 5 | Dữ liệu Việt Nam thu thập | Nguồn (số video, số kênh, thành phố, license), tiêu chí chọn video, số khung trước/sau lọc trùng, tỉ lệ ngày/đêm/mưa | Đạt |
| 6 | Gán nháp + người duyệt | Teacher + SAM 3; chính sách tin cậy; **số liệu**: số ảnh duyệt, % nhãn bị sửa, thời gian/ảnh | Đạt, Bình Anh |
| 7 | Label guideline của đội | 4–6 ca khó đặc thù Việt Nam và quyết định (từ `conventions.md`), có ảnh minh họa | Bình Anh |
| 8 | Kiểm soát chất lượng | Gold val (cách xây, phân tầng), duyệt chéo, `qa_check`, tỉ lệ lỗi | Bình Anh |
| 9 | Các vòng cải thiện | **Biểu đồ**: mAP gold val + public qua từng phiên bản dữ liệu; bảng ablation A0–A8 | Hưng |
| 10 | Phân tích lỗi | AP từng lớp trước/sau; 2–3 cặp ảnh lỗi → hành động dữ liệu đã làm → kết quả | Hưng |
| 11 | Tái lập | Recipe → zip → sha256 → notebook → `run_report.json`; seed 71 | Lộc |
| 12 | Bài học và hướng tiếp | Điều gì hiệu quả nhất, điều gì không; nếu có thêm thời gian sẽ làm gì | Lộc |

## 3. Thu thập tư liệu ngay từ đầu (không đợi đến cuối)

| Tư liệu | Khi nào | Ai |
|---|---|---|
| Ảnh trước/sau khi ghép rider (3 ví dụ) | Khi converter Vistas xong | Thắng |
| Thống kê lớp từng nguồn (biểu đồ cột) | Mỗi lần dựng dataset | Lộc |
| Khung hình đại diện mỗi điều kiện (ngày/đêm/mưa/đông) + bảng `vn_sources.csv` | Khi trích khung | Đạt |
| Ví dụ nhãn nháp → nhãn đã duyệt (3 ví dụ) | Trong lúc duyệt | Bình Anh |
| Bảng điểm sau mỗi lượt + ảnh lỗi Bước 5 | Sau mỗi lượt | Hưng |
| Chụp màn hình leaderboard / lịch sử nộp | Sau mỗi lần nộp | Lộc |

Tất cả để trong `Drive/RoboTaxi_T66/slides_assets/`.

## 4. Câu hỏi phản biện cần chuẩn bị

- Vì sao chọn các public dataset này? Ánh xạ Motorcyclist làm thế nào, kiểm chứng ra sao?
- Làm sao biết gold val đại diện cho test? Tương quan gold val với public thế nào?
- Nhãn nháp có làm mô hình học theo lỗi của teacher không? Xử lý thế nào?
- Lớp yếu nhất là lớp nào, vì sao, đã làm gì?
- Dữ liệu có rò rỉ train/val không? Kiểm tra bằng cách nào?
- License của dữ liệu dùng có cho phép không?

## 5. Lịch chuẩn bị

| Giờ (28/11) | Việc |
|---|---|
| 07:00–09:30 | Mỗi chủ nội dung điền slide của mình |
| 09:30–10:30 | Hưng biên tập, thống nhất hình thức; chèn số liệu cuối |
| 10:30–11:00 | Tập lần 1, bấm giờ; sửa |
| 11:00–11:30 | Cập nhật điểm private (nếu được công bố), tập lần 2 |
