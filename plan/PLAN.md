# Kế hoạch tổng thể: AIInAction RoboTaxi Data Centric Competition

Lead: **Nguyễn Hữu Lộc** · Thành viên: Minh Thắng, Bảo Long, Đạt, Nguyễn Bình Anh, Quang Hưng · **`TEAM_ID = 71`**

Tài liệu này là bản kế hoạch chính. Các tài liệu chi tiết:

| File | Nội dung | Ai đọc kỹ |
|---|---|---|
| [`milestones.md`](milestones.md) | 13 milestone M0–M12 (gắn cổng G0–G5), tiêu chí xong, chỉ số theo dõi | Lộc (theo dõi), cả đội |
| [`assignments.md`](assignments.md) | Bảng việc từng người có mã việc, hạn, trạng thái; bảng tổng quan cho leader | Cả đội (tự cập nhật) |
| [`schedule.md`](schedule.md) | Lịch theo giờ cho từng người, ca ngủ, họp nhanh | Cả đội |
| [`data_pipeline.md`](data_pipeline.md) | Đặc tả kỹ thuật: converter, ghép rider, raster/polygon, lọc, gộp, gán nhãn nháp | Lộc, Thắng, Long, Đạt |
| [`class_mapping.md`](class_mapping.md) | Ánh xạ nhãn từng public dataset về 11 lớp | Thắng, Long |
| [`labeling_sop.md`](labeling_sop.md) | Quy trình CVAT, gold val, duyệt nhãn, kiểm soát chất lượng | Cả đội (bắt buộc) |
| [`conventions.md`](conventions.md) | Các ca khó đã chốt của đội (sổ quy ước sống) | Cả đội (bắt buộc) |
| [`training_submission.md`](training_submission.md) | Ngân sách GPU, thiết kế ablation, công thức huấn luyện, chiến lược 20 lượt nộp, chọn 2 bài cuối | Lộc, Hưng |
| [`presentation.md`](presentation.md) | Câu chuyện, dàn slide, số liệu cần thu thập dọc đường | Hưng, Lộc, cả đội |
| [`experiment_log.md`](experiment_log.md) | Nhật ký dữ liệu, huấn luyện, nộp bài | Người chạy và người nộp |

---

## 1. Hiểu đúng cuộc chơi

| Yếu tố | Sự thật | Hệ quả hành động |
|---|---|---|
| Mô hình khóa: `yolo26n-seg`, 640 | Năng lực mô hình nhỏ, cố định | Không phí thời gian dò tham số; mọi giờ công dồn vào **dữ liệu** |
| Điểm = mask mAP@[.5:.95], **trung bình đều 11 lớp** | Lớp hiếm nặng ngang lớp phổ biến | Mỗi lớp phải đủ ví dụ; một lớp gần 0 AP kéo tụt cả điểm |
| IoU từ 0,50 tới 0,95 | Biên mask lệch vài pixel cũng mất điểm ở ngưỡng cao | Polygon phải sát; ưu tiên nguồn có mask chất lượng (Vistas, IDD) hơn mask thô |
| Test ẩn: ảnh giao thông Việt Nam thật (suy luận từ đề) | Lệch miền là kẻ thù số 1 | **Ảnh Việt Nam thu thập từ video/bộ ảnh sẵn có + gold val Việt Nam** là trục chính |
| Bộ chấm bỏ GT < 64 px (thang 640), `max_det = 100` | Vật cực nhỏ không tính | Không cố gán vật tí hon; tập trung vật cỡ vừa và lớn |
| 20 lượt nộp, một bài trong hàng đợi | Phản hồi leaderboard ít và chậm | Quyết định bằng **gold val nội bộ**, leaderboard chỉ để xác nhận |
| Xếp hạng = 50% private + **50% thuyết trình** | Quy trình và câu chuyện có giá trị ngang điểm số | Ghi số liệu, ảnh minh họa **từ vòng đầu tiên**, không để đến cuối mới làm slide |
| Phải công khai mọi public dataset | Bị hỏi về nguồn và license | Ghi lại nguồn ngay khi tải (bảng trong `class_mapping.md`) |

### Mốc giờ (UTC+7)

`09:00 27/11` mở nộp → **`10:30 28/11` đóng nộp** → `11:00` công bố private → `11:30` thuyết trình.

---

## 2. Mục tiêu (north star)

1. **Điểm private cao và ổn định**: tối ưu điểm trên gold val Việt Nam, không dò leaderboard public.
2. **Không lớp nào bị bỏ rơi**: mục tiêu mỗi lớp có ít nhất **1.000 instance trong train** (Cyclist, Other vehicle tối thiểu 400) và ít nhất **15 instance trong gold val**.
3. **Một quy trình dữ liệu kể được thành câu chuyện**: thu thập → ánh xạ → gán nháp → người duyệt → đo → sửa, có số liệu từng vòng.
4. **Tái lập được 100%**: mỗi bài nộp truy ngược được về zip dữ liệu, notebook và `run_report.json`.

---

## 3. Chiến lược: “bánh đà dữ liệu” 4 tầng

```
 Tầng 1  Public dataset đã ánh xạ chuẩn ──┐
                                           ├─► Teacher lớn (l/x-seg, 1024) ──► Gán nháp ảnh Việt Nam
 Tầng 2  Ảnh Việt Nam (video sẵn có)  ────┘                                        │
                                                                                     ▼
 Tầng 3  Người duyệt (CVAT + SAM) theo SOP ◄──── ưu tiên ảnh mô hình kém tự tin ◄────┘
                    │
                    ▼
 Tầng 4  Gộp có trọng số → huấn luyện yolo26n-seg → đo trên gold val → phân tích lỗi → quay lại tầng 2/3
```

### Tầng 1: Public dataset (nền rộng)

| Ưu tiên | Dataset | Lấy bao nhiêu | Lý do |
|---|---|---|---|
| 1 | **Mapillary Vistas v2** | 3.000–5.000 ảnh, chọn ảnh nhiều xe 2 bánh / lớp hiếm | Instance mask đủ cả biển báo, đèn, rider; chất lượng biên tốt |
| 2 | **IDD Segmentation** | 2.000–4.000 ảnh | Giao thông hỗn hợp gần Việt Nam nhất (xe máy, xe ba bánh, xe tải) |
| 3 | **COCO 2017** (lọc ảnh đường phố) | ≤ 1.500 ảnh | Bổ sung Bicycle/Cyclist/Motorcycle; **phải gán nháp thêm biển báo** |
| 4 | Cityscapes / BDD100K | Chỉ khi thời gian còn | Thiếu instance biển báo, đèn; cần gán nháp bù |
| 5 | Roboflow Universe (giao thông Việt Nam có polygon) | Tùy chất lượng | Đúng miền; phải đối chiếu quy ước trước khi dùng |

### Tầng 2: Ảnh Việt Nam (trục chính)

Đội **không tự quay**; toàn bộ ảnh Việt Nam lấy từ **video và bộ ảnh đã có sẵn trên mạng**.

- **Video YouTube**: dashcam ô tô, video đi đường, walking tour ở Hà Nội, TP.HCM, Đà Nẵng và các tỉnh. Từ khóa gợi ý: `dashcam Hà Nội`, `Saigon traffic 4K`, `Hanoi street night`, `lái xe trời mưa Hà Nội`, `giờ cao điểm ngã tư`.
  - Ưu tiên bật bộ lọc **Creative Commons** của YouTube. Video thường thì ghi lại nguồn, chỉ dùng trong cuộc thi, không phát tán lại ảnh.
  - Tải bằng `yt-dlp` ở 1080p.
- **Bộ ảnh giao thông Việt Nam công khai** trên Kaggle, Roboflow Universe, Hugging Face, kể cả bộ chỉ có bbox hoặc không có nhãn (ví dụ dữ liệu biển báo Zalo AI Challenge 2020): lấy **ảnh**, còn nhãn thì làm lại bằng gán nháp + người duyệt. Kiểm tra license từng bộ.
- **Tiêu chí chọn video**:
  - Góc nhìn giống camera xe: dashcam phía trước hoặc ngang tầm mắt người.
  - Loại bỏ: góc drone/flycam, timelapse, video có chữ hoặc watermark lớn, video bị làm mờ hay chỉnh màu quá mức.
  - Đa dạng: nhiều thành phố; ngày, đêm, mưa; ngã tư có đèn; đường lớn, phố nhỏ, quốc lộ có xe tải.
- **Chống một video áp đảo**: 30–50 video, mỗi video lấy **≤ 150 khung**.
- Mục tiêu: **3.000 khung hình** sau khi lọc trùng, trong đó **≥ 800 ảnh được người duyệt**.
- **Gold val tách riêng theo kênh/video**: ảnh gold val lấy từ kênh YouTube hoặc bộ ảnh **không dùng** cho train, để tránh rò rỉ (cùng kênh thường cùng camera, cùng tuyến đường).

### Tầng 3: Gán nháp và người duyệt

- Đợt 1 (ngay, không cần huấn luyện): mô hình COCO pretrained lớn + SAM 3 cho lớp thiếu (biển báo, phương tiện khác).
- Đợt 2 (từ ~20:00): **teacher** huấn luyện trên dữ liệu đã gộp, suy luận với TTA trên ảnh Việt Nam.
- Người duyệt theo [`labeling_sop.md`](labeling_sop.md); ảnh chưa duyệt chỉ được dùng nếu qua bộ lọc tin cậy (xem [`data_pipeline.md`](data_pipeline.md) §6).

### Tầng 4: Gộp và đo

- Mỗi phiên bản dữ liệu được dựng từ một **công thức (recipe) YAML** cố định → zip + manifest + sha256.
- Mọi so sánh dùng **cùng gold val**, cùng ngân sách huấn luyện ablation.
- Chỉ giữ thay đổi nào tăng gold val **≥ 0,5 điểm** (dưới mức này coi là nhiễu).

---

## 4. Cổng quyết định (decision gates)

| Cổng | Hạn | Điều kiện qua cổng | Nếu không qua |
|---|---|---|---|
| **G0: Pipeline** | 27/11 13:00 | Notebook chạy trọn Bước 0–6 với `TEAM_ID=71`, tạo được `submission.zip` hợp lệ | Lộc + Hưng dồn sức sửa, mọi người khác vẫn làm dữ liệu |
| **G1: Baseline** | 27/11 15:30 | Lượt nộp #1 có điểm; gold val v0 (≥ 60 ảnh) sẵn sàng | Dùng val chia ngẫu nhiên từ Vistas tạm thời |
| **G2: Công thức dữ liệu** | 27/11 21:00 | Ablation A1–A4 xong; chọn tổ hợp nguồn public tốt nhất; gold val v1 (≥ 150 ảnh) | Giữ tổ hợp tốt nhất hiện có, không mở thêm nguồn |
| **G3: Đóng băng dữ liệu** | 28/11 01:30 | `merged_final` được dựng, QA sạch, ≥ 800 ảnh Việt Nam đã duyệt | Đóng băng với số ảnh duyệt hiện có, phần còn lại dùng nhãn nháp đã lọc |
| **G4: Nộp cuối** | 28/11 09:30 | Các lượt huấn luyện dài đã nộp và có điểm | Nộp bản tốt nhất hiện có |
| **G5: Chọn bài** | 28/11 10:15 | Đã chọn 2 bài trên website | Hệ thống tự lấy 2 bài gần nhất (tránh để xảy ra) |

---

## 5. Phân công và trách nhiệm (RACI rút gọn)

| Mảng | Chịu trách nhiệm chính | Hỗ trợ | Đầu ra bắt buộc |
|---|---|---|---|
| Điều phối, cổng quyết định, nộp bài | **Lộc** | Hưng | `experiment_log.md`, quyết định từng cổng |
| Khung code chung (`tools/common.py`, `build_dataset.py`, `qa_check.py`) | **Lộc** | Thắng | Script dựng dataset từ recipe |
| Mapillary Vistas (+ Cityscapes) | **Minh Thắng** | Lộc | `vistas_yolo/` + thống kê lớp + 30 ảnh xem mẫu |
| IDD + COCO + Roboflow | **Bảo Long** | Thắng | `idd_yolo/`, `coco_yolo/` + thống kê + ảnh mẫu |
| Tìm và tải video/bộ ảnh Việt Nam, trích khung, lọc trùng, gán nháp | **Đạt** | Long (gán nháp, tìm video đêm/mưa) | `vn_sources.csv` (danh sách nguồn + license), `vn_raw/`, `vn_pseudo/`, hàng đợi duyệt |
| CVAT, gold val, điều phối duyệt nhãn, sổ quy ước | **Nguyễn Bình Anh** | Cả đội | `gold_val_vX.zip`, `conventions.md`, thống kê duyệt |
| Huấn luyện, ablation, teacher, phân tích lỗi | **Quang Hưng** | Lộc | Bảng kết quả, ảnh lỗi, `best.pt` trên Drive |
| Slide và thuyết trình | **Quang Hưng** (biên tập), **Lộc** (trình bày) | Cả đội cung cấp số liệu, ảnh | Slide hoàn chỉnh trước 10:45 28/11 |

Quy tắc chung: **ai xong việc chính thì chuyển sang duyệt nhãn**, vì đây luôn là nút thắt cổ chai.

---

## 6. Ngân sách tài nguyên

| Tài nguyên | Có | Kế hoạch dùng |
|---|---|---|
| GPU | Kaggle (khoảng 30 giờ/tuần/tài khoản, T4×2 hoặc P100) × 6 tài khoản; Colab; ai có Colab Pro / GPU thuê thì báo ngay | Ablation chạy song song trên 3–4 tài khoản; teacher trên GPU mạnh nhất |
| Giờ công duyệt nhãn | ~6 người × ~8 giờ hữu ích | ≈ 150 ảnh gold val (~8 giờ công) + 800–1.000 ảnh duyệt nhanh (~25 giờ công) |
| Lượt nộp | 20 | 12 cho ablation, 5 cho bản cuối, 3 dự phòng (chi tiết trong `training_submission.md`) |
| Dung lượng | Drive chung | Ảnh resize về cạnh dài 1280, JPEG chất lượng 92 |

---

## 7. Rủi ro và phương án

| Rủi ro | Khả năng | Ảnh hưởng | Phòng ngừa / phản ứng |
|---|---|---|---|
| Đăng ký IDD/Cityscapes bị duyệt chậm | Cao | Trung bình | Đăng ký ngay 11:00; Vistas + COCO + ảnh Việt Nam đủ để chạy |
| Converter ghép rider sai → Motorcyclist hỏng | Trung bình | **Rất cao** (lớp phổ biến nhất ở Việt Nam) | Bắt buộc xem 30 ảnh mẫu mỗi nguồn trước khi gộp; unit test trên ảnh khó |
| Dữ liệu public làm tụt điểm do lệch miền | Trung bình | Cao | Ablation từng nguồn; tăng trọng số ảnh Việt Nam |
| Nhãn nháp nhiễu (sót vật → dạy mô hình coi là nền) | Cao | Cao | Bộ lọc tin cậy; ảnh nghi ngờ vào hàng duyệt hoặc loại bỏ |
| Colab ngắt giữa chừng | Cao | Trung bình | Lượt dài chạy trên Kaggle; luôn đặt `DRIVE_OUTPUT_DIR` |
| Hàng đợi chấm đông sáng 28/11 | Cao | Cao | Bản cuối nộp **trước 09:30**; không dồn |
| Kiệt sức ban đêm → nhãn ẩu | Cao | Cao | Ngủ luân phiên theo `schedule.md`; duyệt chéo 10% |
| Video sẵn có kém chất lượng (nén mạnh, mờ chuyển động, góc quay khác camera xe) | Trung bình | Trung bình | Lọc theo tiêu chí §3 Tầng 2; bỏ khung mờ; giới hạn 150 khung/video |
| Bị hỏi về bản quyền video | Trung bình | Thấp–Trung bình | Ưu tiên video Creative Commons; ghi đủ nguồn trong `vn_sources.csv`; không công bố lại ảnh |
| Mất file / không tái lập | Thấp | Rất cao | Recipe + manifest + sha256 cho mọi phiên bản; mọi thứ lên Drive |
| Gold val quá nhỏ → so sánh nhiễu | Trung bình | Cao | ≥ 150 ảnh, phủ đủ lớp; ngưỡng cải thiện 0,5 điểm |

---

## 8. Định nghĩa “xong” (Definition of Done) trước 10:30 28/11

- [ ] 2 bài đã được chọn trên website, ghi lý do chọn vào `experiment_log.md`
- [ ] Zip dữ liệu cuối + recipe + manifest + notebook đã chạy + `run_report.json` lưu trên Drive cho **cả 2 bài**
- [ ] Bảng public dataset đầy đủ (tên, link, license, số ảnh, ánh xạ)
- [ ] Slide có biểu đồ điểm qua các vòng, ảnh lỗi trước/sau, thống kê duyệt nhãn
- [ ] Cả đội đã tập thuyết trình ít nhất 1 lần

---

## 9. Việc cần làm ngay (trong 45 phút)

- [ ] **Lộc**: tạo cấu trúc repo (`tools/`, `configs/`), Drive chung, sheet nhật ký; chia việc trong nhóm chat
- [ ] **Minh Thắng**: đăng ký và tải Mapillary Vistas v2 (bắt đầu bằng phần validation 2.000 ảnh cho nhanh)
- [ ] **Bảo Long**: đăng ký IDD, Cityscapes; tải COCO 2017 annotations + ảnh train
- [ ] **Quang Hưng**: chạy notebook mẫu với `TEAM_ID=71` trên Kaggle bằng dữ liệu mẫu (qua cổng G0)
- [ ] **Đạt**: lập `vn_sources.csv` (link, kênh, thành phố, ngày/đêm/mưa, license), cài `yt-dlp`, chuẩn bị script trích khung + lọc trùng; chọn trước 3–5 video riêng cho gold val và giao cho Bình Anh
- [ ] **Nguyễn Bình Anh**: dựng CVAT (11 lớp đúng thứ tự id), đọc kỹ guideline, khởi tạo `conventions.md` cùng Lộc
