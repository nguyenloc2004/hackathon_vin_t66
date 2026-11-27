# Milestone chi tiết

Leader (**Lộc**) dùng file này để theo dõi tiến độ **theo mốc thời gian**. Việc của từng người nằm trong [`assignments.md`](assignments.md); mỗi milestone liệt kê **mã việc** (ví dụ `DAT-03`) cần xong để qua mốc.

Trạng thái: ⬜ chưa bắt đầu · 🟡 đang làm · ✅ xong · ⛔ bị chặn · ⏭️ bỏ qua (ghi lý do)

Cách cập nhật: tại mỗi buổi họp nhanh (`12:00 · 15:00 · 18:00 · 21:00 · 00:30 · 07:00 · 09:30`), Lộc đổi cột **Trạng thái**, ghi **giờ xong thực tế**, rồi chép các quyết định vào [`experiment_log.md`](experiment_log.md). Milestone trễ hơn 30 phút thì áp dụng ngay cột **Nếu trễ**, không chờ.

---

## Tổng quan

| # | Milestone | Hạn (UTC+7) | Cổng | Người chốt | Trạng thái | Xong lúc |
|---|---|---|---|---|---|---|
| M0 | Khởi động: tài khoản, nguồn, công cụ | 27/11 11:45 | — | Lộc | ⬜ | |
| M1 | Pipeline chạy trọn + nguồn gold val | 27/11 13:00 | **G0** | Lộc, Hưng | ⬜ | |
| M2 | Nguồn public đầu tiên qua QA | 27/11 14:00 | — | Lộc | ⬜ | |
| M3 | Baseline có điểm + gold val v0 | 27/11 15:30 | **G1** | Lộc | ⬜ | |
| M4 | Dữ liệu đợt 1 + `merged_v1` + hạ tầng train | 27/11 18:00 | — | Lộc | ⬜ | |
| M5 | Ablation nguồn public + teacher bắt đầu | 27/11 19:30 | — | Hưng | ⬜ | |
| M6 | Chốt công thức public + gold val v1 khóa | 27/11 21:00 | **G2** | Lộc | ⬜ | |
| M7 | Gán nháp đợt 2 + `merged_v2` | 28/11 00:30 | — | Lộc | ⬜ | |
| M8 | Đóng băng dữ liệu, khởi chạy lượt dài | 28/11 01:30 (F1–F3 chạy trước 02:00) | **G3** | Lộc | ⬜ | |
| M9 | Trực đêm: lượt dài chạy xong | 28/11 07:00 | — | Thắng, Long | ⬜ | |
| M10 | Nộp các bản cuối | 28/11 09:30 | **G4** | Lộc | ⬜ | |
| M11 | Chọn 2 bài + hồ sơ tái lập | 28/11 10:15 | **G5** | Lộc | ⬜ | |
| M12 | Slide xong + tập thuyết trình | 28/11 11:30 | — | Hưng, Lộc | ⬜ | |

### Chỉ số theo dõi (cập nhật mỗi buổi họp)

| Chỉ số | Mục tiêu | 15:00 | 18:00 | 21:00 | 00:30 | 07:00 |
|---|---|---|---|---|---|---|
| Ảnh gold val đã duyệt chéo | 60–80 (G1) → 150 (G2) | | | | | |
| Số video Việt Nam (train) | 30–50 | | | | | |
| Khung Việt Nam sau lọc trùng | ~3.000 | | | | | |
| Ảnh Việt Nam đã người duyệt | ≥ 800 (G3) | | | | | |
| Ảnh public đã qua QA (Vistas / IDD / COCO) | 3–5k / 2–4k / ≤ 1,5k | | | | | |
| Lượt nộp đã dùng | ≤ 20 | | | | | |
| mAP gold val tốt nhất (×100) | tăng dần | | | | | |
| Lớp có AP thấp nhất (gold val) | — | | | | | |

---

## M0 · Khởi động · hạn 11:45 27/11

**Mục tiêu:** ai cũng có công cụ, tài khoản và đầu vào để bắt tay làm ngay.

| Việc | Mã | Trạng thái |
|---|---|---|
| Repo, Drive chung (`RoboTaxi_T66/`), sheet nhật ký, chia việc trong nhóm chat | LOC-01 | ⬜ |
| Hỏi BTC: "Ảnh test chụp từ camera gắn trên xe (phía trước) phải không?" | LOC-02 | ⬜ |
| Đăng ký + bắt đầu tải Mapillary Vistas v2 (validation trước) | THANG-01 | ⬜ |
| Đăng ký IDD, Cityscapes; tải COCO 2017 annotations + ảnh train | LONG-01 | ⬜ |
| Khởi tạo `vn_sources.csv`, cài `yt-dlp` | DAT-01 | ⬜ |
| Dựng CVAT project `RoboTaxi_T66` với 11 lớp đúng thứ tự id | BANH-01 | ⬜ |
| Tạo dataset dữ liệu mẫu trên Kaggle, chạy thử notebook BTC | HUNG-01 | ⬜ |

**Tiêu chí xong:** mọi người đã chạy `python tools/test_common.py` → `ALL OK` trên máy mình.
**Nếu trễ:** đăng ký IDD/Cityscapes chưa được duyệt thì không chờ, Long làm COCO trước.

---

## M1 · Pipeline + nguồn gold val · hạn 13:00 27/11 · **G0**

**Mục tiêu:** chắc chắn nộp được bài; Bình Anh có ảnh để bắt đầu gold val.

| Việc | Mã | Trạng thái |
|---|---|---|
| Notebook BTC chạy trọn Bước 0–6 trên Kaggle với `TEAM_ID=71`, ra `submission.zip` hợp lệ | HUNG-02 | ⬜ |
| `convert_vistas.py --check 5` khớp ~100% trên dữ liệu thật | THANG-02 | ⬜ |
| 3–5 video gold val (kênh **khác** train, ưu tiên dashcam) → trích khung → giao Bình Anh | DAT-02 | ⬜ |
| Chọn 80 khung gold val, gán nháp bằng COCO x-seg, nạp vào CVAT | BANH-02 | ⬜ |
| `qa_check.py`, `visualize.py` dùng được trên output của converter | LOC-03 | ⬜ |

**Tiêu chí xong:** có `submission.zip` từ notebook (chưa cần nộp) và 80 khung gold val đã nằm trong CVAT.
**Nếu trễ:** Lộc và Hưng dồn sức sửa notebook; những người khác vẫn làm dữ liệu.

---

## M2 · Nguồn public đầu tiên qua QA · hạn 14:00 27/11

**Mục tiêu:** có ít nhất một nguồn sạch để train baseline.

| Việc | Mã | Trạng thái |
|---|---|---|
| Vistas validation (~2.000 ảnh) đã chuyển, `qa_check` sạch, 30 ảnh xem bằng mắt **ĐẠT** | THANG-03 | ⬜ |
| COCO đường phố `--max-images 1500` đã chuyển, `qa_check` sạch | LONG-02 | ⬜ |
| Lộc cùng xem ảnh mẫu, ghi ĐẠT/KHÔNG ĐẠT vào `experiment_log.md` | LOC-04 | ⬜ |

**Tiêu chí xong:** Motorcyclist/Cyclist trong ảnh mẫu phủ cả xe lẫn người; tỉ lệ ảnh bị loại vì rider không ghép được ở mức hợp lý (đọc `stats.json`).
**Nếu trễ:** baseline chỉ dùng Vistas validation.

---

## M3 · Baseline + gold val v0 · hạn 15:30 27/11 · **G1**

| Việc | Mã | Trạng thái |
|---|---|---|
| Train baseline A0 (notebook mặc định + Vistas) → 📤 **lượt nộp #1** | HUNG-03 | ⬜ |
| Gold val v0: 60–80 ảnh, đã duyệt chéo 100%, xuất `gold_val_v0.zip` + sha256 | BANH-03 | ⬜ |
| 20–30 video ban ngày cho train đã tải, trích khung, lọc trùng | DAT-03 | ⬜ |
| Converter IDD chạy (nếu đã có quyền) | LONG-03 | ⬜ |

**Tiêu chí xong:** lượt #1 có điểm trên leaderboard; `gold_val_v0.zip` nằm trên Drive và đã ghi vào log.
**Nếu trễ:** tạm dùng val chia ngẫu nhiên từ Vistas, ghi rõ trong log.

---

## M4 · Dữ liệu đợt 1 + hạ tầng train · hạn 18:00 27/11

| Việc | Mã | Trạng thái |
|---|---|---|
| `build_dataset.py` + recipe → `merged_v1` (zip + manifest + sha256), upload lên Kaggle Datasets (Private) | LOC-05 | ⬜ |
| Vistas mở rộng lên 3–5k ảnh (split training) | THANG-04 | ⬜ |
| IDD toàn bộ + thống kê lớp; Roboflow xem nhanh | LONG-04 | ⬜ |
| Khung Việt Nam đợt 1 đã gán nháp bằng COCO x-seg | DAT-04 | ⬜ |
| Notebook đội `notebooks/train_t66.ipynb` (preset, ô gom output); script `tools/train_teacher.py` | HUNG-04 | ⬜ |
| Ablation A1 (Vistas) và A2 (Vistas + IDD) đang chạy song song | HUNG-05 | ⬜ |
| Gold val tăng dần về 150 ảnh | BANH-04 | ⬜ |

**Tiêu chí xong:** `merged_v1` gắn được vào notebook Kaggle; A1, A2 đã khởi chạy.

---

## M5 · Ablation nguồn public + teacher · hạn 19:30 27/11

| Việc | Mã | Trạng thái |
|---|---|---|
| Ablation A3 (+COCO đã gán bù Sign) đang chạy | HUNG-06 | ⬜ |
| Teacher (`yolo26l/m-seg`, 1024) bắt đầu huấn luyện trên GPU mạnh nhất | HUNG-07 | ⬜ |
| Video **đêm / mưa** đã tải và trích khung | LONG-05 | ⬜ |
| Bộ ảnh Việt Nam công khai (Kaggle/Roboflow/HF) đã lọc đúng góc nhìn (bỏ CCTV, drone) | DAT-05 | ⬜ |
| Phân tích lỗi A1/A2 (AP từng lớp, ma trận nhầm lẫn) | LOC-06 | ⬜ |
| Duyệt chéo gold val | THANG-05 | ⬜ |

---

## M6 · Chốt công thức public + gold val v1 · hạn 21:00 27/11 · **G2**

| Việc | Mã | Trạng thái |
|---|---|---|
| Ablation A1–A3 có kết quả trên gold val, so sánh từng lớp | HUNG-08 | ⬜ |
| 📤 Lượt nộp #2–#3; kiểm tra gold val có tương quan với public LB không | HUNG-09 | ⬜ |
| **Gold val v1 = 150 ảnh, khóa phiên bản** (không đổi sau mốc này) | BANH-05 | ⬜ |
| Ảnh Việt Nam đợt 1 đã vào hàng duyệt CVAT (ưu tiên lớp hiếm, đêm, mưa) | DAT-06 | ⬜ |
| Quyết định nguồn public + trọng số, ghi vào log | LOC-07 | ⬜ |

**Tiêu chí xong:** mỗi lớp có ≥ 15 instance trong gold val; công thức public đã ghi trong `experiment_log.md`.
**Nếu trễ:** giữ tổ hợp tốt nhất hiện có, không mở thêm nguồn.

---

## M7 · Gán nháp đợt 2 + `merged_v2` · hạn 00:30 28/11

| Việc | Mã | Trạng thái |
|---|---|---|
| Teacher gán nháp đợt 2 (TTA) toàn bộ ảnh Việt Nam → `confident.txt` + `review_queue.csv` | DAT-07 | ⬜ |
| Cả đội duyệt ảnh Việt Nam (mục tiêu ≥ 500 ảnh lúc 00:30) | Cả đội | ⬜ |
| QA chéo 10% mỗi task, thống kê tỉ lệ lỗi | BANH-06 | ⬜ |
| `merged_v2` = public đã chốt + Việt Nam đã duyệt (×2) + Việt Nam nháp tin cậy | LOC-08 | ⬜ |
| Ablation A4 (+Việt Nam nháp) / A5 (+Việt Nam duyệt); 📤 #4–#6 | HUNG-10 | ⬜ |

---

## M8 · Đóng băng dữ liệu · hạn 01:30 28/11 · **G3**

| Việc | Mã | Trạng thái |
|---|---|---|
| Duyệt lô cuối, **không ai duyệt sau 02:00** | DAT-08, BANH-07 | ⬜ |
| `merged_final` (+ biến thể `merged_final_vnheavy` ×3 Việt Nam) dựng xong, QA sạch | LOC-09 | ⬜ |
| 2–3 cấu hình lượt dài sẵn sàng | HUNG-11 | ⬜ |
| Khởi chạy F1 (Kaggle A); 📤 #7 (merged_final, lượt ngắn) | LOC-10 | ⬜ |
| Khởi chạy F2, F3 (Kaggle B, C) **trước 02:00** | HUNG-12 | ⬜ |

**Tiêu chí xong:** ≥ 800 ảnh Việt Nam đã duyệt trong `merged_final`; cả 3 lượt dài đang chạy.
**Nếu trễ:** đóng băng với số ảnh duyệt hiện có, phần còn lại dùng nhãn nháp đã qua lọc tin cậy.

---

## M9 · Trực đêm · 01:30 → 07:00 28/11

| Việc | Mã | Trạng thái |
|---|---|---|
| Ca trực 03:00–06:30: theo dõi F1–F3, lưu checkpoint, xử lý ngắt (`resume=True`) | THANG-06, LONG-06 | ⬜ |
| Ghi log trực: giờ, epoch, mAP val theo epoch, sự cố | THANG-06, LONG-06 | ⬜ |
| Slide phần "dữ liệu public" + biểu đồ thống kê lớp | THANG-07, LONG-07 | ⬜ |

**Quy tắc:** luôn có ít nhất 1 người thức; **không mở lượt mới sau 04:00**.

---

## M10 · Nộp bản cuối · hạn 09:30 28/11 · **G4**

| Việc | Mã | Trạng thái |
|---|---|---|
| Đánh giá F1–F3 trên gold val; 📤 #8–#14 (bản cuối + checkpoint trung gian) | LOC-11 | ⬜ |
| Phân tích lỗi bản tốt nhất, lưu ảnh trước/sau cho slide | HUNG-13 | ⬜ |

**Quy tắc:** không nộp lượt mới sau 09:45 trừ khi hàng đợi trống.

---

## M11 · Chọn 2 bài · hạn 10:15 28/11 · **G5**

| Việc | Mã | Trạng thái |
|---|---|---|
| Chọn bài A (gold val cao nhất) và bài B (public cao nhất hoặc khác công thức), ghi lý do | LOC-12 | ⬜ |
| Hồ sơ tái lập cho **cả 2 bài** trên Drive: zip dữ liệu, recipe, manifest, notebook đã chạy, `run_report.json` | LOC-13 | ⬜ |
| Bảng public dataset đầy đủ (tên, link, license, số ảnh, ánh xạ) | THANG-08, LONG-08 | ⬜ |

---

## M12 · Thuyết trình · 11:30 28/11

| Việc | Mã | Trạng thái |
|---|---|---|
| Slide hoàn chỉnh **trước 10:45** | HUNG-14 | ⬜ |
| Mỗi người gửi phần số liệu/ảnh của mình (xem `assignments.md`) | Cả đội | ⬜ |
| Tập thuyết trình ≥ 2 lần, chuẩn bị câu hỏi phản biện | LOC-14 | ⬜ |
