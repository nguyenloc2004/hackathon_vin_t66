# Phân công công việc theo thành viên

Mỗi người có một bảng việc riêng, **xếp theo giờ**. Mã việc (ví dụ `DAT-03`) khớp với [`milestones.md`](milestones.md), nên leader nhìn được cả hai chiều: *mốc này còn thiếu việc của ai* và *người này đang ở bước nào*.

Trạng thái: ⬜ chưa bắt đầu · 🟡 đang làm · ✅ xong · ⛔ bị chặn · ⏭️ bỏ qua (ghi lý do)

**Quy tắc cập nhật**
- Xong một việc: đổi sang ✅, ghi **giờ xong** và **link đầu ra** (Drive, Kaggle, commit) vào cột *Xong lúc / đầu ra*. Nếu không tự sửa file được thì nhắn vào nhóm chat theo mẫu: `✅ DAT-03 15:10 <link>`, Lộc sẽ cập nhật.
- Bị chặn: đổi sang ⛔ và ghi một dòng vào mục **Đang bị chặn** cuối file. Không im lặng chờ quá 15 phút.
- Xong hết việc chính: chuyển sang **duyệt nhãn** (đây luôn là nút thắt cổ chai).

---

## Bảng tổng quan cho leader

Cập nhật tại mỗi buổi họp nhanh (`12:00 · 15:00 · 18:00 · 21:00 · 00:30 · 07:00 · 09:30`).

| Thành viên | Vai trò chính | Tổng việc | ✅ Xong | 🟡 Đang làm | ⛔ Bị chặn | Việc hiện tại | Cập nhật lúc |
|---|---|---|---|---|---|---|---|
| Lộc | Điều phối, dựng dataset, nộp bài | 14 | 0 | 0 | 0 | | |
| Minh Thắng | Mapillary Vistas | 9 | 0 | 0 | 0 | | |
| Bảo Long | IDD, COCO, Roboflow, video đêm/mưa | 9 | 0 | 0 | 0 | | |
| Đạt | Ảnh Việt Nam: thu thập, trích khung, gán nháp | 9 | 0 | 0 | 0 | | |
| Nguyễn Bình Anh | CVAT, gold val, điều phối duyệt nhãn | 8 | 0 | 0 | 0 | | |
| Quang Hưng | Huấn luyện, ablation, teacher, slide | 14 | 0 | 0 | 0 | | |

---

## Lộc · điều phối, khung code, dựng dataset, nộp bài

| Mã | Việc | Hạn | Tiêu chí xong / đầu ra | Mốc | Trạng thái | Xong lúc / đầu ra |
|---|---|---|---|---|---|---|
| LOC-01 | Repo, Drive `RoboTaxi_T66/`, sheet nhật ký, chia việc trong nhóm chat | 11:45 27/11 | Mọi người truy cập được Drive và repo | M0 | ⬜ | |
| LOC-02 | Hỏi BTC góc camera của ảnh test | 11:45 | Có câu trả lời, báo Đạt và Bình Anh | M0 | ⬜ | |
| LOC-03 | `qa_check.py`, `visualize.py` dùng được trên output converter | 13:00 | Chạy được trên Vistas val | M1 | ⬜ | |
| LOC-04 | Cùng xem 30 ảnh mẫu mỗi nguồn, ghi ĐẠT/KHÔNG ĐẠT | 14:00 | Dòng log cho Vistas, COCO | M2 | ⬜ | |
| LOC-05 | Recipe + `build_dataset.py` → `merged_v1`, upload Kaggle Datasets (Private) | 18:00 | zip + manifest + sha256; ghi version Kaggle | M4 | ⬜ | |
| LOC-06 | Phân tích lỗi A1/A2 | 19:30 | 3 dòng kết luận trong log | M5 | ⬜ | |
| LOC-07 | Chốt nguồn public + trọng số | 21:00 | Quyết định G2 trong log | M6 | ⬜ | |
| LOC-08 | Dựng `merged_v2` (+Việt Nam duyệt ×2, +nháp tin cậy) | 00:30 28/11 | zip + manifest trên Drive và Kaggle | M7 | ⬜ | |
| LOC-09 | Dựng `merged_final` + `merged_final_vnheavy`, QA sạch | 01:30 | `qa_check` không lỗi | M8 | ⬜ | |
| LOC-10 | Khởi chạy F1; 📤 #7 | 01:30 | F1 đang chạy, #7 đã vào hàng đợi | M8 | ⬜ | |
| LOC-11 | Đánh giá F1–F3 trên gold val; 📤 #8–#14 | 09:30 | Bảng so sánh trong log | M10 | ⬜ | |
| LOC-12 | Chọn 2 bài, ghi lý do | 10:15 | Đã xác nhận trên website | M11 | ⬜ | |
| LOC-13 | Hồ sơ tái lập cho cả 2 bài lên Drive | 10:30 | zip, recipe, manifest, notebook, `run_report.json` | M11 | ⬜ | |
| LOC-14 | Trình bày; tập ≥ 2 lần cùng cả đội | 11:30 | Đã tập 2 lần | M12 | ⬜ | |

---

## Minh Thắng · Mapillary Vistas (+ Cityscapes)

> Converter đã có sẵn (`tools/convert_vistas.py`): **bỏ qua bước viết converter**, làm thẳng từ kiểm tra giải mã.

| Mã | Việc | Hạn | Tiêu chí xong / đầu ra | Mốc | Trạng thái | Xong lúc / đầu ra |
|---|---|---|---|---|---|---|
| THANG-01 | Đăng ký + tải Vistas v2 (validation trước); `test_common.py` → `ALL OK` | 11:45 27/11 | Dữ liệu trên máy/Drive | M0 | ⬜ | |
| THANG-02 | `convert_vistas.py --check 5` trên dữ liệu thật | 13:00 | Khớp ~100%; nếu không khớp thì báo Lộc ngay | M1 | ⬜ | |
| THANG-03 | Chuyển Vistas validation (~2.000 ảnh), `qa_check`, xem 30 ảnh (+20 ảnh rider) | 14:00 | `converted/vistas` + `stats.json`; ĐẠT trong log; báo Hưng làm baseline | M2 | ⬜ | |
| THANG-04 | Mở rộng Vistas (split training) lên 3–5k ảnh; ghi dòng Vistas vào bảng nguồn `class_mapping.md` | 18:00 | Output + thống kê lớp | M4 | ⬜ | |
| THANG-05 | 🔁 Duyệt chéo gold val | 19:30 | Số ảnh đã duyệt báo Bình Anh | M5 | ⬜ | |
| THANG-09 | 🔁 Duyệt ảnh Việt Nam đợt 1 | 00:30 28/11 | Số ảnh đã duyệt báo Bình Anh | M7 | ⬜ | |
| THANG-06 | Trực ca 03:00–06:30: theo dõi F1–F3, lưu checkpoint, xử lý ngắt | 06:30 | Log trực ca trong `experiment_log.md`; bàn giao cho Lộc | M9 | ⬜ | |
| THANG-07 | Slide phần "dữ liệu public" | 06:30 | Gửi Hưng | M9 | ⬜ | |
| THANG-08 | Bảng public dataset (Vistas, Cityscapes): tên, link, license, số ảnh, ánh xạ | 10:15 | Hoàn chỉnh trong `class_mapping.md` | M11 | ⬜ | |

---

## Bảo Long · IDD, COCO, Cityscapes, Roboflow, video đêm/mưa

| Mã | Việc | Hạn | Tiêu chí xong / đầu ra | Mốc | Trạng thái | Xong lúc / đầu ra |
|---|---|---|---|---|---|---|
| LONG-01 | Đăng ký IDD, Cityscapes; tải COCO 2017 annotations + ảnh train | 11:45 27/11 | Đã gửi đăng ký; COCO trên máy | M0 | ⬜ | |
| LONG-02 | `convert_coco.py --max-images 1500`, `qa_check`, xem 30 ảnh | 14:00 | `converted/coco`; ghi nhớ **bắt buộc** gán bù Sign trước khi gộp | M2 | ⬜ | |
| LONG-03 | Converter IDD (nếu đã có quyền) | 15:30 | Output + đọc kỹ bảng nhãn gốc → role | M3 | ✅ | 14:25 27/11 · `convert_idd.py` chạy xong (fix `polegroup`, lọc polygon `<1.5 px²`) |
| LONG-04 | IDD toàn bộ + thống kê; Roboflow xem nhanh (kiểm tra quy ước Motorcyclist) | 18:00 | `converted/idd` + `stats.json`; ghi ĐẠT/KHÔNG ĐẠT | M4 | 🔄 | Xong IDD (`5.006` ảnh, `54.367` inst, ĐẠT); còn xem nhanh Roboflow |
| LONG-05 | Tìm + tải video **đêm / mưa**, trích khung, đưa vào gán nháp | 21:00 | Khung trong `vn_raw/`, ghi nguồn vào `vn_sources.csv` | M5 | ⬜ | |
| LONG-09 | 🔁 Duyệt ảnh Việt Nam | 00:30 28/11 | Số ảnh đã duyệt báo Bình Anh | M7 | ⬜ | |
| LONG-06 | Trực ca 03:00–06:30 cùng Thắng | 06:30 | Log trực ca | M9 | ⬜ | |
| LONG-07 | Biểu đồ thống kê lớp cho slide; ảnh trước/sau | 09:30 | Gửi Hưng | M9 | ⬜ | |
| LONG-08 | Bảng public dataset (IDD, COCO, Roboflow): tên, link, license, số ảnh, ánh xạ | 10:15 | Hoàn chỉnh trong `class_mapping.md` | M11 | ⬜ | |

---

## Đạt · ảnh Việt Nam: thu thập, trích khung, lọc trùng, gán nháp

**Chỉ tiêu:** 30–50 video, ≤ 150 khung/video, ~3.000 khung sau lọc trùng. Ưu tiên **dashcam ô tô nhìn phía trước** (~70%), ngang tầm mắt (~20%), camera hông (~10%). **Bỏ** CCTV gắn cột, drone, timelapse, watermark lớn. **Chủ động săn lớp hiếm**: ba gác, xích lô, xe chở hàng, xe đạp, xe buýt. Cắt bỏ phần capo/táp-lô và dải chữ giờ/GPS.

| Mã | Việc | Hạn | Tiêu chí xong / đầu ra | Mốc | Trạng thái | Xong lúc / đầu ra |
|---|---|---|---|---|---|---|
| DAT-01 | Khởi tạo `vn_sources.csv` (id, link, kênh, thành phố, điều kiện, góc nhìn, license, dùng cho); cài `yt-dlp` | 11:45 27/11 | File trên Drive, ≥ 10 nguồn ứng viên | M0 | ⬜ | |
| DAT-02 | **3–5 video gold val** (kênh khác train), trích khung, giao Bình Anh | 13:00 | `vn_frames_gold/` trên Drive; cột "dùng cho = gold" | M1 | ⬜ | |
| DAT-03 | 20–30 video ban ngày cho train: tải, trích khung, lọc trùng (kể cả `--against` gold val) | 15:30 | `vn_raw/` + số khung trước/sau lọc | M3 | ⬜ | |
| DAT-04 | Gán nháp đợt 1 (`pseudo_label.py --model yolo26x-seg.pt --imgsz 1024`) | 18:00 | `vn_pseudo/` | M4 | ⬜ | |
| DAT-05 | Bộ ảnh Việt Nam công khai (Kaggle/Roboflow/HF): lấy ảnh, lọc đúng góc nhìn, kiểm tra license | 19:30 | Ảnh trong `vn_raw/`, nguồn trong `vn_sources.csv` | M5 | ⬜ | |
| DAT-06 | Đưa ảnh đợt 1 vào CVAT (ưu tiên lớp hiếm, đêm, mưa) | 21:00 | Task `review_{n}` trong CVAT | M6 | ⬜ | |
| DAT-07 | Gán nháp đợt 2 bằng teacher (`--augment`) cho toàn bộ ảnh Việt Nam | 00:30 28/11 | `confident.txt` + `review_queue.csv` | M7 | ⬜ | |
| DAT-08 | 🔁 Duyệt lô cuối; ngủ từ 02:00 | 01:30 | Số ảnh đã duyệt báo Bình Anh | M8 | ⬜ | |
| DAT-09 | Slide: quy trình thu thập + gán nháp (số video, số khung trước/sau lọc, phân bố ngày/đêm/mưa) | 09:30 | Gửi Hưng | M12 | ⬜ | |

---

## Nguyễn Bình Anh · CVAT, gold val, điều phối duyệt nhãn, sổ quy ước

| Mã | Việc | Hạn | Tiêu chí xong / đầu ra | Mốc | Trạng thái | Xong lúc / đầu ra |
|---|---|---|---|---|---|---|
| BANH-01 | Dựng CVAT `RoboTaxi_T66`, 11 lớp đúng thứ tự id, màu cố định, bật SAM; đọc guideline | 11:45 27/11 | Cả đội đăng nhập được | M0 | ⬜ | |
| BANH-02 | Chọn 80 khung gold val, gán nháp COCO x-seg, nạp vào CVAT | 13:00 | Task `gold_{n}` | M1 | ⬜ | |
| BANH-03 | **Gold val v0**: 60–80 ảnh, duyệt chéo 100%, xuất `gold_val_v0.zip` + sha256 | 15:30 | File trên Drive + ghi log | M3 | ⬜ | |
| BANH-04 | Gold val tăng dần về 150; họp hiệu chỉnh lúc 13:00 và 19:30 | 18:00 | Tiến độ báo ở họp 18:00 | M4 | ⬜ | |
| BANH-05 | **Gold val v1 = 150 ảnh, khóa phiên bản**; mỗi lớp ≥ 15 instance | 21:00 | `gold_val_v1.zip` + sha256; không đổi sau G2 | M6 | ⬜ | |
| BANH-06 | Phân việc duyệt; QA chéo 10% mỗi task; trả task có > 15% ảnh lỗi | 00:30 28/11 | Bảng thống kê duyệt (số ảnh, phút/ảnh, tỉ lệ sửa, tỉ lệ lỗi) | M7 | ⬜ | |
| BANH-07 | Duyệt lô cuối, chốt thống kê; ngủ từ 02:00 | 01:30 | Số liệu cuối gửi Lộc | M8 | ⬜ | |
| BANH-08 | Slide: guideline, quy trình duyệt, số liệu QA | 09:30 | Gửi Hưng | M12 | ⬜ | |

---

## Quang Hưng · huấn luyện, ablation, teacher, biên tập slide

| Mã | Việc | Hạn | Tiêu chí xong / đầu ra | Mốc | Trạng thái | Xong lúc / đầu ra |
|---|---|---|---|---|---|---|
| HUNG-01 | Tạo dataset dữ liệu mẫu trên Kaggle, chạy thử notebook BTC | 11:45 27/11 | Notebook chạy tới Bước 3 | M0 | ⬜ | |
| HUNG-02 | Notebook chạy trọn Bước 0–6, `TEAM_ID=71`, ra `submission.zip` hợp lệ | 13:00 | **G0** | M1 | ⬜ | |
| HUNG-03 | Baseline A0 (Vistas) → 📤 **#1** | 15:30 | Có điểm leaderboard; ghi log | M3 | ⬜ | |
| HUNG-04 | `notebooks/train_t66.ipynb` (preset ablation/final, ô gom output về `/kaggle/working/{RUN_NAME}`) + `tools/train_teacher.py`. **Không sửa Bước 4** | 18:00 | Commit lên repo | M4 | ⬜ | |
| HUNG-05 | Ablation A1 (Vistas) và A2 (Vistas + IDD) song song | 18:00 | Đang chạy trên 2 tài khoản | M4 | ⬜ | |
| HUNG-06 | Ablation A3 (+COCO đã gán bù Sign) | 19:30 | Đang chạy | M5 | ⬜ | |
| HUNG-07 | Teacher `yolo26l/m-seg` @ 1024 bắt đầu huấn luyện | 19:30 | `teacher.pt` dự kiến xong khoảng 21:00–22:00 | M5 | ⬜ | |
| HUNG-08 | Kết quả A1–A3 trên gold val, AP từng lớp | 21:00 | Bảng trong log | M6 | ⬜ | |
| HUNG-09 | 📤 #2–#3; kiểm tra tương quan gold val ↔ public | 21:00 | Ghi kết luận | M6 | ⬜ | |
| HUNG-10 | Ablation A4 / A5; 📤 #4–#6 | 00:30 28/11 | Bảng trong log | M7 | ⬜ | |
| HUNG-11 | 2–3 cấu hình lượt dài (F1–F3) | 01:30 | Sẵn trong notebook | M8 | ⬜ | |
| HUNG-12 | Khởi chạy F2, F3 trên Kaggle B, C **trước 02:00** rồi ngủ | 02:00 | Đang chạy; `DRIVE_OUTPUT_DIR`/output Kaggle đã đặt | M8 | ⬜ | |
| HUNG-13 | Phân tích lỗi bản tốt nhất, ảnh trước/sau | 09:30 | Ảnh trong Drive cho slide | M10 | ⬜ | |
| HUNG-14 | Biên tập slide hoàn chỉnh | 10:45 | File slide cuối | M12 | ⬜ | |

---

## Đang bị chặn

Ghi mỗi vấn đề một dòng. Lộc xử lý hoặc phân người trong 15 phút.

| Giờ | Mã việc | Ai | Bị chặn bởi gì | Người gỡ | Đã gỡ lúc |
|---|---|---|---|---|---|
| | | | | | |
