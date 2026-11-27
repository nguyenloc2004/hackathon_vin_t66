# Kế hoạch đội: AIInAction RoboTaxi Data Centric Competition

Lead: **Nguyễn Hữu Lộc** · Đội 6 người · `TEAM_ID = 71`

> Tài liệu gốc: [`docs/labeling_guideline.md`](../docs/labeling_guideline.md), [`docs/train_and_export.ipynb`](../docs/train_and_export.ipynb), ảnh chụp website trong `docs/`.
> Tài liệu liên quan: [`class_mapping.md`](class_mapping.md) (ánh xạ public dataset về 11 lớp), [`experiment_log.md`](experiment_log.md) (nhật ký huấn luyện và nộp bài).

---

## 1. Tóm tắt đề

| Mục | Nội dung |
|---|---|
| Bài toán | Instance segmentation ảnh giao thông Việt Nam, 11 lớp cố định (0 Car … 10 Traffic light) |
| Mô hình | **Cố định** `yolo26n-seg`, `imgsz = 640`, Ultralytics `8.4.214`. Đội chỉ cải thiện **dữ liệu** |
| Điểm | `100 × mask mAP@[.5:.95]`, **trung bình đều 11 lớp**, trên bộ test ẩn |
| Nộp bài | `.zip` (≤ 42 MB) chứa đúng một `model.onnx`, xuất bằng Bước 4 của notebook mẫu. **20 lượt**, mỗi lúc chỉ 1 bài trong hàng đợi chấm |
| Chọn bài | Tối đa **2 bài** tính điểm private, chọn trước khi đóng nộp (không chọn thì hệ thống lấy 2 bài gần nhất) |
| Xếp hạng cuối | **50% thứ hạng private + 50% thuyết trình** |
| Bắt buộc lưu | Notebook, zip dữ liệu, `run_report.json`; danh sách mọi public dataset (tên, link, license, số ảnh, cách ánh xạ) |
| Cấm | Nộp trùng checkpoint đội khác (cả hai đội bị trừ 10 điểm) |

### Mốc thời gian (giờ Việt Nam)

| Thời điểm | Sự kiện |
|---|---|
| 09:00 27/11 | Mở nộp bài |
| **10:30 28/11** | **Đóng nộp bài** |
| 11:00 28/11 | Công bố bảng xếp hạng private |
| 11:30 28/11 | Thuyết trình giải pháp |

### Hệ quả cho chiến lược

- mAP trung bình đều theo lớp, nên **lớp hiếm quan trọng ngang Car**: Cyclist, Bicycle, Other vehicle, Bus, Traffic light phải có đủ ví dụ.
- Điểm tính trên IoU từ 0,50 đến 0,95, nên **độ sát biên của mask** ảnh hưởng lớn. Polygon cẩu thả làm mất điểm nhiều hơn so với việc thiếu ảnh.
- Bộ test ẩn nhiều khả năng là **ảnh đường phố Việt Nam thật** (cạnh dài khoảng 1280 px), nên dữ liệu phải gần miền này. 10 ảnh mẫu chỉ là ảnh vẽ, không dùng huấn luyện.
- Hệ thống chấm bỏ qua mask nhỏ hơn 64 px ở thang 640 và chỉ giữ tối đa 100 dự đoán mỗi ảnh.

---

## 2. Chiến lược dữ liệu

### 2.1 Public dataset (nền dữ liệu lớn)

| Ưu tiên | Dataset | Vì sao | Rủi ro |
|---|---|---|---|
| 1 | **Mapillary Vistas v2** | Instance mask cho cả biển báo (mặt trước/sau) và đèn giao thông, có nhãn rider, ảnh đa dạng toàn cầu | Cần đăng ký; nhiều vùng "group" |
| 2 | **IDD (India Driving Dataset)** | Cảnh hỗn loạn giống Việt Nam: nhiều xe máy, autorickshaw, xe tải | Đăng ký duyệt tay, có thể chậm |
| 3 | **COCO 2017** (chỉ ảnh đường phố) | Polygon sẵn, nhiều person/motorcycle/bicycle | Không gán đủ biển báo, phải ghép rider; mask thô |
| 4 | **Cityscapes** | Polygon chất lượng cao | Duyệt tài khoản lâu; biển báo và đèn **không có instance** |
| 5 | **Roboflow Universe** (giao thông Việt Nam, có polygon) | Đúng miền | Chất lượng nhãn không đồng đều, license khác nhau |

**Đăng ký Mapillary, IDD, Cityscapes ngay trong giờ đầu.**

### 2.2 Ánh xạ về 11 lớp (chi tiết trong [`class_mapping.md`](class_mapping.md))

Mọi converter phải áp dụng cùng các quy tắc sau:

1. **Ghép rider + xe**: rider chồng lấn với motorcycle/bicycle thì gộp thành **một** instance `Motorcyclist`/`Cyclist` (hợp mask). Xe không có rider là `Motorcycle`/`Bicycle`. Rider không tìm được xe thì **bỏ ảnh** (hoặc đánh dấu để người xem lại).
2. **Mask không chồng nhau**: raster hóa theo thứ tự từ xa đến gần (vật phía trước ghi đè), rồi chuyển lại thành polygon. Một instance bị tách mảnh thì ghi nhiều polygon trên **cùng một dòng nhãn** (nối các polygon lại theo cách Ultralytics hỗ trợ) hoặc giữ mảnh lớn nhất.
3. **Lọc vật nhỏ**: bỏ đối tượng có cạnh ngắn của bbox nhỏ hơn **1,25% cạnh dài của ảnh**.
4. **Biển báo** chỉ phủ mặt biển, không phủ cột; lấy cả mặt trước và mặt sau. **Đèn giao thông** phủ cả hộp đèn.
5. **Đầu kéo** là `Truck`, **rơ-moóc/container** là `Other vehicle`.
6. **Vùng crowd/group** (person-group, vehicle-group): nếu chiếm diện tích lớn thì **bỏ ảnh**, vì các đối tượng không được gán nhãn sẽ dạy mô hình coi chúng là nền.
7. **Dataset thiếu lớp** (ví dụ COCO không gán đủ biển báo): hoặc chỉ lấy ảnh không có lớp bị thiếu, hoặc **gán nhãn giả bằng teacher** cho các lớp thiếu trước khi đưa vào.
8. Resize ảnh về cạnh dài **1280 px** để giảm dung lượng và tăng tốc huấn luyện.

### 2.3 Dữ liệu Việt Nam tự thu thập

- Nguồn: video camera hành trình hoặc đi đường ở Hà Nội và TP.HCM (ghi lại nguồn và license), ảnh tự chụp.
- Trích khoảng 1 khung hình mỗi 2–3 giây, **loại ảnh gần trùng** bằng perceptual hash.
- Ưu tiên cảnh khó: **đêm, mưa, ngược sáng, đông xe máy, ngã tư có đèn, xe ba gác/xích lô, xe đầu kéo**.
- Mục tiêu: 2.000–5.000 ảnh, gán nhãn nháp tự động rồi người duyệt (mục 2.4).

### 2.4 Gán nhãn nháp: teacher → người duyệt

1. Huấn luyện **teacher** lớn (`yolo26x-seg` hoặc `yolo26l-seg`, có thể `imgsz` 1024) trên dữ liệu public đã gộp. Teacher chỉ dùng để làm dữ liệu, **không nộp**.
2. Teacher dự đoán trên ảnh Việt Nam; với các lớp yếu có thể dùng thêm **SAM 3** (prompt bằng chữ) hoặc Grounding DINO + SAM 2.
3. Đưa nhãn nháp vào **CVAT**. Người duyệt sửa lớp, sửa biên, thêm đối tượng bị sót, xóa nhãn sai. **Ưu tiên ảnh mà mô hình kém tự tin** (active learning).
4. Duyệt chéo: ít nhất 10% ảnh do người thứ hai kiểm tra lại theo checklist ở mục 5.

### 2.5 Bộ val chuẩn (gold val)

- Khoảng **200 ảnh Việt Nam thật**, **không trùng** nguồn video với dữ liệu train (lấy từ video khác để tránh rò rỉ).
- Gán **tay, cẩn thận, đúng guideline**; có thể dùng SAM trong CVAT để làm biên sát hơn.
- Đây là **thước đo chính** để quyết định giữ hay bỏ dữ liệu. Chỉ có 20 lượt nộp, nên không dùng leaderboard để dò tìm.
- Đặt vào `images/val`, `labels/val` để notebook dùng làm val, thay vì để notebook chia ngẫu nhiên.

### 2.6 Cân bằng lớp và augmentation

- Đếm số instance mỗi lớp (Bước 2 của notebook). Lớp có ít hơn khoảng 500 instance trong train thì **lấy thêm ảnh chứa lớp đó**.
- Thêm khoảng 3–5% **ảnh nền khó** (áp phích, kính phản chiếu, showroom) với nhãn rỗng để giảm báo nhầm.
- Augmentation có thể thử ở Bước 3: `copy_paste=0.1–0.3`, `mixup=0.05`, `close_mosaic=10`, `hsv_v` cao hơn để mô phỏng cảnh đêm. **Chỉ giữ thay đổi nào cải thiện điểm trên gold val.**

---

## 3. Phân công

| Thành viên | Vai trò | Đầu ra |
|---|---|---|
| **Nguyễn Hữu Lộc** (lead) | Pipeline chung, script gộp dữ liệu (`merge.py`), quản lý lượt nộp, quyết định giữ/bỏ dữ liệu | Repo, `datasets/merged_vX.zip`, nhật ký nộp |
| **Minh Thắng** | Mapillary Vistas (+ Cityscapes nếu được duyệt) sang YOLO-seg theo quy ước | `vistas_yolo.zip`, bảng ánh xạ |
| **Bảo Long** | IDD + COCO (lọc ảnh đường phố) + Roboflow Việt Nam | `idd_yolo.zip`, `coco_yolo.zip`, bảng ánh xạ |
| **Đạt** | Thu thập ảnh Việt Nam, loại trùng, gán nhãn nháp (teacher / SAM 3) | 2–5k ảnh Việt Nam có nhãn nháp |
| **Nguyễn Bình Anh** | Dựng CVAT, gán **gold val** khoảng 200 ảnh, điều phối việc duyệt nhãn | `gold_val.zip`, `conventions.md` |
| **Quang Hưng** | Huấn luyện (baseline, ablation, teacher) trên Kaggle/Colab, phân tích lỗi; từ 06:00 làm slide | Bảng kết quả, slide |

- Ai xong việc thì chuyển sang **duyệt nhãn** (đây luôn là việc tốn công nhất).
- GPU: mỗi tài khoản Kaggle có khoảng 30 giờ/tuần (T4×2 hoặc P100), nên 6 tài khoản có thể chạy song song nhiều thí nghiệm. Colab miễn phí dễ ngắt kết nối, nên luôn đặt `DRIVE_OUTPUT_DIR`.

---

## 4. Lịch trình

| Khung giờ | Việc | Phụ trách | Lượt nộp |
|---|---|---|---|
| **27/11 10:30–12:00** | Lập repo + Drive chung; đăng ký Mapillary/IDD/Cityscapes; dựng CVAT; chạy thử notebook `TEAM_ID=71` trên dữ liệu nhỏ để kiểm tra môi trường | Cả đội | 0 |
| **12:00–15:00** | Viết converter Vistas/IDD/COCO; thu ảnh Việt Nam; gán gold val | Thắng, Long, Đạt, Bình Anh | — |
| **~15:00** | **Baseline**: Vistas đã ánh xạ, huấn luyện ngắn | Hưng, Lộc | #1 |
| **15:00–20:00** | Gộp Vistas + IDD (+ COCO); ablation từng nguồn trên gold val; bắt đầu huấn luyện teacher | Lộc, Hưng | #2–#4 |
| **20:00–02:00** | Teacher gán nhãn nháp ảnh Việt Nam; chia ca duyệt; bổ sung lớp hiếm và ảnh nền khó | Đạt + cả đội duyệt | #5–#9 |
| **28/11 02:00–07:00** | Ngủ luân phiên (luôn có 1–2 người trực); chạy huấn luyện dài (60–100 epoch) với bộ dữ liệu tốt nhất | Trực ca | #10–#12 |
| **07:00–09:30** | Thử các biến thể cuối; nộp bản cuối, **không dồn đến phút chót** | Lộc, Hưng | #13–#17 |
| **≤ 10:15** | **Chọn 2 bài**: (a) điểm public cao nhất, (b) tốt nhất trên gold val. Lưu đủ notebook, zip dữ liệu, `run_report.json` | Lộc | 3 dự phòng |
| **06:00–11:30** | Hoàn thiện slide và tập thuyết trình | Hưng + cả đội | — |

### Ngân sách 20 lượt nộp

- Khoảng 12 lượt cho **ablation dữ liệu**: mỗi lần chỉ thay đổi một yếu tố.
- Khoảng 5 lượt cho **bản cuối**.
- **3 lượt dự phòng** (lỗi, nộp lại).
- Trước mỗi lần nộp: phải có điểm gold val và ghi vào [`experiment_log.md`](experiment_log.md).

---

## 5. Quy tắc làm việc

### Dữ liệu và huấn luyện

- Cả đội dùng `TEAM_ID = 71` và cùng một zip dữ liệu cho mỗi phiên bản (`merged_v1`, `merged_v2`, …), lưu trên Drive chung.
- Notebook chỉ sửa **ô tham số** và **Bước 3**. **Không sửa Bước 4** (xuất ONNX và đóng gói).
- Mỗi lượt huấn luyện ghi lại: phiên bản dữ liệu, hash (`run_report.json`), tham số, mAP gold val, AP từng lớp, điểm public (nếu đã nộp).
- **Không chia sẻ checkpoint ra ngoài đội.**

### Checklist duyệt nhãn (theo mục 6 của guideline)

- [ ] Không sót vật nhỏ nhưng vẫn vượt ngưỡng (người đi bộ ở xa, biển báo, đèn giao thông)
- [ ] Mask Motorcyclist/Cyclist phủ **cả xe lẫn người**
- [ ] Mask không chồng nhau; không tràn nền hay bóng đổ
- [ ] Xe đạp điện kiểu xe máy là Motorcycle; xe 9–15 chỗ là Car; xe ba gác là Other vehicle
- [ ] Không gán biển quảng cáo, áp phích, phản chiếu, người ngồi trong ô tô
- [ ] Không tách một vật thành hai instance, không gộp hai vật thành một

### Ca khó

Áp dụng cây quyết định ở mục 5 của guideline. Ca nào vẫn phân vân thì ghi vào `plan/conventions.md` (Bình Anh quản lý) để cả đội gán nhất quán.

---

## 6. Slide thuyết trình (50% điểm)

1. Bài toán và thách thức của giao thông Việt Nam
2. Quy trình dữ liệu: thu thập, ánh xạ, gán nháp, duyệt nhãn, gold val
3. **Bảng các public dataset**: tên, link, license, số ảnh dùng, cách ánh xạ về 11 lớp
4. Label guideline của đội và các ca khó đã quyết định
5. Quy trình duyệt nhãn và kiểm soát chất lượng
6. **Các vòng cải thiện kèm số liệu**: biểu đồ mAP qua từng phiên bản dữ liệu, AP từng lớp
7. Phân tích lỗi: ví dụ trước và sau khi cải thiện
8. Bài học và hướng phát triển

---

## 7. Rủi ro và phương án dự phòng

| Rủi ro | Phương án |
|---|---|
| Cityscapes/IDD duyệt tài khoản chậm | Dựa vào Mapillary + COCO + dữ liệu Việt Nam tự gán |
| Colab ngắt kết nối giữa chừng | Dùng Kaggle cho các lượt dài; `DRIVE_OUTPUT_DIR` để lưu `best.pt` |
| Hàng đợi chấm đông vào sáng 28/11 | Bản cuối phải nộp **trước 09:30** |
| Dữ liệu public kéo điểm xuống (lệch miền) | Ablation từng nguồn trên gold val; tăng tỉ trọng ảnh Việt Nam |
| Bài nộp bị từ chối (sai định dạng) | Luôn chạy đủ Bước 4–5 của notebook trước khi nộp |
| Mất dữ liệu hoặc không tái lập được | Lưu zip dữ liệu từng phiên bản, `run_report.json`, notebook trên Drive |

---

## 8. Việc cần làm ngay (30 phút đầu)

- [ ] **Lộc**: tạo repo + thư mục Drive chung, chia quyền, tạo sheet nhật ký
- [ ] **Minh Thắng, Bảo Long**: đăng ký Mapillary Vistas, IDD, Cityscapes; tải COCO annotations
- [ ] **Quang Hưng**: chạy notebook mẫu với `TEAM_ID=71` trên Kaggle để kiểm tra môi trường và Bước 4
- [ ] **Đạt**: lập danh sách nguồn video/ảnh Việt Nam, viết script trích khung hình + loại trùng
- [ ] **Nguyễn Bình Anh**: dựng CVAT với 11 lớp đúng thứ tự id, tạo `plan/conventions.md`
