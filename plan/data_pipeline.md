# Đặc tả kỹ thuật pipeline dữ liệu

Mục tiêu: mọi nguồn dữ liệu đi qua **cùng một bộ quy tắc**, cho ra nhãn YOLO-seg đúng quy ước của bộ chấm, và mỗi phiên bản dataset tái lập được từ một file recipe.

## 1. Cấu trúc repo và Drive

```
hackathon_vin_t66/
├── docs/                      # đề bài (không sửa)
├── plan/                      # kế hoạch
├── tools/
│   ├── common.py              # CLASSES, mask→polygon, lọc kích thước, ghép rider, ghi nhãn
│   ├── convert_vistas.py
│   ├── convert_idd.py
│   ├── convert_coco.py
│   ├── extract_frames.py      # video → khung hình
│   ├── dedup.py               # loại ảnh trùng/gần trùng (phash)
│   ├── pseudo_label.py        # gán nháp bằng teacher / COCO model / SAM 3
│   ├── cvat_io.py             # đẩy nhãn nháp vào CVAT, lấy nhãn đã duyệt về
│   ├── qa_check.py            # kiểm tra nhãn tự động
│   ├── visualize.py           # vẽ mask lên ảnh để xem bằng mắt
│   └── build_dataset.py       # recipe → zip + manifest
├── configs/
│   └── recipes/merged_v1.yaml ...
└── data/                      # .gitignore, dữ liệu thật nằm trên Drive
```

Drive chung:

```
RoboTaxi_T66/
├── raw/            # dữ liệu gốc tải về (vistas, idd, coco, video)
├── converted/      # mỗi nguồn sau khi chuyển: images/ labels/ stats.json preview/
├── cvat_exports/   # nhãn đã duyệt xuất từ CVAT
├── gold_val/       # gold_val_v0.zip, v1.zip ...
├── datasets/       # merged_vX.zip + manifest
└── runs/           # best.pt, submission.zip, run_report.json theo tên lượt chạy
```

Quy ước tên: `merged_v{N}`, lượt chạy `R{NN}_{recipe}_{ghi-chú}` (ví dụ `R07_merged_v2_cp02`).

## 2. Định dạng đầu ra (bắt buộc)

- Ảnh `.jpg`, **cạnh dài 1280 px** (resize giữ tỉ lệ, không phóng to ảnh nhỏ hơn), JPEG quality 92.
- Mỗi ảnh một file `.txt` cùng tên; mỗi dòng: `class x1 y1 x2 y2 ...` chuẩn hóa về [0, 1], ≥ 3 điểm.
- Ảnh không có đối tượng: file `.txt` rỗng (ảnh nền).
- Tên file có tiền tố nguồn để tránh trùng: `vistas_xxx.jpg`, `idd_xxx.jpg`, `coco_xxx.jpg`, `vn_{video}_{frame}.jpg`.
- Mỗi nguồn kèm `stats.json`: số ảnh, số instance mỗi lớp, số ảnh bị loại và lý do.

## 3. Quy tắc chung trong `common.py`

### 3.1 Từ mask sang polygon

1. Raster hóa **mọi** instance của ảnh lên một bản đồ instance duy nhất (`H×W`, int32), **theo thứ tự từ sau ra trước** (vật phía trước ghi đè). Kết quả tự động thỏa “mask không chồng nhau”.
   - Vistas: dùng trực tiếp ảnh instance PNG (đã không chồng nhau).
   - IDD / Cityscapes: polygon trong JSON được vẽ theo thứ tự xuất hiện (sau đè trước) → vẽ đúng thứ tự đó.
   - COCO: mask chồng nhau → sắp xếp theo diện tích **giảm dần** rồi vẽ (vật nhỏ thường ở trước), sau đó cho người xem mẫu.
2. Với mỗi instance: `cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_NONE)` → `approxPolyDP(epsilon = 0.7 px ở thang 1280)`.
3. Bỏ mảnh có diện tích < 0,15% diện tích bbox của instance hoặc < 20 px².
4. Nếu còn nhiều mảnh (vật bị cột/dây che): nối thành **một dòng nhãn** bằng `ultralytics.data.converter.merge_multi_segment` (cách Ultralytics dùng khi chuyển COCO). Không tách thành 2 instance.
5. Lỗ (holes) bị bỏ qua (YOLO-seg không biểu diễn được), chấp nhận.

### 3.2 Lọc kích thước (theo guideline mục 2.3)

- Tính bbox **phần nhìn thấy** sau khi raster hóa.
- Bỏ instance nếu `min(w, h) < 0.0125 × max(W, H)` (≈ 16 px ở cạnh dài 1280).
- Không cần bỏ ảnh; chỉ bỏ instance.

### 3.3 Ghép rider với xe (quan trọng nhất)

Đầu vào: danh sách rider `R` và xe hai bánh `V` (motorcycle, bicycle) **sau** khi raster hóa.

```
với mỗi cặp (r, v):
    điểm(r, v) = |dilate(mask_r, 7px) ∩ mask_v| / min(|mask_r|, |mask_v|)
    điều kiện phụ: tâm-đáy bbox của r nằm trong bbox của v mở rộng 30%
gán tham lam theo điểm giảm dần, mỗi rider ≤ 1 xe, một xe nhận nhiều rider
ngưỡng ghép: điểm ≥ 0.05
```

- Xe có ≥ 1 rider → instance mới = hợp mask (xe + mọi rider), lớp `Motorcyclist` (5) hoặc `Cyclist` (7).
- Xe không rider → `Motorcycle` (4) / `Bicycle` (6).
- Rider không ghép được → **loại cả ảnh** (ghi vào `stats.json`), vì giữ lại sẽ tạo nhãn sai quy ước.
- Unit test bắt buộc: 10 ảnh khó (2–3 người trên xe, xe đứng sát nhau, xe đạp cạnh xe máy). Xem bằng mắt trước khi chạy toàn bộ.

### 3.4 Loại ảnh

Loại ảnh nếu một trong các điều kiện sau đúng:

- Vùng “group/crowd” (person-group, vehicle-group, `iscrowd=1`) chiếm > 1% diện tích ảnh.
- Có rider không ghép được.
- Ảnh có > 100 instance sau lọc (vượt `max_det` của bộ chấm; nhãn kiểu này dạy mô hình điều không đạt được) → giữ tối đa 5% số ảnh như vậy.
- Ảnh gần trùng với gold val (phash, khoảng Hamming ≤ 4).

## 4. Converter từng nguồn

Ánh xạ nhãn chi tiết: [`class_mapping.md`](class_mapping.md).

| Nguồn | Đọc gì | Lưu ý riêng |
|---|---|---|
| **Vistas v2** | `v2.0/instances/*.png` (giá trị pixel: `label_id * 256 + instance_id`, kiểm tra lại trong README của dataset) + `config_v2.0.json` | Ảnh gốc rất lớn: resize trước khi tìm contour để nhanh hơn. Chọn ảnh theo điểm ưu tiên (§5) |
| **IDD** | `gtFine/*/*_gtFine_polygons.json` (định dạng kiểu Cityscapes) | Polygon vẽ theo thứ tự; nhãn `rider` cần ghép; `autorickshaw` → Other vehicle |
| **COCO** | `instances_train2017.json` | Chỉ lấy ảnh có ≥ 3 đối tượng thuộc nhóm giao thông và không có lớp ngoài trời bất thường; COCO thiếu nhãn biển báo (chỉ có `stop sign`) → **bắt buộc** gán nháp lớp Sign bằng teacher trước khi đưa vào, nếu không thì chỉ lấy ảnh không thấy biển báo |
| **Roboflow** | Xuất định dạng YOLOv8 segmentation | Đọc `data.yaml` của bộ đó để ánh xạ id; xem 30 ảnh để kiểm tra quy ước Motorcyclist |

**Cổng chất lượng mỗi nguồn (bắt buộc trước khi gộp):** chạy `qa_check.py` + `visualize.py` 30 ảnh ngẫu nhiên → người phụ trách và Lộc cùng xem → ghi “ĐẠT/KHÔNG ĐẠT” vào `experiment_log.md`.

## 5. Chọn ảnh có ưu tiên

Không cần dùng hết Vistas/IDD (giới hạn bởi thời gian huấn luyện). Chọn ảnh có giá trị cao nhất:

```
trọng số lớp w_c = (N_max / N_c) ^ 0.5          # N_c: số instance lớp c trong toàn nguồn
điểm ảnh = Σ_c w_c × min(n_c(ảnh), 10)
          + 2 × [ảnh có Motorcyclist hoặc Cyclist]
          + 1 × [ảnh có Traffic light hoặc Sign]
lấy top-K theo điểm, cộng thêm 20% ảnh ngẫu nhiên (để giữ phân phối tự nhiên)
```

## 6. Ảnh Việt Nam: từ video tới nhãn

1. **Thu thập** (đội không tự quay):
   - Ghi mọi nguồn vào `vn_sources.csv`: `id, link, kênh/bộ dữ liệu, thành phố, điều kiện (ngày/đêm/mưa), góc nhìn, license, dùng cho (train/gold)`.
   - Video YouTube tải bằng `yt-dlp -f "bv*[height<=1080]" <link>` (chỉ lấy hình, không cần tiếng).
   - Bộ ảnh công khai (Kaggle, Roboflow, Hugging Face): chỉ lấy ảnh, bỏ nhãn gốc nếu không theo quy ước của cuộc thi.
2. **Trích khung** (`extract_frames.py`):
   - Lấy 1 khung mỗi 2 giây.
   - Bỏ đoạn đầu/cuối video (intro, logo kênh).
   - Bỏ khung mờ (phương sai Laplacian thấp).
   - Bỏ khung có chữ phủ lớn.
   - Tối đa **150 khung/video**.
3. **Lọc trùng** (`dedup.py`): phash, khoảng Hamming ≤ 6 trong cùng video → giữ 1; chạy thêm lọc trùng **giữa các video** (kênh hay đăng lại cùng cảnh).
4. **Tách nguồn gold val theo kênh**: video của gold val lấy từ kênh/bộ dữ liệu **không dùng cho train**, để tránh rò rỉ.
5. **Gán nháp đợt 1** (`pseudo_label.py`, chạy ngay):
   - Mô hình COCO pretrained lớn (`yolo26x-seg.pt` hoặc tương đương) cho person, car, bus, truck, motorcycle, bicycle, traffic light → ánh xạ + ghép rider như §3.3.
   - **SAM 3** (prompt bằng chữ) cho: `traffic sign`, `cargo tricycle`, `cyclo`, `trailer`, `excavator`, `crane truck`.
6. **Gán nháp đợt 2** (sau khi có teacher): teacher 11 lớp suy luận với TTA (`augment=True`), `imgsz` 1024.
7. **Chính sách tin cậy** cho nhãn nháp:

| Trường hợp | Xử lý |
|---|---|
| Mọi dự đoán của ảnh có conf ≥ 0,5 hoặc < 0,15 | Ảnh dùng được **không cần duyệt** (giữ dự đoán ≥ 0,5) |
| Có dự đoán 0,15–0,5 | Ảnh vào **hàng duyệt**, chưa được dùng cho tới khi duyệt |
| Ảnh có lớp hiếm (Cyclist, Other vehicle, Bus, Bicycle) | **Ưu tiên duyệt** |
| Ảnh đêm/mưa | Ưu tiên duyệt (teacher kém nhất ở đây) |

8. Duyệt trong CVAT theo [`labeling_sop.md`](labeling_sop.md), xuất về, chạy lại `common.py` (raster theo thứ tự, lọc kích thước) để chuẩn hóa.

## 7. Dựng dataset từ recipe

`configs/recipes/merged_v2.yaml` (ví dụ):

```yaml
name: merged_v2
seed: 71
image_long_side: 1280
val:
  source: gold_val_v1            # luôn dùng gold val, không để notebook chia ngẫu nhiên
train:
  - {source: vistas,      max_images: 4000, select: priority}
  - {source: idd,         max_images: 3000, select: priority}
  - {source: vn_reviewed, repeat: 2}       # nhân bản để tăng trọng số miền Việt Nam
  - {source: vn_pseudo_confident, max_images: 1500}
  - {source: hard_negatives, max_images: 200}
exclude_near_duplicates_of_val: true
```

`build_dataset.py` xuất:

- `merged_v2.zip` với cấu trúc `images/{train,val}`, `labels/{train,val}` (notebook nhận cấu trúc đã chia sẵn).
- `merged_v2.manifest.json`: recipe, danh sách file, số instance mỗi lớp theo nguồn, sha256.

Lưu ý: ảnh nhân bản (`repeat: 2`) đặt tên `..._r1.jpg`; Bước 2 của notebook sẽ báo “ảnh trùng trong train”, đây là chủ ý, chỉ cần chắc chắn **không** có rò rỉ train/val.

## 8. `qa_check.py`: kiểm tra tự động

| Kiểm tra | Mức |
|---|---|
| id lớp ngoài 0–10, < 3 điểm, tọa độ ngoài [0, 1] | Lỗi (dừng) |
| Hai instance chồng nhau > 2% diện tích nhỏ hơn | Lỗi |
| Instance dưới ngưỡng kích thước | Cảnh báo (tự loại) |
| Polygon tự cắt / diện tích 0 | Lỗi |
| Ảnh > 100 instance | Cảnh báo |
| Ảnh train gần trùng ảnh val | Lỗi |
| Phân phối lớp: lớp < 400 instance trong train | Cảnh báo, báo Lộc |
| Motorcyclist có mask chỉ là người (tỉ lệ cao/rộng bất thường, không có phần bánh xe ở đáy) | Cảnh báo, đưa vào danh sách xem |

Bước 2 của notebook là lớp kiểm tra thứ hai, không thay thế `qa_check.py`.
