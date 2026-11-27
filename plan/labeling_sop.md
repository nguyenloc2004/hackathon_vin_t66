# Quy trình gán nhãn và kiểm soát chất lượng (SOP)

**Mọi thành viên phải đọc trước khi chạm vào CVAT:** [`docs/labeling_guideline.md`](../docs/labeling_guideline.md) và [`conventions.md`](conventions.md).

## 1. Thiết lập CVAT (Bình Anh)

- Một project `RoboTaxi_T66` với **11 nhãn đúng thứ tự id**: Car, Bus, Truck, Other vehicle, Motorcycle, Motorcyclist, Bicycle, Cyclist, Pedestrian, Sign, Traffic light. Loại công cụ: **polygon**.
- Màu nhãn cố định (dễ nhận ra lỗi): Motorcyclist hồng, Motorcycle xanh ngọc, Cyclist cam, Bicycle xanh lá, Pedestrian xanh dương nhạt.
- Bật công cụ AI hỗ trợ: **Segment Anything** (interactor: click điểm dương/âm ra polygon).
- Mỗi task tối đa **40 ảnh**, đặt tên `gold_{n}` / `review_{n}`; giao cho một người gán, một người duyệt chéo.
- Xuất: định dạng **Ultralytics YOLO Segmentation**; kiểm tra lại thứ tự lớp trong `data.yaml` xuất ra (nhiều công cụ đánh số lại).

## 2. Hai loại công việc

| Loại | Mục đích | Mức độ cẩn thận | Thời gian dự kiến/ảnh |
|---|---|---|---|
| **Gold val** | Thước đo thay cho test ẩn | Tối đa: biên sát từng pixel có thể, không sót vật nào vượt ngưỡng | 5–8 phút |
| **Duyệt nhanh** (nhãn nháp train) | Sửa lỗi lớn của nhãn nháp | Sửa lớp sai, thêm vật bị sót, xóa nhãn sai, sửa biên lệch rõ | 1–2 phút |

## 3. Gold val: đặc tả

- **Kích thước**: v0 = 60–80 ảnh (trước 15:30), v1 = **150 ảnh** (trước 21:00). Không đổi gold val sau G2 (để các kết quả so sánh được).
- **Nguồn**: video/bộ ảnh Việt Nam sẵn có, **khác kênh YouTube / khác bộ dữ liệu** so với train (ghi rõ cột “dùng cho = gold” trong `vn_sources.csv`). Ưu tiên góc dashcam ô tô giống robotaxi.
- **Phân tầng** (mục tiêu):

| Điều kiện | Tỉ lệ |
|---|---|
| Ban ngày, thường | 50% |
| Ban đêm | 20% |
| Mưa / u ám / ngược sáng | 10% |
| Ngã tư có đèn, đông đúc | 20% |

- **Phủ lớp**: mỗi lớp ≥ 15 instance. Nếu thiếu (thường là Cyclist, Other vehicle, Bus), chủ động tìm khung hình có lớp đó.
- **Quy trình**: gán nháp bằng mô hình → người A sửa kỹ → người B duyệt toàn bộ → Bình Anh kiểm tra ngẫu nhiên 20%.
- **Khóa phiên bản**: xuất `gold_val_v1.zip` + sha256, ghi vào `experiment_log.md`.

## 4. Thao tác chuẩn trong một ảnh

1. **Quét toàn ảnh trước** theo thứ tự: xe lớn → xe máy/xe đạp → người → biển báo → đèn. Phóng to các góc ảnh và vùng xa.
2. **Với mỗi vật**: xác định lớp theo cây quyết định (guideline mục 5) → kiểm tra ngưỡng kích thước (cạnh ngắn ≥ ~1,25% cạnh dài ảnh; ở ảnh 1280 px là 16 px) → vẽ / sửa polygon.
3. **Biên**: bám sát vật, không phủ bóng đổ, ánh đèn pha, phản chiếu. Sai số 1–2 px là chấp nhận được.
4. **Che khuất**: vật phía trước giữ pixel; vật phía sau chỉ phủ phần nhìn thấy. Vật bị cột/dây che vẫn là **một** instance (dùng nhóm polygon trong CVAT).
5. **Motorcyclist**: một polygon phủ cả xe + mọi người ngồi trên + hàng buộc trên xe. Kiểm tra phần bánh xe và chân người có trong mask.
6. **Sign**: chỉ mặt biển, không cột. Mỗi tấm biển một instance, kể cả biển phụ.
7. **Traffic light**: cả hộp đèn. Mỗi hộp một instance.
8. Không chắc → hỏi trong kênh `#quy-uoc`, Bình Anh/Lộc chốt trong 10 phút và ghi vào `conventions.md`.

## 5. Checklist trước khi bấm “Xong” một ảnh

- [ ] Không sót vật vượt ngưỡng (đặc biệt: người đi bộ xa, biển báo, đèn, xe máy bị che một nửa)
- [ ] Motorcyclist/Cyclist phủ **cả xe và người**; xe không người là Motorcycle/Bicycle
- [ ] Không mask nào chồng lên mask khác
- [ ] Không gán: biển quảng cáo, áp phích, người trong ô tô, phản chiếu, xe đẩy tay, động vật
- [ ] Đúng lớp cho ca dễ nhầm: xe đạp điện kiểu xe máy, xe 9–15 chỗ, xe ba gác, đầu kéo + rơ-moóc
- [ ] Không tách một vật thành hai instance, không gộp hai vật thành một

## 6. Kiểm soát chất lượng

| Cơ chế | Tần suất | Ai |
|---|---|---|
| Duyệt chéo toàn bộ gold val | 100% | Người thứ hai |
| Kiểm tra ngẫu nhiên ảnh duyệt nhanh | 10% mỗi task | Bình Anh |
| `qa_check.py --expect configs/qa/vn_reviewed.yaml` trên mỗi lô xuất từ CVAT (gold: `gold_val.yaml --against ...`) | Mỗi lô | Người xuất |
| `label_audit.py` (teacher soi gold val + ảnh đã duyệt), nghi vấn quay lại hàng duyệt | Khi có teacher, rồi mỗi lô lớn | Hưng chạy, Bình Anh phân việc |
| Đo **tỉ lệ ảnh có lỗi** trong mẫu kiểm tra | Mỗi task | Bình Anh ghi lại |
| Họp hiệu chỉnh (calibration) 10 phút: cả đội cùng gán 3 ảnh khó, so sánh | 13:00 và 19:30 | Bình Anh chủ trì |

Ngưỡng: task có **> 15% ảnh lỗi** trong mẫu kiểm tra → trả lại người gán sửa toàn bộ task. Mẫu kiểm tra 10% có thể lấy từ `qa/sample.csv` (cột `verdict`) để `qa_check` tự tính tỉ lệ này; gold val dùng ngưỡng 5%.

**Số liệu cần ghi cho slide**: số ảnh đã duyệt, thời gian trung bình/ảnh, tỉ lệ nhãn nháp bị sửa (thêm/xóa/đổi lớp/sửa biên), tỉ lệ lỗi trong kiểm tra chéo.

## 7. Thứ tự ưu tiên của hàng duyệt

1. Gold val (cho tới khi đủ 150 ảnh)
2. Ảnh Việt Nam có lớp hiếm (Cyclist, Other vehicle, Bus, Bicycle, Truck đầu kéo)
3. Ảnh ban đêm / mưa
4. Ảnh mà mô hình hiện tại kém tự tin nhất (nhiều dự đoán conf 0,15–0,5)
5. Ảnh lỗi nhiều nhất từ Bước 5 của notebook (phân tích lỗi trên val)
6. Phần còn lại
