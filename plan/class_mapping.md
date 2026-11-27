# Ánh xạ public dataset về 11 lớp

> Tên nhãn dưới đây là **bản nháp**: người phụ trách mỗi dataset phải đối chiếu với file config thật (ví dụ `config_v2.0.json` của Mapillary) và cập nhật bảng này trước khi chạy converter.

## Bảng 11 lớp của cuộc thi

| ID | Lớp | ID | Lớp |
|---|---|---|---|
| 0 | Car | 6 | Bicycle |
| 1 | Bus | 7 | Cyclist |
| 2 | Truck | 8 | Pedestrian |
| 3 | Other vehicle | 9 | Sign |
| 4 | Motorcycle | 10 | Traffic light |
| 5 | Motorcyclist | | |

## Quy tắc ghép rider + xe (dùng chung cho mọi dataset)

Thuật toán chi tiết (điểm ghép, ngưỡng, unit test): [`data_pipeline.md`](data_pipeline.md) §3.3.

1. Với mỗi rider, tìm motorcycle/bicycle có **diện tích chồng lấn bbox lớn nhất** (hoặc mask kề nhau trong khoảng vài pixel).
2. Có xe: **hợp mask** rider + xe thành `Motorcyclist` (5) hoặc `Cyclist` (7). Một xe có thể có nhiều rider: gộp tất cả vào cùng instance.
3. Xe không có rider: `Motorcycle` (4) hoặc `Bicycle` (6).
4. Rider không tìm được xe: **bỏ ảnh** hoặc đưa vào danh sách để người kiểm tra.

## Mapillary Vistas v2 (Minh Thắng)

| Nhãn gốc | Lớp đích |
|---|---|
| `object--vehicle--car` | 0 Car |
| `object--vehicle--bus` | 1 Bus |
| `object--vehicle--truck` | 2 Truck |
| `object--vehicle--trailer`, `object--vehicle--caravan`, `object--vehicle--other-vehicle`, `object--vehicle--wheeled-slow` | 3 Other vehicle |
| `object--vehicle--motorcycle` | 4, hoặc 5 nếu ghép với rider |
| `object--vehicle--bicycle` | 6, hoặc 7 nếu ghép với rider |
| `human--rider--motorcyclist` | ghép vào 5 |
| `human--rider--bicyclist` | ghép vào 7 |
| `human--rider--other-rider` | xem từng ca (thường ghép vào Other vehicle) |
| `human--person--individual` | 8 Pedestrian |
| `object--traffic-sign--front`, `--back`, `--direction-*`, `--temporary-*` | 9 Sign |
| `object--traffic-sign--information-parking`, `--ambiguous` | xem xét; mặc định **bỏ đối tượng** |
| `object--traffic-light--*` (general, pedestrians, cyclists…) | 10 Traffic light |
| `object--vehicle--on-rails` | 3 Other vehicle (hoặc bỏ, ghi lại quyết định) |
| `human--person--person-group`, `object--vehicle--vehicle-group` | vùng lớn thì **bỏ ảnh** |

## IDD Segmentation (Bảo Long)

| Nhãn gốc | Lớp đích |
|---|---|
| `car` | 0 Car |
| `bus` | 1 Bus |
| `truck` | 2 Truck |
| `autorickshaw`, `trailer`, `caravan`, `vehicle fallback` | 3 Other vehicle |
| `motorcycle` | 4, hoặc 5 nếu ghép với rider |
| `bicycle` | 6, hoặc 7 nếu ghép với rider |
| `rider` | ghép với xe gần nhất |
| `person` | 8 Pedestrian |
| `traffic sign` | 9 Sign (chỉ lấy biển giao thông, kiểm tra lại) |
| `traffic light` | 10 Traffic light |
| `animal` | bỏ |

## COCO 2017 (Bảo Long), chỉ ảnh đường phố

| Nhãn gốc | Lớp đích |
|---|---|
| `car` | 0 Car |
| `bus` | 1 Bus |
| `truck` | 2 Truck (kiểm tra: COCO gộp cả bán tải vào truck) |
| `motorcycle` + `person` chồng lấn | 5 Motorcyclist |
| `motorcycle` | 4 Motorcycle |
| `bicycle` + `person` chồng lấn | 7 Cyclist |
| `bicycle` | 6 Bicycle |
| `person` | 8 Pedestrian |
| `stop sign` | 9 Sign (**thiếu các loại biển khác**: cần gán nhãn giả bằng teacher) |
| `traffic light` | 10 Traffic light |
| các nhãn `iscrowd=1` | bỏ ảnh nếu vùng lớn |

## Cityscapes (Minh Thắng, nếu được duyệt)

| Nhãn gốc | Lớp đích |
|---|---|
| `car` | 0 Car |
| `bus` | 1 Bus |
| `truck` | 2 Truck |
| `trailer`, `caravan`, `train` | 3 Other vehicle |
| `motorcycle` / `bicycle` + `rider` | 4/5, 6/7 theo quy tắc ghép |
| `person` | 8 Pedestrian |
| `traffic sign`, `traffic light` | **Không có instance**: tách thành phần liên thông từ polygon, hoặc gán nhãn giả bằng teacher |

## Nhật ký dataset (đưa vào slide)

| Dataset | Link | License | Số ảnh dùng | Ghi chú lọc |
|---|---|---|---|---|
| Mapillary Vistas v2 | | CC BY-NC-SA 4.0 (kiểm tra lại) | | |
| IDD | | (kiểm tra lại) | | |
| COCO 2017 | | CC BY 4.0 (annotations) | | |
| Cityscapes | | (kiểm tra lại) | | |
| Roboflow … | | | | |
| Ảnh Việt Nam từ video YouTube | xem `vn_sources.csv` | theo từng video (ưu tiên CC) | | số video, số kênh |
| Bộ ảnh Việt Nam công khai (Kaggle/Roboflow/HF) | | | | chỉ dùng ảnh, nhãn làm lại |
