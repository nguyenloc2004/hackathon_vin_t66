# tools/: bộ công cụ dữ liệu của đội

Mọi nguồn dữ liệu đi qua **cùng một bộ quy tắc** trong [`common.py`](common.py), cho ra nhãn YOLO-seg đúng quy ước của cuộc thi:

- mask **không chồng nhau** (vật phía trước giữ pixel);
- **rider + xe = một instance** Motorcyclist / Cyclist (xích lô, ba gác có người → Other vehicle);
- bỏ vật có cạnh ngắn < 1,25% cạnh dài ảnh;
- vật bị cột/dây che vẫn là **một** instance (các mảnh được nối thành một dòng nhãn);
- ảnh resize cạnh dài 1280.

Đặc tả chi tiết: [`plan/data_pipeline.md`](../plan/data_pipeline.md).

## Cài đặt

```bash
pip install -r tools/requirements.txt          # máy cá nhân: chuyển đổi, kiểm tra, dựng dataset
pip install ultralytics==8.4.214 yt-dlp       # thêm khi cần gán nháp (GPU) / tải video
python tools/test_common.py                   # kiểm thử nhanh, phải in "ALL OK"
python tools/test_qa.py                       # kiểm thử qa_check + label_audit, phải in "ALL OK"
```

## Ai dùng gì

| Người | Việc | Lệnh |
|---|---|---|
| **Minh Thắng** | Mapillary Vistas | `convert_vistas.py` → `qa_check.py --expect configs/qa/vistas.yaml --render` |
| **Bảo Long** | IDD, COCO, Cityscapes, Roboflow | `convert_idd.py`, `convert_coco.py`, `import_yolo.py` → `qa_check.py --expect configs/qa/<nguồn>.yaml --render` |
| **Đạt** | Ảnh Việt Nam | `yt-dlp` → `extract_frames.py` → `dedup.py` → `pseudo_label.py` (Kaggle GPU) |
| **Nguyễn Bình Anh** | Nhãn đã duyệt từ CVAT, gold val | `import_yolo.py --order file` → `qa_check.py` (`vn_reviewed.yaml` / `gold_val.yaml --against ...`) |
| **Lộc** | Gộp phiên bản dataset | `build_dataset.py` → `qa_check.py` (profile `merged` tự nhận) |
| **Quang Hưng** | Mô hình soi nhãn (L6) | `label_audit.py --model teacher.pt` (Kaggle GPU) |

## Các bước

### 1. Chuyển public dataset

```bash
# Vistas: CHẠY --check TRƯỚC để xác nhận cách giải mã ảnh instance (phải khớp ~100%)
python tools/convert_vistas.py --root D:/data/vistas --split validation --out D:/data/converted/vistas --check 5
python tools/convert_vistas.py --root D:/data/vistas --split training --out D:/data/converted/vistas --workers 8

# IDD (cùng script cho Cityscapes: --root .../cityscapes --prefix cs)
python tools/convert_idd.py --root D:/data/IDD_Segmentation --split train --out D:/data/converted/idd --workers 8

# COCO: chỉ ảnh đường phố; SAU ĐÓ bắt buộc gán nháp thêm biển báo (bước 3, --fill-classes)
python tools/convert_coco.py --ann D:/data/coco/annotations/instances_train2017.json \
    --images D:/data/coco/train2017 --out D:/data/converted/coco --max-images 1500   # plan: ≤ 1.500 ảnh, chỉ mở rộng nếu A2 cho thấy COCO giúp

# Bộ YOLO-seg có sẵn (Roboflow, Kaggle). Tên lớp khác thì viết file map JSON: {"motorbike": "Motorcycle", "xe ba gac": "Other vehicle", "logo": null}
python tools/import_yolo.py --src D:/data/raw/rf_hcm --out D:/data/converted/rf_hcm --prefix rfhcm --map maps/rf_hcm.json
```

Mỗi lệnh ghi `stats.json` (số ảnh, số instance mỗi lớp, ảnh bị loại và lý do) và `rejected.csv` vào thư mục output.
Các converter in bảng **nhãn gốc → role** ở đầu: đọc kỹ, sai ở đây là sai cả bộ.

### 2. Ảnh Việt Nam từ video

```bash
yt-dlp -f "bv*[height<=1080]" -o "D:/data/raw/vn_videos/%(title)s [%(id)s].%(ext)s" <link>
python tools/extract_frames.py --videos D:/data/raw/vn_videos --out D:/data/raw/vn_frames       # 1 khung / 2 s, ≤ 150 khung / video
python tools/dedup.py --dir D:/data/raw/vn_frames --threshold 6 --move-to D:/data/raw/vn_dup    # bỏ khung gần trùng
# video dành cho gold val: thư mục riêng, --prefix vngold
python tools/extract_frames.py --videos D:/data/raw/vn_videos_gold --out D:/data/raw/vn_frames_gold --prefix vngold
python tools/dedup.py --dir D:/data/raw/vn_frames --against D:/data/raw/vn_frames_gold --threshold 4 --move-to D:/data/raw/vn_leak
```

### 3. Gán nháp (chạy trên Kaggle/Colab GPU)

```bash
# Đợt 1: mô hình COCO pretrained lớn (chưa cần huấn luyện). Mọi ảnh đều vào hàng duyệt vì thiếu Other vehicle / biển báo.
python tools/pseudo_label.py --model yolo26x-seg.pt --images vn_frames --out vn_pseudo --imgsz 1024
# Đợt 2: teacher 11 lớp, có TTA. confident.txt = ảnh dùng ngay; review_queue.csv = ảnh cần duyệt, đã xếp ưu tiên
python tools/pseudo_label.py --model teacher.pt --images vn_frames --out vn_pseudo_t --augment
# Bổ sung lớp còn thiếu cho COCO (nhãn có sẵn được giữ nguyên)
python tools/pseudo_label.py --model teacher.pt --images converted/coco/images --labels-in converted/coco/labels \
    --fill-classes 9 10 --out converted/coco_filled
```

Đưa ảnh trong `review_queue.csv` vào CVAT (import định dạng Ultralytics YOLO Segmentation), duyệt theo [`plan/labeling_sop.md`](../plan/labeling_sop.md), xuất lại rồi:

```bash
python tools/import_yolo.py --src D:/data/cvat_exports/review_01 --out D:/data/converted/vn_reviewed --prefix vn --order file
```

### 4. Cổng chất lượng (mỗi nguồn, gold val, bộ gộp)

Một công cụ, ba profile (`source` / `gold` / `merged`). Quy trình đầy đủ (ai chạy, khi nào, chấm mẫu, xử lý FAIL) và bảng mã kiểm tra: [`plan/qa_process.md`](../plan/qa_process.md).

```bash
# nguồn: kiểm tra tự động + tạo mẫu phân tầng ~70–90 ảnh (qa/sample.csv) + vẽ ra qa/preview/
python tools/qa_check.py --data D:/data/converted/vistas --expect configs/qa/vistas.yaml --render
#   -> mở qa/preview/, điền cột verdict (ok/bad) và error_type (sot/sai_lop/bien/rider/khac) trong qa/sample.csv
python tools/qa_check.py --data D:/data/converted/vistas --expect configs/qa/vistas.yaml   # chạy lại -> PASS / FAIL

# gold val: chặt hơn, bắt buộc kiểm tra rò rỉ với ảnh train Việt Nam
python tools/qa_check.py --data D:/data/gold_val/gold_val_v1 --expect configs/qa/gold_val.yaml     --against D:/data/raw/vn_frames D:/data/converted/vn_reviewed/images
```

Mọi kết quả nằm trong `<dataset>/qa/` (`report.json`, `sample.csv`, `preview/`, `audit/`). **Sao lưu `qa/` trước khi xóa / chuyển đổi lại một nguồn** vì `sample.csv` chứa phần chấm của người xem. Thêm `--archive reports/qa` để chép báo cáo vào repo rồi commit.

Kết luận in ở cuối: `FAIL` (mã thoát 1) / `PASS_AUTO` (tự động sạch, mẫu chưa chấm xong) / `PASS`. Chép **dòng log** mà lệnh in ra vào `plan/experiment_log.md`.
Lỗi đã biết và chấp nhận được thì ghi vào `waivers` (kèm lý do) của file kỳ vọng, không được lờ đi. Xem [`configs/qa/README.md`](../configs/qa/README.md).
Dữ liệu đổi (chuyển đổi lại, sửa nhãn) thì chạy với `--resample` để lấy mẫu mới. Muốn xem riêng một lớp: `visualize.py --classes 5 7`.

**Tầng L6: mô hình soi nhãn** (khi đã có baseline / teacher; mô hình **không** được train trên dữ liệu đang soi):

```bash
# Kaggle GPU: dự đoán được lưu vào <out>/preds để soi lại trên máy cá nhân
python tools/label_audit.py --data /kaggle/input/gold-val-v1 --model teacher.pt --augment --out /kaggle/working/audit_gold
# máy cá nhân, đổi ngưỡng không cần GPU
python tools/label_audit.py --data D:/data/gold_val/gold_val_v1 --preds D:/audit_gold/preds --render 40   # -> gold_val_v1/qa/audit/
```

`audit_queue.csv` (sót nhãn / sai lớp / biên lệch / nhãn thừa, đã xếp ưu tiên) đưa vào CVAT như `review_queue.csv`.

### Xem dataset bằng viewer

```bash
python tools/viewer/server.py --root D:/data            # tự tìm mọi thư mục có images/ (sâu tối đa 4 cấp)
python tools/viewer/server.py --data D:/data/converted/vistas D:/data/gold_val/gold_val_v1
# -> mở http://localhost:8000
```

- Xem ảnh + mask từng instance (zoom, kéo, bật/tắt từng lớp, xem đỉnh polygon), bảng instance có kích thước bbox và % diện tích.
- Lọc theo split, lớp, ảnh nền, **mã QA** (đọc `qa/flags.json`), **mẫu người xem** (chưa chấm / ok / bad), **nghi vấn L6** (`qa/audit/`). Vật bị cờ có viền đứt nét đỏ (lỗi) / vàng (cảnh báo).
- **Chấm mẫu ngay trên viewer**: ảnh trong `qa/sample.csv` có khung chấm bên phải; phím `O` = ok rồi sang ảnh chưa chấm kế tiếp, `B` = bad (chọn loại lỗi), `N` = ảnh chưa chấm kế tiếp. Lưu thẳng vào `qa/sample.csv`. Chấm xong chạy lại `qa_check.py` để ra kết luận.
- **Ảnh gốc**: phím `H` ẩn mọi lớp phủ; ô "Viền mask" tắt riêng viền. Lựa chọn hiển thị được nhớ giữa các lần mở.
- Link `http://localhost:8000/#ds=<id>&k=<split>/<tên ảnh>` mở thẳng một ảnh (tiện báo nhau trong kênh đội; mỗi người chạy server trên máy mình).
- Cần `qa_check.py` bản có `qa/flags.json` (chạy lại nếu bộ đã chạy QA từ trước). Viewer chỉ nghe ở `127.0.0.1`; đóng file `sample.csv` trong Excel trước khi chấm trên viewer.

### 5. Dựng phiên bản dataset

```bash
cp configs/recipes/example.yaml configs/recipes/merged_v1.yaml    # sửa đường dẫn, số ảnh
python tools/build_dataset.py --recipe configs/recipes/merged_v1.yaml
python tools/qa_check.py --data D:/data/datasets/merged_v1 --render   # profile merged: rò rỉ train/val, manifest, zip, mục tiêu lớp
```

Kết quả: `merged_v1/` + `merged_v1.zip` (images/{train,val}, labels/{train,val}, notebook nhận trực tiếp) + `merged_v1.manifest.json` (recipe, số ảnh và instance theo nguồn và lớp, sha256).
Không ghi đè phiên bản cũ; commit file recipe vào repo để tái lập.

## Đã kiểm thử / chưa kiểm thử

- Đã chạy trên dữ liệu giả lập + 10 ảnh mẫu của ban tổ chức: `common.py` (ghép rider, nối mảnh, mask không chồng, lọc vật nhỏ, crowd), cả 3 converter, `import_yolo`, `qa_check`, `visualize`, `extract_frames`, `dedup`, `build_dataset`, và logic dựng scene của `pseudo_label` (với dự đoán giả).
- **Chưa chạy trên dữ liệu thật**: định dạng Vistas v2.0 (dùng `--check`), tên nhãn IDD/Cityscapes (đọc bảng in ra), và `pseudo_label.py` với ultralytics thật trên GPU.
- `qa_check` (bản theo profile) đã chạy trên dữ liệu thật: Vistas val (1.877 ảnh, ~25 s) và IDD train (5.006 ảnh, ~2,5 phút). `label_audit` mới chạy với dự đoán giả (`--preds`); chế độ `--model` chưa chạy với ultralytics thật.
- **SAM 3 chưa được tích hợp**: hiện dùng mô hình COCO / teacher; SAM nên dùng trực tiếp trong CVAT khi duyệt.
