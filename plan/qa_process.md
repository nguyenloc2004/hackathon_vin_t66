# Quy trình QA dữ liệu

Tài liệu này là **quy trình chuẩn** để kiểm tra chất lượng mọi bộ dữ liệu của đội: từng nguồn đã chuyển đổi, gold val, và bộ gộp trước khi đưa lên Kaggle. Ai chạy QA, ai chấm mẫu, ai quyết định đều làm theo file này.

Công cụ: [`tools/qa_check.py`](../tools/qa_check.py) (tầng L0–L5), [`tools/label_audit.py`](../tools/label_audit.py) (tầng L6), file kỳ vọng [`configs/qa/`](../configs/qa/README.md). Cài đặt và lệnh đầy đủ: [`tools/README.md`](../tools/README.md) mục 4.

---

## 1. Nguyên tắc

1. **Một bộ kiểm tra, ba cổng.** Cùng một `qa_check.py` chạy ở cả ba cổng; chỉ **profile** khác nhau (`source` / `gold` / `merged`). Bộ gộp là hợp của các nguồn, nên kiểm tra nhãn phải giống hệt nhau. Chạy lại ở cổng gộp còn bắt được lỗi do chính `build_dataset.py` gây ra.
2. **Mỗi nguồn khai báo mình phải trông thế nào.** File `configs/qa/<nguồn>.yaml` nói lớp nào được phép vắng, lớp nào tối thiểu bao nhiêu, mã nào nghiêm hơn. Ví dụ COCO thô được thiếu Sign, nhưng `coco_filled` có dưới 200 Sign là lỗi chặn.
3. **Máy ra kết luận, không ghi tay "ĐẠT".** Kết luận chỉ có ba giá trị:

   | Kết luận | Nghĩa | Được làm gì |
   |---|---|---|
   | `FAIL` | Còn lỗi chưa miễn trừ (mã thoát 1) | Không được dùng. Sửa rồi chạy lại |
   | `PASS_AUTO` | Kiểm tra tự động sạch, mẫu người xem **chưa chấm xong** | Được dùng cho ablation nhanh; **chưa** được vào `merged_final` |
   | `PASS` | Tự động sạch **và** mẫu người xem dưới ngưỡng lỗi | Được vào recipe, lên Kaggle |

4. **Người xem là một phần bắt buộc của cổng.** Máy chỉ bắt được lỗi hình học. Sót vật, sai lớp, ghép rider sai thì phải có người nhìn (L5) hoặc mô hình soi (L6).
5. **Miễn trừ phải có lý do, ghi trong file.** Lỗi đã biết và chấp nhận được thì ghi vào `waivers` của file kỳ vọng. Không lờ đi, không tắt cả mã kiểm tra chỉ để qua cổng. Mục đã miễn trừ vẫn hiện trong báo cáo.
6. **Mọi thứ truy vết được.** `report.json` ghi commit của tools, ngưỡng, profile, số liệu từng split. Bản sao được commit vào `reports/qa/` (cờ `--archive`).

---

## 2. Ba cổng và lịch chạy

| Cổng | Profile | Dữ liệu | Khi nào (gắn milestone) | Người chạy | Người chấm mẫu | Người chốt |
|---|---|---|---|---|---|---|
| **A. Nguồn** | `source` | Vistas, IDD, COCO/`coco_filled`, Việt Nam nháp | Sau mỗi lần chuyển đổi. Vistas, COCO trước 14:00 (M2); IDD trước 18:00 (M4) | Người phụ trách nguồn | Người phụ trách + Lộc | Lộc |
| **A'. Lô duyệt CVAT** | `source` (`vn_reviewed.yaml`) | Mỗi lô ảnh Việt Nam đã duyệt | Mỗi lô, trước khi đưa vào `merged_v2` (M7, 00:30) | Người xuất lô | Bình Anh (thay cho QA chéo 10%) | Bình Anh |
| **B. Gold val** | `gold` | `gold_val_v0`, `gold_val_v1` | v0 trước 15:30 (M3); v1 trước khi khóa 21:00 (M6, G2) | Bình Anh | Người duyệt thứ hai | Bình Anh + Lộc |
| **C. Bộ gộp** | `merged` | `merged_vN`, `merged_final` | Ngay sau `build_dataset.py`, trước khi đưa lên Kaggle; `merged_final` trước 01:30 (M8, G3) | Lộc | Lộc (20 ảnh nhanh) | Lộc |
| **L6. Mô hình soi** | — | Gold val, ảnh Việt Nam đã duyệt | Gold val: 19:30–20:30, **trước** khi khóa. Ảnh VN: khi teacher xong (~22:00) | Hưng (GPU) | Người duyệt theo hàng đợi | Bình Anh |

Không có bộ dữ liệu nào được dùng để huấn luyện nếu chưa qua cổng của nó.

---

## 3. Cổng A: từng nguồn

### 3.1 Các bước

```bash
# 1) kiểm tra tự động + tạo mẫu 30 ảnh + vẽ ảnh mẫu
python tools/qa_check.py --data D:/data/converted/vistas --expect configs/qa/vistas.yaml --render

# 2) mở D:/data/converted/vistas/qa/preview/, chấm vào qa/sample.csv (mục 6)

# 3) chạy lại cùng lệnh: đọc kết luận, lưu bản sao báo cáo vào repo
python tools/qa_check.py --data D:/data/converted/vistas --expect configs/qa/vistas.yaml --archive reports/qa
```

4. Chép dòng `Dòng cho experiment_log.md` mà lệnh in ra vào bảng **Cổng chất lượng** của [`experiment_log.md`](experiment_log.md).
5. Báo Lộc ở kênh đội: tên nguồn, kết luận, 1–2 vấn đề lớn nhất.

### 3.2 Đọc báo cáo

Kết quả nằm trong `<dataset>/qa/`:

| File | Nội dung |
|---|---|
| `report.json` | Báo cáo đầy đủ: số ảnh, số instance mỗi lớp, mọi mã kiểm tra (mức, số lượng, tối đa 50 ví dụ), kết quả chấm mẫu |
| `sample.csv` | Mẫu người xem. **Đây là công sức của người chấm**: sao lưu `qa/` trước khi xóa hoặc chuyển đổi lại thư mục |
| `preview/` | Ảnh mẫu: trái là ảnh gốc, phải là nhãn |
| `audit/` | Kết quả `label_audit.py` (nếu đã chạy) |

Mỗi dòng in ra có dạng `[LỖI][L1] overlap_partial: 3` kèm ví dụ `all/<tên>.txt:<dòng>`. Mở đúng dòng đó trong file nhãn, hoặc vẽ riêng ảnh đó:

```bash
python tools/visualize.py --data D:/data/converted/vistas --stems vistas_abc --out D:/tmp/xem
```

### 3.3 Đọc các cảnh báo đặc trưng của nguồn public

Kinh nghiệm từ lần chạy đầu trên Vistas val và IDD train:

- **`multipart_overlap` (hàng chục đến hàng trăm)**: bình thường. YOLO-seg chỉ có một polygon/vật và không có lỗ, nên vật bị che còn vài mẩu phải nối bằng cạnh đi-về. Không cần sửa.
- **`small_instance` (vài trăm)**: vật vừa dưới ngưỡng 1,25% do làm tròn polygon. Chỉ đáng lo nếu nhiều vật nhỏ hẳn (< 10 px).
- **`reject_frac`**: xem lý do trong `stats.json`. IDD bị loại 28% ảnh vì `unmatched_rider`, nghĩa là mất nhiều ảnh có người lái xe máy. Đây là vấn đề của converter, cần báo Lộc.
- **`dup_within`**: IDD, Vistas lấy từ chuỗi ảnh liên tiếp nên có cặp gần trùng. Với nguồn public chỉ là cảnh báo; với ảnh Việt Nam là lỗi (phải chạy `dedup.py` trước).

---

## 4. Cổng B: gold val

Gold val là thước đo của mọi quyết định, nên chặt nhất.

```bash
python tools/qa_check.py --data D:/data/gold_val/gold_val_v1 --expect configs/qa/gold_val.yaml \
    --against D:/data/raw/vn_frames D:/data/converted/vn_reviewed/images --render
```

- **Luôn kèm `--against`** tất cả ảnh Việt Nam dùng cho train. Thiếu thì báo `leak_unchecked`.
- Profile `gold` nâng các mã sau lên lỗi: `overlap_light`, `ped_in_vehicle`, `self_intersect`, `small_instance`, `dup_within`, `leak_ref`, `class_missing`.
- Mẫu người xem 40 ảnh (20 ngẫu nhiên), ngưỡng lỗi **5%**.
- Theo dõi thêm: mỗi lớp ≥ 15 instance, ảnh đêm ≥ 15% (ước lượng theo độ sáng; tỉ lệ đúng xem cột điều kiện trong `vn_sources.csv`).

**Thứ tự để khóa `gold_val_v1` trước 21:00:**

1. 19:30: cổng B tự động sạch.
2. 19:30–20:30: L6 soi gold val bằng mô hình ablation tốt nhất hiện có (train trên public nên chưa từng thấy gold val). Sửa các nghi vấn đúng.
3. Chạy lại cổng B, chấm mẫu → `PASS`.
4. Xuất `gold_val_v1.zip` + sha256, ghi vào `experiment_log.md`. **Từ đây không sửa gold val.** Nếu sau này phát hiện lỗi nặng (> 5% ảnh), Lộc quyết định ra `v1.1` và đánh giá lại các ứng viên trên cả hai bản.

---

## 5. Cổng C: bộ gộp

```bash
python tools/build_dataset.py --recipe configs/recipes/merged_v2.yaml
python tools/qa_check.py --data D:/data/datasets/merged_v2 --render --archive reports/qa
```

Profile `merged` được nhận tự động khi có `images/train`. Ở cổng này các kiểm tra quan trọng nhất là:

| Mã | Kiểm tra | Nếu lỗi |
|---|---|---|
| `leak_train_val` | Ảnh val trùng tên hoặc gần trùng ảnh train | Recipe đang tắt `exclude_near_duplicates_of_val`, hoặc gold val lẫn nguồn train: dừng, báo Lộc |
| `manifest_mismatch`, `zip_mismatch` | Manifest, zip và dữ liệu trên đĩa khớp nhau | Có người sửa thư mục sau khi build: dựng lại phiên bản mới, không vá tay |
| `class_missing` / `class_below_target` | Train ≥ 1.000 instance/lớp (Cyclist, Other vehicle ≥ 400); val ≥ 15 | Tăng `max_images`/`repeat` cho nguồn có lớp đó, hoặc chọn `select: priority` |
| `split_layout`, `label_orphan` | Cấu trúc `images/{train,val}`, `labels/{train,val}` đúng như notebook cần | Dựng lại |

Mẫu người xem ở cổng này chỉ cần lướt 20 ảnh để bắt lỗi ghép nguồn. Nhãn đã được kiểm ở cổng A. Bước 2 của notebook cuộc thi là lớp kiểm tra thứ hai, không thay thế cổng C.

`merged_final` phải đạt `PASS` trước 01:30 (G3).

---

## 6. Chấm mẫu người xem (tầng L5)

Lần chạy đầu tạo `qa/sample.csv`. Mẫu có 30 ảnh (gold val: 40) chia ba nhóm. Seed 71 nên ai chạy cũng ra cùng mẫu.

| Nhóm (cột `reason`) | Số ảnh | Mục đích |
|---|---|---|
| `flag:<mã>` | 10 | Ảnh bị cảnh báo, xoay vòng qua các mã: xác nhận cảnh báo là thật hay nhiễu |
| `rare` | 10 | Ảnh có Bus, Other vehicle, Bicycle, Cyclist hoặc Motorcyclist: nơi converter hay sai nhất |
| `random` | 10 | Ước lượng tỉ lệ lỗi chung của nguồn |

**Cách chấm.** Cách nhanh nhất là dùng viewer: `python tools/viewer/server.py --data <dataset>`, lọc "Mẫu người xem: chưa chấm", chấm bằng phím `O` / `B`, kết quả lưu thẳng vào `qa/sample.csv` ([`tools/README.md`](../tools/README.md), mục viewer). Viewer còn tô viền các vật bị cờ, nên kiểm tra cảnh báo nhanh hơn. Cách thủ công: mở `qa/preview/` cạnh file CSV (Excel mở được vì file có BOM).

1. Mỗi ảnh, đối chiếu với checklist ở [`labeling_sop.md`](labeling_sop.md) §5.
2. Điền `verdict`: `ok` nếu ảnh không có lỗi **đáng kể**; `bad` nếu có ít nhất một lỗi làm sai bài học của mô hình.
3. Nếu `bad`, điền `error_type` (lỗi chính) và `note` (dòng nhãn, mô tả ngắn):

   | `error_type` | Ví dụ | Sửa ở đâu |
   |---|---|---|
   | `sot` | Bỏ sót xe máy bị che một nửa, biển báo xa | Nhãn nháp / người duyệt; nếu là nguồn public: chấp nhận hoặc gán bù bằng teacher |
   | `sai_lop` | Ba gác gán Truck, xe đạp điện gán Bicycle | Bảng ánh xạ [`class_mapping.md`](class_mapping.md) hoặc quy ước [`conventions.md`](conventions.md) |
   | `bien` | Mask lệch rõ, phủ cả bóng đổ | Người duyệt; với converter: kiểm tra resize/raster |
   | `rider` | Motorcyclist chỉ có người, xe và người tách hai instance | Logic ghép rider trong `common.py` |
   | `khac` | Mọi lỗi khác, ghi rõ ở `note` | |

4. Chạy lại lệnh. Kết luận `FAIL` với mã `human_audit` nếu tỉ lệ `bad` vượt ngưỡng:

   | Dữ liệu | Ngưỡng |
   |---|---|
   | Nguồn public, Việt Nam nháp | 10% (3/30) |
   | Lô duyệt CVAT | 15% (= ngưỡng trả task của SOP) |
   | Gold val | 5% (2/40) |

Không chấm sau 02:00 (giống quy tắc duyệt nhãn). Sau khi sửa dữ liệu, chạy với `--resample` để lấy mẫu mới. Bản đã chấm cũ được tự lưu thành `qa/sample.<giờ>.csv`.

---

## 7. Khi kết luận là FAIL

| Loại lỗi | Mã | Hành động |
|---|---|---|
| Định dạng nhãn | `class_id`, `few_points`, `coord_range`, `parse_error`, `img_unreadable` | Lỗi converter hoặc file xuất CVAT. Sửa công cụ rồi chuyển đổi lại, không vá file nhãn bằng tay hàng loạt |
| Vài polygon hỏng lẻ tẻ | `zero_area`, `overlap_partial`, `self_intersect` | ≤ 5 dòng: xóa dòng hoặc sửa trong CVAT, **hoặc** ghi miễn trừ có lý do. Nhiều hơn: tìm nguyên nhân chung |
| Phân phối lớp | `class_below_min`, `class_missing` | Ánh xạ thiếu nhãn gốc (đọc lại bảng in ra của converter), hoặc bước gán bù thất bại |
| Trùng / rò rỉ | `dup_within`, `leak_*` | Chạy `dedup.py` (có `--against` gold val), dựng lại |
| Người xem | `human_audit` | Đọc `error_types` trong báo cáo: một loại chiếm đa số → lỗi hệ thống, sửa công cụ/quy ước; rải rác → trả task cho người gán |

Quá 30 phút mà nguồn vẫn `FAIL` thì Lộc quyết định theo cột **Nếu trễ** của [`milestones.md`](milestones.md): bỏ nguồn, dùng phần đã sạch, hoặc miễn trừ có ghi chép.

### Miễn trừ

Ghi vào file kỳ vọng của nguồn, có lý do cụ thể:

```yaml
waivers:
  - check: zero_area
    item: "all/vistas_aEQC2qt5RDzFgEyxx6_Vzw.txt:5"
    reason: "biển báo 2 px bị converter làm suy biến; đã xem, bỏ qua được (Lộc, 27/11 14:10)"
```

`item` là mẫu glob khớp với ví dụ trong báo cáo, ví dụ `"all/idd_53_*.txt:*"`. Miễn trừ cả một mã cho toàn nguồn phải được Lộc đồng ý.

---

## 8. Tầng L6: mô hình soi nhãn

Mục đích: tìm **sót nhãn** và **sai lớp**, hai loại lỗi `qa_check` không bắt được và hại điểm nhiều nhất.

```bash
# Kaggle/Colab GPU (dự đoán được lưu vào <out>/preds)
python tools/label_audit.py --data /kaggle/input/gold-val-v1 --model best_ablation.pt --augment --out /kaggle/working/audit_gold
# Máy cá nhân, chỉnh ngưỡng không cần GPU; kết quả vào <data>/qa/audit/
python tools/label_audit.py --data D:/data/gold_val/gold_val_v1 --preds D:/audit_gold/preds --render 40
```

| Loại nghi vấn | Điều kiện | Nghi ngờ |
|---|---|---|
| `missing_label` | Mô hình tự tin (conf ≥ 0,6) có vật mà nhãn có sẵn phủ < 20% | Sót nhãn |
| `wrong_class` | Khớp vị trí (IoU ≥ 0,5) nhưng khác lớp, conf ≥ 0,5 | Sai lớp |
| `loose_boundary` | Cùng lớp, IoU < 0,7 | Biên lệch |
| `spurious_label` | Nhãn đủ lớn mà mô hình không thấy gì | Nhãn thừa / gán nhầm |

**Quy tắc chọn mô hình: không bao giờ soi dữ liệu mà mô hình đã được train trên đó.** Mô hình đã học thuộc nhãn sai thì không thấy nhãn sai.

| Soi gì | Dùng mô hình nào |
|---|---|
| Gold val | Ablation tốt nhất hoặc teacher (đều train trên public, chưa thấy gold val) |
| Ảnh Việt Nam đã duyệt | Teacher, **trước** khi ảnh đó được đưa vào train của teacher/bản sau |
| IDD, COCO | Baseline A0 (chỉ train Vistas) |

**Quy trình:**

1. Hưng chạy trên GPU, đưa `audit_queue.csv` + `preview/` lên Drive.
2. Bình Anh chia `audit_queue.csv` (đã xếp ưu tiên) thành task CVAT như hàng duyệt thường.
3. Người duyệt sửa nhãn **chỉ khi nghi vấn đúng**. Mô hình có thể sai, ví dụ người ngồi trong xe đúng là không được gán.
4. Ghi vào bảng L6 của `experiment_log.md`: số nghi vấn đã xem và **tỉ lệ đúng là lỗi thật**. Tỉ lệ thấp (< 30%) thì tăng `--conf-miss`; đây cũng là số liệu cho slide.
5. Chạy lại cổng A'/B trên dữ liệu đã sửa.

---

## 9. Ghi chép và số liệu cho slide

| Ở đâu | Ghi gì | Ai |
|---|---|---|
| [`experiment_log.md`](experiment_log.md), bảng **Cổng chất lượng** | Dòng log mà `qa_check` in ra + mã đã miễn trừ | Người chạy cổng |
| [`experiment_log.md`](experiment_log.md), bảng **Mô hình soi nhãn** | Số nghi vấn, đã duyệt, tỉ lệ lỗi thật | Hưng / Bình Anh |
| `reports/qa/<dataset>.json` (commit) | Bản sao `report.json` qua `--archive` | Người chạy cổng |
| `configs/qa/<nguồn>.yaml` (commit) | Kỳ vọng + miễn trừ có lý do | Người phụ trách nguồn, Lộc duyệt |

Số liệu nên thu cho phần quy trình dữ liệu trong [`presentation.md`](presentation.md):

- Mỗi nguồn: số ảnh vào → số ảnh bị loại (lý do) → số ảnh qua cổng; tỉ lệ lỗi trong mẫu người xem.
- Gold val: tỉ lệ lỗi mẫu, số nhãn được sửa nhờ L6, độ phủ lớp.
- L6: số nghi vấn, tỉ lệ là lỗi thật, phân bố loại lỗi (sót / sai lớp / biên).
- Trước/sau: 2–3 ảnh lỗi do QA bắt được và cách đã sửa.

---

## 10. Checklist nhanh

**Người phụ trách nguồn (cổng A):**
- [ ] Chạy `qa_check.py --expect configs/qa/<nguồn>.yaml --render`
- [ ] Đọc mọi dòng `[LỖI]`; xem 2–3 ví dụ của mỗi `[CẢNH BÁO]` có số lượng bất thường
- [ ] Chấm 30 ảnh trong `qa/sample.csv`
- [ ] Chạy lại với `--archive reports/qa` → `PASS`
- [ ] Dòng log vào `experiment_log.md`, báo Lộc

**Bình Anh (gold val, lô CVAT):**
- [ ] Gold val chạy kèm `--against` toàn bộ ảnh train Việt Nam
- [ ] L6 trên gold val trước 20:30, sửa xong trước khi khóa
- [ ] Mỗi lô CVAT qua cổng A' với `vn_reviewed.yaml`
- [ ] Khóa `gold_val_v1` (zip + sha256) chỉ khi `PASS`

**Lộc (bộ gộp):**
- [ ] Mọi nguồn trong recipe đã `PASS` (hoặc `PASS_AUTO` nếu chỉ dùng cho ablation)
- [ ] `qa_check` trên `merged_vN` không có `leak_train_val`, `manifest_mismatch`, `zip_mismatch`
- [ ] Bảng lớp: không lớp nào dưới mục tiêu mà không có lý do
- [ ] `merged_final` `PASS` trước 01:30

---

## Phụ lục: bảng mã kiểm tra

Mức theo profile `source / gold / merged` (E = lỗi, W = cảnh báo, — = tắt). Nguồn chuẩn là bảng `CHECKS` trong [`qa_check.py`](../tools/qa_check.py); file kỳ vọng có thể đổi mức qua khóa `severity`.

| Tầng | Mã | Kiểm tra | source | gold | merged |
|---|---|---|:-:|:-:|:-:|
| L0 cấu trúc | `img_unreadable`, `stem_conflict` | Ảnh hỏng; hai ảnh cùng tên khác đuôi | E | E | E |
| | `img_not_resized` | Cạnh dài > 1280 | W | W | W |
| | `label_orphan` | Nhãn không có ảnh | W | W | E |
| | `split_layout` | Đủ `images/{train,val}` + `labels/{train,val}` | — | — | E |
| L1 nhãn | `class_id`, `few_points`, `coord_range`, `zero_area`, `parse_error` | Định dạng YOLO-seg | E | E | E |
| | `overlap_partial` | Hai mask chồng một phần > 20% vật nhỏ (tính trên phần "đặc", bỏ đường mảnh < 3 px) | E | E | E |
| | `overlap_light` | Chồng > 5% | W | E | W |
| | `multipart_overlap` | Chồng do polygon nhiều mảnh (cạnh nối cắt ngang vật khác, hoặc vật lớn bị tô kín lỗ). Sản phẩm phụ của định dạng, không phải lỗi gán | W | W | W |
| | `contained` | Vật nằm trọn trong vật lớn (lỗ polygon) | W | W | W |
| | `ped_in_vehicle` | Người nằm trọn trong xe (guideline: không gán) | W | E | W |
| | `self_intersect` | Polygon tự cắt kiểu hình nơ (diện tích triệt tiêu) | W | E | W |
| | `small_instance` | Vật dưới 1,25% cạnh dài | W | E | W |
| | `rider_tall` | Motorcyclist/Cyclist cao/rộng > 3,2 (mask chỉ có người?) | W | W | W |
| | `image_crowded` | Ảnh > 100 instance | W | W | W |
| L2 phân phối | `crowded_frac` | > 5% ảnh có > 100 instance | E | E | E |
| | `class_missing` | Lớp có 0 instance (trừ `absent_ok`) | W | E | E |
| | `class_below_min` / `class_below_target` | Dưới `min` (lỗi) / `target` (cảnh báo). Mặc định: train gộp ≥ 1.000 (Cyclist, Other vehicle ≥ 400); val và gold ≥ 15 | E / W | E / W | E / W |
| | `background_frac` | > 10% ảnh nền | W | W | W |
| | `reject_frac`, `stats_mismatch` | Converter loại > 25% ảnh; `stats.json` không khớp đĩa | W | — | — |
| | `night_frac` | Ảnh đêm dưới ngưỡng (ước lượng theo độ sáng) | — | W | — |
| L3 chéo tập | `dup_within` | Ảnh gần trùng trong split (pHash ≤ 4; bỏ qua `_rN` do `repeat`) | W | E | W |
| | `leak_train_val` | Ảnh val trùng tên / gần trùng ảnh train | — | — | E |
| | `leak_ref` | Gần trùng ảnh tham chiếu `--against` | W | E | E |
| | `leak_unchecked` | Gold val chạy không kèm `--against` | — | W | — |
| L4 toàn vẹn | `manifest_missing` | Không thấy manifest cạnh thư mục | — | — | W |
| | `manifest_mismatch`, `zip_mismatch` | Manifest khớp số ảnh và instance trên đĩa; zip khớp sha256, danh sách file, cấu trúc | — | — | E |
| L5 người xem | `human_audit` | Tỉ lệ ảnh lỗi trong mẫu vượt ngưỡng (mục 6) | E | E | E |
| | `human_audit_pending` | Mẫu chưa chấm xong (kết luận `PASS_AUTO`) | W | W | W |
