# Sổ quy ước của đội (sống, cập nhật liên tục)

Người quản lý: **Nguyễn Bình Anh** · Người chốt cuối: **Lộc**

Nguyên tắc: guideline của ban tổ chức luôn ưu tiên. File này chỉ ghi **ca guideline chưa nói rõ** hoặc ca đặc thù Việt Nam. Mỗi quyết định ghi: ca, quyết định, lý do, người chốt, giờ. Mục đánh dấu **[ĐỀ XUẤT]** là chưa chốt, phải chốt ở buổi họp gần nhất.

## Phương tiện

| Ca | Quyết định | Lý do |
|---|---|---|
| Xe máy điện kiểu tay ga (VinFast, Yadea…) | Motorcycle / Motorcyclist | Guideline: xe điện 2 bánh kiểu xe máy |
| Xe đạp điện có bàn đạp, khung xe đạp | Bicycle / Cyclist | Guideline mục 3 |
| Xe đạp điện dáng tay ga nhưng có bàn đạp nhỏ | **[ĐỀ XUẤT]** Motorcycle / Motorcyclist (theo hình dáng) | Phân loại theo hình dáng nhìn thấy |
| Shipper/xe ôm ngồi trên xe đang đỗ chờ khách | Motorcyclist | Guideline: đã ngồi trên yên là Motorcyclist |
| Người đứng chống chân, một chân chạm đất, vẫn ngồi trên yên | Motorcyclist | Như trên |
| Người ngồi nghiêng trên yên xe đang đỗ ở vỉa hè | Motorcyclist | Guideline: người ngồi trên yên xe đang đỗ |
| Xe bán tải (Ranger, Hilux…) | Car | Guideline |
| Xe tải nhỏ (Kia K200, Hyundai Porter, xe tải van chở hàng) | Truck | Có thùng chở hàng |
| Xe van chở khách 9–15 chỗ (Ford Transit, Solati, limousine 9 chỗ) | Car | Guideline: 9–15 chỗ là Car |
| Xe khách 16 chỗ trở lên, xe buýt mini 29 chỗ | Bus | Guideline |
| Xe ba gác máy, xe lôi, xích lô, xe ba bánh chở hàng | Other vehicle (gộp người lái/đạp vào cùng instance) | Guideline |
| Xe máy kéo thùng hàng tự chế | **[ĐỀ XUẤT]** Other vehicle cho cả cụm | Không còn là xe máy thuần |
| Xe đầu kéo + container | Truck (đầu kéo) + Other vehicle (container/rơ-moóc) | Guideline |
| Xe cẩu, máy xúc, xe lu | Other vehicle | Guideline |
| Xe cứu thương dạng van, xe cảnh sát | Car | Guideline |
| Xe cứu hỏa, xe chở rác, xe bồn | Truck | Guideline |
| Xe điện du lịch / xe golf chở khách | Car | Guideline |
| Xe đẩy hàng rong, xe kéo tay | Không gán (người đẩy là Pedestrian) | Guideline |
| Xe lăn | Không gán xe lăn; người ngồi và người đẩy là Pedestrian | Guideline |

## Người

| Ca | Quyết định |
|---|---|
| Người ngồi ghế nhựa quán vỉa hè | Pedestrian |
| Người bế/cõng trẻ em | Một instance Pedestrian |
| Người cầm ô | Pedestrian, mask gồm ô |
| Người trong ô tô, xe buýt nhìn qua kính | Không gán |
| Người đứng trên thùng xe tải | **[ĐỀ XUẤT]** Không gán riêng, nằm trong mask Truck (coi như hàng hóa nhìn thấy trên xe) |
| Người trên áp phích, màn hình LED | Không gán |

## Biển báo và đèn

| Ca | Quyết định |
|---|---|
| Biển tên đường chính thức (xanh dương, chữ trắng) | Sign |
| Biển chỉ dẫn trên giá long môn | Sign (mỗi tấm một instance) |
| Biển “Cấm đỗ xe” tạm thời, cọc tiêu, rào chắn | Biển tạm có mặt biển: Sign; cọc tiêu/rào chắn: không gán |
| Biển hiệu cửa hàng, biển quảng cáo, băng rôn tuyên truyền | Không gán |
| Biển nhìn từ phía sau | Sign |
| Đèn đếm ngược **liền hộp** với đèn | Thuộc cùng instance Traffic light |
| Đèn đếm ngược **hộp riêng** đặt cạnh đèn | **[ĐỀ XUẤT]** Traffic light riêng (một hộp một instance) |
| Đèn vàng nhấp nháy cảnh báo công trường nhỏ | Không gán |

## Lịch sử quyết định

| Giờ | Ca | Quyết định | Người chốt |
|---|---|---|---|
| | | | |
