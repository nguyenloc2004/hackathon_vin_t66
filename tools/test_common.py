"""Kiểm thử nhanh common.py trên dữ liệu tổng hợp: python tools/test_common.py"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, cv2, common as C

def rect(x0,y0,x1,y1): return np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],float)

# 1) xe máy + người ngồi -> 1 Motorcyclist; xe máy trống -> Motorcycle; người đứng xa -> Pedestrian
s = C.Scene(720, 1280)
s.paint_polygons("motorcycle", [rect(100,300,220,400)])
s.paint_polygons("rider", [rect(130,220,190,360)])          # chồng lên xe
s.paint_polygons("motorcycle", [rect(500,300,620,400)])
s.paint_polygons("person", [rect(900,200,950,400)])
s.paint_polygons("car", [rect(1000,500,1010,510)])          # quá nhỏ -> bỏ
r = C.finalize(s)
print("1:", r.reject, sorted(r.classes), "dropped", r.dropped_small)
assert r.reject is None and sorted(r.classes) == [4,5,8] and r.dropped_small == 1

# 2) rider không có xe -> loại ảnh; với 'person' -> Pedestrian
s = C.Scene(720,1280); s.paint_polygons("rider",[rect(100,100,160,300)])
assert C.finalize(s).reject == "unmatched_rider"
s = C.Scene(720,1280); s.paint_polygons("rider",[rect(100,100,160,300)])
assert C.finalize(s, unmatched_rider="person").classes == [8]

# 3) ô tô bị cột che -> 2 mảnh nối thành 1 dòng, vẽ lại ra gần đúng diện tích
s = C.Scene(720,1280)
s.paint_polygons("car",[rect(200,300,600,500)])
s.paint_polygons("occluder",[rect(390,0,410,720)])
r = C.finalize(s); assert len(r.lines)==1
pts = np.array(r.lines[0].split()[1:],float).reshape(-1,2)*[1280,720]
m = np.zeros((720,1280),np.uint8); cv2.fillPoly(m,[np.round(pts).astype(np.int32)],1)
truth = (s.id_map==1)
iou = (m.astype(bool)&truth).sum()/(m.astype(bool)|truth).sum()
print("3: parts bridged, IoU vs truth =", round(float(iou),4)); assert iou > 0.97

# 4) crowd lớn -> loại
s = C.Scene(720,1280); s.paint_polygons("crowd",[rect(0,0,400,400)]); s.paint_polygons("car",[rect(500,300,700,400)])
assert C.finalize(s).reject == "crowd"

# 5) mask không chồng nhau: xe phía trước đè xe phía sau
s = C.Scene(720,1280); s.paint_polygons("car",[rect(100,100,500,400)]); s.paint_polygons("bus",[rect(300,200,800,500)])
r = C.finalize(s); assert sorted(r.classes)==[0,1]

# 6) IoU của polygon xuất ra so với mask thật cho hình tròn
s = C.Scene(720,1280); m0=np.zeros((720,1280),np.uint8); cv2.circle(m0,(640,360),80,1,-1)
s.paint_mask("sign", m0>0); r = C.finalize(s)
pts = np.array(r.lines[0].split()[1:],float).reshape(-1,2)*[1280,720]
m = np.zeros((720,1280),np.uint8); cv2.fillPoly(m,[np.round(pts*16).astype(np.int32)],1,shift=4)
iou=(m&m0).sum()/(m|m0).sum(); print("6: circle IoU", round(float(iou),4)); assert iou>0.97

# 7) phash
img=np.random.RandomState(0).randint(0,255,(360,640,3),np.uint8)
h1=C.phash(img); h2=C.phash(cv2.GaussianBlur(img,(3,3),0)); h3=C.phash(255-img)
d=C.hamming_to_all(h1,np.array([h2,h3],np.uint64)); print("7: hamming", d); assert d[0]<=6 and d[1]>20

# 8) riders_from_boxes
print("8:", C.riders_from_boxes([(130,220,190,360),(900,200,950,400),(230,280,280,400)],[(100,300,220,400)]))
print("ALL OK")
