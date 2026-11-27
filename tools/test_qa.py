"""Kiểm thử qa_check.py và label_audit.py trên dữ liệu tổng hợp: python tools/test_qa.py"""
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
import label_audit as LA  # noqa: E402
import qa_check as Q  # noqa: E402

W, H = 640, 360
RNG = np.random.RandomState(71)


def noise_img():
    small = RNG.randint(0, 255, (9, 16, 3)).astype(np.uint8)   # ảnh mịn -> pHash ổn định, ảnh khác nhau xa nhau
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float) / [W, H]


def line(cls, pts):
    return f"{cls} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)


def put(root, split, stem, img, lines):
    sub = Path(split) if split else Path()
    C.save_image(root / "images" / sub / f"{stem}.jpg", img)
    C.write_label(root / "labels" / sub / f"{stem}.txt", lines)


def run(argv):
    return Q.run(Q.parse_args(argv))


def counts(report):
    return {k: v["count"] for k, v in report["checks"].items()}


tmp = Path(tempfile.mkdtemp(prefix="qa_test_"))
try:
    # ---------------------------------------------------------------- 1) nguồn có lỗi đủ loại
    src = tmp / "src"
    dup = noise_img()
    put(src, "", "a_ok", noise_img(), [line(0, rect(100, 100, 300, 250)), line(8, rect(400, 100, 440, 300))])
    put(src, "", "b_badcls", noise_img(), [line(12, rect(10, 10, 100, 100))])
    put(src, "", "c_coord", noise_img(), [line(0, rect(10, 10, 100, 100) * 1.0 + [0, 0.95])])
    put(src, "", "d_overlap", noise_img(), [line(0, rect(100, 100, 300, 250)), line(2, rect(200, 100, 400, 250))])
    bowtie = np.array([[100, 100], [300, 250], [300, 60], [100, 200]], float) / [W, H]   # hai vòng ngược chiều, lệch nhau
    put(src, "", "e_twist", noise_img(), [line(0, bowtie)])
    put(src, "", "f_dup1", dup, [line(0, rect(100, 100, 300, 250))])
    put(src, "", "f_dup2", dup, [line(0, rect(100, 100, 300, 250))])
    put(src, "", "g_pedcar", noise_img(), [line(0, rect(100, 100, 400, 300)), line(8, rect(200, 150, 240, 250))])
    put(src, "", "h_tiny", noise_img(), [line(9, rect(10, 10, 14, 14))])
    C.write_label(src / "labels" / "orphan.txt", [line(0, rect(1, 1, 50, 50))])
    shutil.copy(src / "images" / "a_ok.jpg", src / "images" / "a_ok.png")   # cùng stem, khác đuôi
    r = run(["--data", str(src), "--no-sample"])
    c = counts(r)
    print("1:", r["verdict"], c)
    assert r["profile"] == "source" and r["verdict"] == "FAIL"
    for code in ("class_id", "coord_range", "overlap_partial", "self_intersect", "dup_within", "ped_in_vehicle",
                 "small_instance", "label_orphan", "stem_conflict"):
        assert c.get(code), code
    assert c["self_intersect"] == 1 and c["overlap_partial"] == 1
    assert (src / "qa" / "report.json").is_file()

    # ---------------------------------------------------------------- 1b) người phía sau bị che còn hai mẩu: cạnh nối
    #      cắt ngang người phía trước -> chỉ là multipart_overlap (cảnh báo), không phải overlap_partial
    s = C.Scene(H, W)
    s.paint_polygons("person", [np.array([[100, 200], [200, 200], [200, 260], [100, 260]], float)])
    s.paint_polygons("person", [np.array([[105, 90], [195, 90], [195, 310], [105, 310]], float)])
    res = C.finalize(s)
    br = tmp / "bridge"
    put(br, "", "b", noise_img(), res.lines)
    c = counts(run(["--data", str(br), "--no-sample", "--no-hash"]))
    print("1b:", len(res.lines), c)
    assert len(res.lines) == 2 and c.get("multipart_overlap") == 1 and not c.get("overlap_partial")

    # ---------------------------------------------------------------- 1c) file kiểu cũ ở gốc được chuyển vào qa/
    leg = tmp / "legacy"
    put(leg, "", "x", noise_img(), [line(0, rect(100, 100, 300, 250))])
    (leg / "qa_sample.csv").write_text("key,split,stem,reason,verdict,error_type,note\n", encoding="utf-8")
    run(["--data", str(leg), "--no-hash", "--archive", str(tmp / "archive")])
    assert not (leg / "qa_sample.csv").exists() and (leg / "qa" / "sample.csv").is_file()
    assert (tmp / "archive" / "legacy.json").is_file()

    # ---------------------------------------------------------------- 2) đa giác nhiều mảnh (nối đi-về) không bị báo tự cắt
    s = C.Scene(H, W)
    s.paint_polygons("car", [np.array([[100, 100], [400, 100], [400, 250], [100, 250]], float)])
    s.paint_polygons("occluder", [np.array([[240, 0], [260, 0], [260, 360], [240, 360]], float)])
    res = C.finalize(s)
    multi = tmp / "multi"
    put(multi, "", "m", noise_img(), res.lines)
    r = run(["--data", str(multi), "--no-sample", "--no-hash"])
    print("2:", counts(r))
    assert not counts(r).get("self_intersect")

    # ---------------------------------------------------------------- 3) file kỳ vọng + miễn trừ
    exp = tmp / "exp.yaml"
    exp.write_text("profile: source\nabsent_ok: [Bus]\nclasses:\n  Sign: {min: 5}\nseverity:\n  dup_within: off\n"
                   "waivers:\n  - {check: overlap_partial, item: 'all/d_overlap.txt:*', reason: test}\n",
                   encoding="utf-8")
    r = run(["--data", str(src), "--no-sample", "--expect", str(exp)])
    c = counts(r)
    print("3:", c.get("class_below_min"), r["checks"]["overlap_partial"])
    assert c["class_below_min"] == 1 and "dup_within" not in c
    assert r["checks"]["overlap_partial"]["count"] == 0 and r["checks"]["overlap_partial"]["waived"] == 1
    assert "all/Bus" not in r["checks"]["class_missing"]["examples"]

    # ---------------------------------------------------------------- 4) bộ gộp: rò rỉ train/val, manifest
    mg = tmp / "datasets" / "merged_t"
    shared = noise_img()
    for i in range(3):
        put(mg, "train", f"t{i}", noise_img(), [line(i, rect(100, 100, 300, 250))])
    put(mg, "train", "t0_r1", C.read_image(mg / "images" / "train" / "t0.jpg"), [line(0, rect(100, 100, 300, 250))])
    put(mg, "train", "leak", shared, [line(0, rect(100, 100, 300, 250))])
    put(mg, "val", "v0", shared, [line(0, rect(100, 100, 300, 250))])
    put(mg, "val", "v1", noise_img(), [line(1, rect(100, 100, 300, 250))])
    r = run(["--data", str(mg), "--no-sample"])
    c = counts(r)
    print("4a:", r["profile"], c)
    assert r["profile"] == "merged" and c.get("leak_train_val") == 1 and c.get("manifest_missing") == 1
    assert not c.get("dup_within"), "ảnh nhân bản _r1 là chủ ý, không tính trùng"
    (mg / "images" / "train" / "leak.jpg").unlink()
    (mg / "labels" / "train" / "leak.txt").unlink()
    inst = {s: {C.CLASSES[k]: 0 for k in range(11)} for s in ("train", "val")}
    for k, n in ((0, 2), (1, 1), (2, 1)):
        inst["train"][C.CLASSES[k]] = n
    inst["val"]["Car"], inst["val"]["Bus"] = 1, 1
    man = {"images": {"train": 4, "val": 2}, "instances_per_class": inst}
    (mg.parent / "merged_t.manifest.json").write_text(json.dumps(man), encoding="utf-8")
    r = run(["--data", str(mg), "--no-sample"])
    c = counts(r)
    print("4b:", c)
    assert not c.get("leak_train_val") and not c.get("manifest_mismatch") and not c.get("manifest_missing")
    man["images"]["train"] = 99
    (mg.parent / "merged_t.manifest.json").write_text(json.dumps(man), encoding="utf-8")
    assert counts(run(["--data", str(mg), "--no-sample"])).get("manifest_mismatch") == 1

    # ---------------------------------------------------------------- 5) mẫu người xem: tạo -> chấm -> kết luận
    ok = tmp / "ok"
    for i in range(12):
        put(ok, "", f"k{i}", noise_img(), [line(i % 11, rect(100, 100, 300, 250))])
    r = run(["--data", str(ok)])
    print("5a:", r["verdict"], r["human_audit"])
    assert r["verdict"] == "PASS_AUTO" and r["human_audit"]["pending"] == 12
    with open(ok / "qa" / "sample.csv", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    for i, row in enumerate(rows):
        row["verdict"] = "bad" if i < 3 else "ok"
        row["error_type"] = "sot" if i < 3 else ""
    with open(ok / "qa" / "sample.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=Q.SAMPLE_COLUMNS)
        wr.writeheader()
        wr.writerows(rows)
    r = run(["--data", str(ok)])
    print("5b:", r["verdict"], r["human_audit"])
    assert r["verdict"] == "FAIL" and counts(r)["human_audit"] == 1 and r["human_audit"]["error_types"] == {"sot": 3}
    for row in rows:
        row["verdict"] = "ok"
    with open(ok / "qa" / "sample.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=Q.SAMPLE_COLUMNS)
        wr.writeheader()
        wr.writerows(rows)
    assert run(["--data", str(ok)])["verdict"] == "PASS"
    run(["--data", str(ok), "--resample"])                       # bản đã chấm được giữ lại, không bị ghi đè
    assert len(list((ok / "qa").glob("sample.*.csv"))) == 1

    # ---------------------------------------------------------------- 6) label_audit: so khớp nhãn với dự đoán
    args = SimpleNamespace(conf_low=0.25, conf_miss=0.6, conf_cls=0.5, match_iou=0.5, good_iou=0.7, max_covered=0.2,
                           spurious=True)
    gt = [(0, rect(100, 100, 300, 250)),        # ô tô: mô hình đồng ý -> không báo
          (4, rect(400, 100, 460, 250)),        # Motorcycle nhưng mô hình nói Motorcyclist -> sai lớp
          (2, rect(20, 20, 220, 90)),           # mô hình chỉ thấy một nửa -> biên lệch
          (1, rect(480, 270, 620, 350))]        # mô hình không thấy gì -> nhãn thừa
    preds = [(0, 0.9, rect(102, 102, 298, 248)),
             (5, 0.8, rect(400, 95, 462, 250)),
             (2, 0.7, rect(20, 20, 150, 90)),
             (8, 0.85, rect(330, 150, 360, 300)),   # người không có nhãn -> sót nhãn
             (8, 0.3, rect(10, 300, 40, 350))]      # conf thấp -> không báo
    found = LA.audit_image(gt, preds, W, H, args)
    types = sorted((f["type"], f["gt_line"]) for f in found)
    print("6:", types)
    assert types == [("loose_boundary", 3), ("missing_label", ""), ("spurious_label", 4), ("wrong_class", 2)]
    pf = tmp / "p.txt"
    LA.write_preds(pf, [(c, s, p) for c, s, p in preds])
    back = LA.read_preds(pf)
    assert len(back) == 5 and abs(back[1][1] - 0.8) < 1e-3 and back[1][0] == 5
    print("ALL OK")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
