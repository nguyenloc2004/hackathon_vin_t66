"""Viewer cho các bộ dataset YOLO-seg 11 lớp của đội, kèm kết quả QA và chấm mẫu người xem.

Xem ảnh + mask từng instance, lọc theo lớp / mã QA / mẫu người xem / nghi vấn của label_audit, chấm verdict cho
qa/sample.csv ngay trên giao diện. Chỉ dùng thư viện chuẩn + tools/common.py; chỉ nghe ở 127.0.0.1.

    python tools/viewer/server.py --data D:/data/converted/vistas D:/data/converted/idd
    python tools/viewer/server.py --root D:/data            # tự tìm mọi thư mục có images/ (sâu tối đa 4 cấp)
    -> mở http://localhost:8000

Cờ QA và mẫu người xem lấy từ <dataset>/qa/ do tools/qa_check.py tạo; nghi vấn L6 từ <dataset>/qa/audit/.
Chấm xong thì chạy lại qa_check.py để tính kết luận.
"""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

QA_DIR = "qa"
SAMPLE_COLUMNS = ["key", "split", "stem", "reason", "verdict", "error_type", "note"]  # = qa_check.SAMPLE_COLUMNS
ERROR_TYPES = ["sot", "sai_lop", "bien", "rider", "khac"]
SKIP_DIRS = {"qa", "preview", "audit", "labels", "__MACOSX"}
LOCK = threading.Lock()


def rgb_hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


# ------------------------------------------------------------------------------------------------ tìm dataset
def find_datasets(roots, max_depth=4):
    found = []

    def walk(d, depth):
        if (d / "images").is_dir():
            found.append(d)
            return
        if depth >= max_depth:
            return
        try:
            subs = sorted(p for p in d.iterdir() if p.is_dir() and p.name not in SKIP_DIRS and not p.name.startswith("."))
        except OSError:
            return
        for p in subs:
            walk(p, depth + 1)

    for r in roots:
        walk(Path(r).resolve(), 0)
    return found


def find_splits(root):
    images = root / "images"
    subs = {p.name for p in images.iterdir() if p.is_dir()} if images.is_dir() else set()
    named = [s for s in ("train", "val") if s in subs]
    return named or [""]


# ------------------------------------------------------------------------------------------------ đọc dữ liệu
def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_sample(root):
    path = root / QA_DIR / "sample.csv"
    if not path.is_file():
        return {}, path
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return {r["key"]: r for r in csv.DictReader(fh) if r.get("key")}, path


def audit_split(root):
    """label_audit chạy với --split thì stem không có tên split; còn lại là 'all'."""
    summ = read_json(root / QA_DIR / "audit" / "audit_summary.json", {})
    return summ.get("split") or "all"


def read_audit(root):
    path = root / QA_DIR / "audit" / "findings.csv"
    out = {}
    if not path.is_file():
        return out
    sp = audit_split(root)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(f"{sp}/{r['stem']}", []).append(r)
    return out


def parse_label(path, with_conf=False):
    """-> list dict {i: số dòng (đếm như qa_check), c: lớp, p: tọa độ phẳng, s: conf}. Dòng hỏng bị bỏ qua."""
    out = []
    if not path.is_file():
        return out
    for ln, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        conf = vals.pop() if len(vals) % 2 else None   # dự đoán của label_audit có conf ở cuối dòng
        if not with_conf:
            conf = None
        out.append({"i": ln, "c": cls, "p": [round(v, 5) for v in vals], "s": conf})
    return out


def count_classes(path):
    counts = [0] * len(C.CLASSES)
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split(maxsplit=1)
            if parts:
                try:
                    c = int(float(parts[0]))
                except ValueError:
                    continue
                if 0 <= c < len(counts):
                    counts[c] += 1
    return counts


class Dataset:
    def __init__(self, did, path, name):
        self.id, self.path, self.name = did, Path(path), name
        self.index = None
        self.by_key = {}

    def build(self):
        root = self.path
        flags = read_json(root / QA_DIR / "flags.json", {})
        sample, _ = read_sample(root)
        audit = read_audit(root)
        images = []
        by_key = {}
        for split in find_splits(root):
            sname = split or "all"
            images_root = root / "images" / split if split else root / "images"
            labels_root = root / "labels" / split if split else root / "labels"
            for img in C.list_images(images_root):
                rel = img.relative_to(images_root).with_suffix("").as_posix()
                key = f"{sname}/{rel}"
                lab = C.label_path_for(img, images_root, labels_root)
                fl = flags.get(key, [])
                rec = {"k": key, "s": sname, "n": rel, "c": count_classes(lab),
                       "q": sorted({f[0] for f in fl if f[1] != "waived"}),
                       "e": any(f[1] == "error" for f in fl),
                       "sm": sample.get(key), "a": sorted({r["type"] for r in audit.get(key, [])})}
                images.append(rec)
                by_key[key] = (img, lab, rel, sname)
        report = read_json(root / QA_DIR / "report.json", None)
        summary = None
        if report:
            summary = {"verdict": report.get("verdict"), "profile": report.get("profile"),
                       "errors": report.get("errors"), "warnings": report.get("warnings"),
                       "created_at": report.get("created_at"), "human_audit": report.get("human_audit"),
                       "checks": {k: {"level": v["level"], "count": v["count"], "message": v["message"],
                                      "waived": v.get("waived", 0)} for k, v in report.get("checks", {}).items()}}
        self.index = {"id": self.id, "name": self.name, "path": str(root), "images": images, "qa": summary,
                      "has_flags": bool(flags), "has_sample": bool(sample), "has_audit": bool(audit),
                      "has_preds": (root / QA_DIR / "audit" / "preds").is_dir()}
        self.by_key = by_key
        return self.index

    def label(self, key):
        if self.index is None:
            self.build()
        if key not in self.by_key:
            return None
        img, lab, rel, sname = self.by_key[key]
        root = self.path
        flags = read_json(root / QA_DIR / "flags.json", {}).get(key, [])
        pat = re.compile(re.escape(f"{sname}/{rel}.txt") + r":(\d+)")   # mục của qa_check: "<split>/<nhãn>.txt:<dòng>"
        line_flags = {}
        for code, lvl, item in flags:
            for m in pat.finditer(item):
                line_flags.setdefault(m.group(1), []).append([code, lvl])
        preds = []
        if (root / QA_DIR / "audit" / "preds").is_dir():
            preds = parse_label(root / QA_DIR / "audit" / "preds" / f"{rel}.txt", with_conf=True)
        sample, _ = read_sample(root)
        return {"key": key, "img": img.relative_to(root).as_posix(), "lines": parse_label(lab),
                "flags": flags, "line_flags": line_flags, "audit": read_audit(root).get(key, []),
                "preds": preds, "sample": sample.get(key)}

    def save_verdict(self, key, verdict, error_type, note):
        with LOCK:
            rows, path = read_sample(self.path)
            if key not in rows:
                raise KeyError(key)
            with open(path, newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                fields = reader.fieldnames or SAMPLE_COLUMNS
                all_rows = list(reader)
            for r in all_rows:
                if r.get("key") == key:
                    r["verdict"], r["error_type"], r["note"] = verdict, error_type if verdict == "bad" else "", note
                    row = r
            tmp = path.with_suffix(".csv.tmp")
            with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(all_rows)
            tmp.replace(path)
            if self.index is not None:
                for rec in self.index["images"]:
                    if rec["k"] == key:
                        rec["sm"] = row
            return row


# ------------------------------------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    datasets: dict = {}

    def log_message(self, fmt, *args):   # yên lặng, trừ lỗi
        pass

    def send_json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path, ctype=None):
        try:
            data = Path(path).read_bytes()
        except OSError:
            return self.send_json({"error": "not found"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", ctype or mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def ds(self, q):
        d = self.datasets.get((q.get("ds") or [""])[0])
        if d is None:
            self.send_json({"error": "không có dataset này"}, 404)
        return d

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                return self.send_file(HERE / "viewer.html", "text/html; charset=utf-8")
            if u.path == "/api/meta":
                return self.send_json({
                    "classes": C.CLASSES, "colors": [rgb_hex(c) for c in C.CLASS_COLORS], "error_types": ERROR_TYPES,
                    "datasets": [{"id": d.id, "name": d.name, "path": str(d.path)} for d in self.datasets.values()]})
            if u.path == "/api/index":
                d = self.ds(q)
                if d is None:
                    return None
                if d.index is None or q.get("refresh"):
                    d.build()
                return self.send_json(d.index)
            if u.path == "/api/label":
                d = self.ds(q)
                if d is None:
                    return None
                res = d.label((q.get("k") or [""])[0])
                return self.send_json(res) if res else self.send_json({"error": "không có ảnh"}, 404)
            if u.path == "/img":
                d = self.ds(q)
                if d is None:
                    return None
                p = (d.path / (q.get("p") or [""])[0]).resolve()
                if not p.is_relative_to(d.path.resolve()) or p.suffix.lower() not in C.IMG_EXTS:
                    return self.send_json({"error": "forbidden"}, 403)
                return self.send_file(p)
            return self.send_json({"error": "not found"}, 404)
        except (ConnectionError, BrokenPipeError):
            return None

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/sample":
            return self.send_json({"error": "not found"}, 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            return self.send_json({"error": "JSON hỏng"}, 400)
        d = self.datasets.get(str(body.get("ds", "")))
        verdict = body.get("verdict", "")
        etype = body.get("error_type", "")
        if d is None or verdict not in ("ok", "bad", "") or (etype and etype not in ERROR_TYPES):
            return self.send_json({"error": "tham số không hợp lệ"}, 400)
        try:
            row = d.save_verdict(body.get("key", ""), verdict, etype, str(body.get("note", ""))[:500])
        except KeyError:
            return self.send_json({"error": "ảnh không có trong qa/sample.csv"}, 404)
        except OSError as e:
            return self.send_json({"error": f"không ghi được sample.csv (đang mở trong Excel?): {e}"}, 409)
        return self.send_json({"ok": True, "row": row})


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="*", default=[], help="các thư mục dataset (có images/, labels/)")
    ap.add_argument("--root", nargs="*", default=[], help="thư mục gốc để tự tìm dataset")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    paths = [Path(p).resolve() for p in args.data] + find_datasets(args.root)
    paths = list(dict.fromkeys(p for p in paths if (p / "images").is_dir()))
    if not paths:
        raise SystemExit("Không tìm thấy dataset nào (cần thư mục có images/). Dùng --data hoặc --root.")
    names = {}
    for i, p in enumerate(paths):
        base = next((str(p.relative_to(Path(r).resolve())) for r in args.root
                     if p.is_relative_to(Path(r).resolve()) and p != Path(r).resolve()), p.name)
        name = base.replace("\\", "/")
        names[str(i)] = Dataset(str(i), p, name if name not in {d.name for d in names.values()} else f"{name} ({i})")
    Handler.datasets = names
    for d in names.values():
        print(f"  [{d.id}] {d.name}  ({d.path})")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Viewer: http://localhost:{args.port}   (Ctrl+C để dừng)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
