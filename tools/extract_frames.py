"""Video -> khung hình cho dữ liệu Việt Nam.

Lấy 1 khung mỗi --every giây, bỏ --skip-start/--skip-end giây đầu/cuối (intro, logo kênh), bỏ khung mờ,
tối đa --max-per-video khung (rải đều cả video), resize cạnh dài 1280.
Tên file: vn_<id video>_<thời điểm tính bằng 1/10 giây>.jpg

Ví dụ:
    python tools/extract_frames.py --videos D:/data/raw/vn_videos --out D:/data/raw/vn_frames
    python tools/extract_frames.py --videos a.mp4 b.mp4 --out D:/data/raw/vn_frames_gold --prefix vngold
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

import common as C

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}


def video_id(path):
    """yt-dlp đặt tên kiểu 'Tiêu đề [abcXYZ123].mp4': ưu tiên lấy id trong ngoặc vuông."""
    m = re.search(r"\[([A-Za-z0-9_-]{6,})\]", path.stem)
    raw = m.group(1) if m else path.stem
    return re.sub(r"[^A-Za-z0-9-]", "", raw)[:40] or "video"


def blur_score(img):
    g = cv2.cvtColor(cv2.resize(img, (640, round(640 * img.shape[0] / img.shape[1]))), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def extract(path, out_dir, prefix, every, skip_start, skip_end, max_frames, blur_thresh, long_side):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  BỎ QUA {path.name}: không mở được")
        return 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n / fps
    t = np.arange(skip_start, max(skip_start, dur - skip_end), every)
    if len(t) > max_frames:
        t = t[np.linspace(0, len(t) - 1, max_frames).round().astype(int)]
    wanted = sorted(set((t * fps).round().astype(int)))
    vid = video_id(path)
    saved = blurry = 0
    idx, wi = 0, 0
    while wi < len(wanted):
        if not cap.grab():
            break
        if idx == wanted[wi]:
            wi += 1
            ok, frame = cap.retrieve()
            if ok:
                if blur_score(frame) < blur_thresh:
                    blurry += 1
                else:
                    frame = C.resize_image(frame, C.resize_scale(*frame.shape[:2], long_side))
                    C.save_image(Path(out_dir) / f"{prefix}_{vid}_{round(idx / fps * 10):06d}.jpg", frame)
                    saved += 1
        idx += 1
    cap.release()
    print(f"  {path.name[:60]:<60} {dur / 60:5.1f} phút -> {saved} khung (bỏ {blurry} khung mờ)")
    return saved, blurry


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True, help="thư mục hoặc danh sách file video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="vn")
    ap.add_argument("--every", type=float, default=2.0, help="giây giữa hai khung")
    ap.add_argument("--skip-start", type=float, default=10.0)
    ap.add_argument("--skip-end", type=float, default=10.0)
    ap.add_argument("--max-per-video", type=int, default=150)
    ap.add_argument("--blur-thresh", type=float, default=60.0, help="phương sai Laplacian nhỏ hơn = mờ")
    ap.add_argument("--long-side", type=int, default=C.DEFAULT_LONG_SIDE)
    args = ap.parse_args()

    videos = []
    for v in args.videos:
        p = Path(v)
        videos += sorted(q for q in p.rglob("*") if q.suffix.lower() in VIDEO_EXTS) if p.is_dir() else [p]
    total = 0
    for v in videos:
        total += extract(v, args.out, args.prefix, args.every, args.skip_start, args.skip_end,
                         args.max_per_video, args.blur_thresh, args.long_side)[0]
    print(f"Tổng: {total} khung từ {len(videos)} video -> {args.out}. Bước tiếp: dedup.py")


if __name__ == "__main__":
    main()
