"""Chèn ảnh màn hình (slide, trang được chia sẻ...) vào bản tóm tắt, theo từng mục có mốc thời gian.

Không gọi LLM. Cách chọn ảnh:
- Chia video thành các "cảnh": quãng màn hình gần như không đổi (dùng lại ảnh xám thu nhỏ mỗi 4 giây).
- Với mỗi mục "### [hh:mm:ss] ..." của bản tóm tắt, lấy cảnh hiển thị LÂU NHẤT trong quãng thời gian của mục
  (mục dài thì lấy tối đa 2 cảnh khác nhau).
- Bỏ khung ít nội dung (chỉ có avatar, màn hình đen...) bằng cách đo mật độ cạnh của ảnh, và không lặp lại
  một màn hình đã dùng ở mục trước.

Ảnh được đặt giữa hai dấu `<!-- vt:img -->` ... `<!-- /vt:img -->` nên chạy lại sẽ thay thế, không chèn trùng.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import media
from .util import log, ts

IMG_DIR = "03_tom_tat_hinh"
THUMB_INTERVAL = 4.0  # trùng với visual.collect để dùng chung cache thumbs_4.0.npy
SCENE_THRESHOLD = 5.0  # chênh lệch sáng trung bình (0–255) coi là đổi màn hình
MIN_DETAIL = 5.0  # dưới mức này là khung trống/avatar (slide, trang web thường 8–16)
MIN_DWELL = 10.0  # cảnh phải hiện ít nhất chừng này giây trong mục mới đáng chèn
DUP_THRESHOLD = 2.5  # hai cảnh chênh ít hơn mức này coi là cùng một màn hình
LONG_SECTION = 150.0  # mục dài hơn được chèn tối đa 2 ảnh

HEAD_RE = re.compile(r"^(#{2,6})\s+\[?((?:\d{1,2}:)?\d{1,2}:\d{2})\]?\s*(.*)$")
BLOCK_RE = re.compile(r"\n\n<!-- vt:img -->.*?<!-- /vt:img -->", re.S)


def strip_images(md: str) -> str:
    return BLOCK_RE.sub("", md)


def _secs(s: str) -> float:
    v = 0
    for p in s.split(":"):
        v = v * 60 + int(p)
    return float(v)


def _detail(jpg: Path, w: int = 480, h: int = 270) -> float:
    """Mật độ cạnh trung bình — slide/trang web nhiều chữ cho điểm cao, avatar/màn hình trống cho điểm thấp."""
    r = subprocess.run(
        [media.ffmpeg_exe(), "-v", "error", "-i", str(jpg), "-vf", f"scale={w}:{h},format=gray",
         "-f", "rawvideo", "-"],
        capture_output=True,
    )
    buf = np.frombuffer(r.stdout, np.uint8)
    if buf.size != w * h:
        return 0.0
    g = buf.reshape(h, w).astype(np.float32)
    return float(np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean())


def _scenes(thumbs: np.ndarray) -> list[dict]:
    starts, last = [0], 0
    for i in range(1, len(thumbs)):
        # So với đầu cảnh hiện tại (không phải khung liền trước) để bắt cả thay đổi từ từ.
        if float(np.mean(np.abs(thumbs[i] - thumbs[last]))) > SCENE_THRESHOLD:
            starts.append(i)
            last = i
    out = []
    for k, i in enumerate(starts):
        j = starts[k + 1] if k + 1 < len(starts) else len(thumbs)
        start, end = i * THUMB_INTERVAL, j * THUMB_INTERVAL
        # Lấy ảnh sau lúc chuyển cảnh vài giây (tránh khung đang chuyển), nhưng vẫn trong cảnh.
        t = start + min((end - start) / 2, 6.0)
        idx = min(int(t / THUMB_INTERVAL), len(thumbs) - 1)
        out.append({"start": start, "end": end, "t": t, "thumb": thumbs[idx]})
    return out


def _sections(lines: list[str], end_limit: float) -> list[dict]:
    heads = []
    for k, ln in enumerate(lines):
        m = HEAD_RE.match(ln)
        if m:
            heads.append({"line": k, "t": _secs(m.group(2)), "title": m.group(3).strip()})
    for n, h in enumerate(heads):
        later = [x["t"] for x in heads[n + 1 :] if x["t"] > h["t"]]
        h["end"] = min(later[0] if later else end_limit, end_limit)
    return heads


def add_images(summary_md: Path, video: Path, work: Path, end_limit: float, max_per_section: int = 2) -> int:
    """Chèn ảnh vào `summary_md` (ghi đè file). Trả về số ảnh đã chèn."""
    md = strip_images(summary_md.read_text(encoding="utf-8"))
    lines = md.split("\n")
    sections = _sections(lines, end_limit)
    img_dir = summary_md.parent / IMG_DIR
    if img_dir.exists():
        shutil.rmtree(img_dir)
    if not sections:
        summary_md.write_text(md, encoding="utf-8")
        log("  ! Bản tóm tắt không có mục nào kèm mốc thời gian, bỏ qua chèn ảnh")
        return 0

    cache = work / f"thumbs_{THUMB_INTERVAL}.npy"
    if cache.exists():
        thumbs = np.load(cache)
    else:
        thumbs = media.sample_thumbs(video, THUMB_INTERVAL)
        np.save(cache, thumbs)
    scenes = _scenes(thumbs)
    frame_dir = work / "summary_frames"
    used: list[np.ndarray] = []
    detail_cache: dict[float, float] = {}

    def frame(sc: dict) -> tuple[Path, float]:
        f = frame_dir / f"scene_{int(sc['t'] * 10):06d}.jpg"
        if not f.exists():
            media.extract_frame(video, sc["t"], f, width=1280)
        if sc["t"] not in detail_cache:
            detail_cache[sc["t"]] = _detail(f)
        return f, detail_cache[sc["t"]]

    total, filled = 0, 0
    for sec in sections:
        t0, t1 = sec["t"], sec["end"]
        cands = []
        for sc in scenes:
            dwell = min(sc["end"], t1) - max(sc["start"], t0)
            if dwell >= MIN_DWELL:
                cands.append((dwell, sc))
        cands.sort(key=lambda x: -x[0])
        limit = max_per_section if t1 - t0 > LONG_SECTION else 1
        picked = []
        for _, sc in cands:
            if len(picked) >= limit:
                break
            if any(float(np.mean(np.abs(sc["thumb"] - u))) < DUP_THRESHOLD for u in used):
                continue
            f, d = frame(sc)
            if d < MIN_DETAIL:
                continue
            picked.append((sc, f))
            used.append(sc["thumb"])
        if not picked:
            continue
        parts = ["<!-- vt:img -->"]
        for sc, f in sorted(picked, key=lambda x: x[0]["t"]):
            name = f"{ts(sc['t']).replace(':', '-')}.jpg"
            img_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, img_dir / name)
            alt = sec["title"].replace("[", "(").replace("]", ")") or "Màn hình"
            parts += [f"![{alt}]({IMG_DIR}/{name})", f"*Màn hình lúc {ts(sc['t'])}*"]
        parts.append("<!-- /vt:img -->")
        lines[sec["line"]] += "\n\n" + "\n\n".join(parts)
        total += len(picked)
        filled += 1
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    log(f"  chèn {total} ảnh màn hình vào {filled}/{len(sections)} mục của bản tóm tắt")
    return total
