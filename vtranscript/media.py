"""Các thao tác ffmpeg: đọc thông tin video, tách audio, lấy khung hình, ghép phụ đề cứng."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np


def ffmpeg_exe() -> str:
    # Mặc định dùng bản ffmpeg tĩnh của imageio-ffmpeg vì bản này có sẵn libass (cần để in phụ đề).
    env = os.environ.get("VT_FFMPEG")
    if env:
        return env
    from imageio_ffmpeg import get_ffmpeg_exe

    return get_ffmpeg_exe()


def _run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run([ffmpeg_exe(), "-hide_banner", *args], **kw)


def probe(video: Path) -> dict:
    """Đọc thời lượng, kích thước, fps từ stderr của `ffmpeg -i` (imageio-ffmpeg không kèm ffprobe)."""
    r = _run(["-i", str(video)], capture_output=True, text=True, errors="replace")
    err = r.stderr
    info: dict = {"duration": 0.0, "width": 0, "height": 0, "fps": 25.0, "has_audio": " Audio:" in err}
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if m:
        info["duration"] = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    for line in err.splitlines():
        if " Video:" in line:
            m = re.search(r"\b(\d{2,5})x(\d{2,5})\b", line)
            if m:
                info["width"], info["height"] = int(m[1]), int(m[2])
            m = re.search(r"(\d+(?:\.\d+)?)\s*fps", line)
            if m:
                info["fps"] = float(m[1])
            break
    if not info["duration"]:
        raise RuntimeError(f"Không đọc được thông tin video: {video}\n{err[-800:]}")
    return info


def extract_audio(video: Path, out_wav: Path) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_wav.with_name(out_wav.stem + ".part.wav")
    r = _run(
        ["-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(tmp)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Tách audio thất bại:\n{r.stderr[-1500:]}")
    tmp.replace(out_wav)


def sample_thumbs(video: Path, interval: float, w: int = 64, h: int = 36) -> np.ndarray:
    """Lấy ảnh xám rất nhỏ mỗi `interval` giây để phát hiện chuyển slide/cảnh. Trả về mảng [N, h, w]."""
    r = _run(
        ["-v", "error", "-i", str(video), "-an", "-vf", f"fps=1/{interval},scale={w}:{h},format=gray",
         "-f", "rawvideo", "-"],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Lấy khung hình thất bại:\n{r.stderr.decode(errors='replace')[-1500:]}")
    buf = np.frombuffer(r.stdout, dtype=np.uint8)
    n = len(buf) // (w * h)
    return buf[: n * w * h].reshape(n, h, w).astype(np.float32)


def extract_frame(video: Path, t: float, out_jpg: Path, width: int = 1600) -> None:
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    r = _run(
        ["-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
         "-vf", f"scale='min({width},iw)':-2", "-q:v", "3", str(out_jpg)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not out_jpg.exists():
        raise RuntimeError(f"Không lấy được khung hình tại {t:.1f}s:\n{r.stderr[-800:]}")


def burn_subtitles(
    video: Path,
    ass_file: Path,
    out_mp4: Path,
    fonts_dir: Path | None,
    duration: float,
    crf: int = 22,
    preset: str = "veryfast",
    on_progress: Callable[[float], None] | None = None,
) -> None:
    """In cứng phụ đề ASS vào video (libx264), giữ nguyên audio gốc.

    ffmpeg chạy với cwd = thư mục chứa file .ass để tránh phải escape đường dẫn có dấu cách/tiếng Việt
    trong filtergraph.
    """
    vf = f"ass={ass_file.name}"
    if fonts_dir:
        vf += f":fontsdir={_filter_escape(str(fonts_dir))}"
    tmp = out_mp4.with_name(out_mp4.stem + ".part.mp4")
    args = [
        ffmpeg_exe(), "-hide_banner", "-y", "-v", "error", "-nostats", "-progress", "pipe:1",
        "-i", str(video.resolve()), "-vf", vf,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-movflags", "+faststart", str(tmp.resolve()),
    ]
    proc = subprocess.Popen(args, cwd=ass_file.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith("out_time_us=") and on_progress and duration:
            try:
                on_progress(min(1.0, int(line.split("=", 1)[1]) / 1e6 / duration))
            except ValueError:
                pass
    err = proc.stderr.read() if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"Ghép phụ đề thất bại:\n{err[-2000:]}")
    tmp.replace(out_mp4)


def _filter_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace(",", "\\,")
