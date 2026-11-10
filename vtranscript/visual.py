"""Thu thập ngữ cảnh hình ảnh: chữ trên slide, tên người tham gia, thương hiệu, thuật ngữ...

Đây là nguồn "chính tả chuẩn" rất mạnh cho bước chuẩn hóa: ASR nghe "giên nhi" nhưng màn hình ghi
"Jane Nhi"; ASR nghe "ác ti vây sừn" nhưng slide ghi "EVENT ACTIVATION".
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import media
from .llm import LLM
from .prompts import VISUAL_SCHEMA, VISUAL_SYSTEM, visual_prompt
from .util import log, ts

KEYS = ("people", "organizations", "products", "terms", "slide_lines")


def pick_keyframes(
    video: Path, duration: float, interval: float, max_frames: int, threshold: float, cache: Path | None = None,
    small_threshold: float = 0.8, max_gap: float = 180.0,
) -> list[float]:
    """Chọn khung hình đáng đọc.

    - Ưu tiên khung có thay đổi lớn (chuyển slide, đổi màn hình chia sẻ).
    - Thêm khung có thay đổi nhỏ (đổi người nói -> nhãn tên đổi) và ít nhất 1 khung mỗi `max_gap` giây,
      để cả những đoạn màn hình gần như đứng yên vẫn được đọc tên người tham gia.
    """
    if cache is not None and cache.exists():
        thumbs = np.load(cache)
    else:
        thumbs = media.sample_thumbs(video, interval)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, thumbs)
    if len(thumbs) == 0:
        return []
    primary, secondary = [0], []
    last = 0
    for i in range(1, len(thumbs)):
        # So với khung đã giữ gần nhất (không phải khung liền trước) để bắt cả thay đổi diễn ra từ từ.
        d = float(np.mean(np.abs(thumbs[i] - thumbs[last])))
        if d > threshold:
            primary.append(i)
            last = i
        elif d > small_threshold or (i - last) * interval >= max_gap:
            secondary.append(i)
            last = i

    def even(xs: list[int], n: int) -> list[int]:
        if n <= 0 or not xs:
            return []
        if len(xs) <= n:
            return xs
        return [xs[k] for k in sorted(set(np.linspace(0, len(xs) - 1, n).round().astype(int).tolist()))]

    if len(primary) + len(secondary) <= max_frames:
        kept = primary + secondary
    else:
        n_sec = min(len(secondary), max(max_frames // 4, max_frames - len(primary)))
        kept = even(primary, max_frames - n_sec) or [0]
        # Khung phụ: mỗi lần chọn khung nằm XA các khung đã chọn nhất -> lấp các khoảng trống dài.
        cand = set(secondary) - set(kept)
        while len(kept) < max_frames and cand:
            best = max(cand, key=lambda c: min(abs(c - k) for k in kept))
            kept.append(best)
            cand.discard(best)
    kept = sorted(set(kept))
    # +interval/2: lấy giữa khoảng để tránh khung chuyển cảnh mờ.
    return [min(duration - 0.5, k * interval + interval / 2) for k in kept]


def collect(
    video: Path,
    work: Path,
    llm: LLM,
    duration: float,
    interval: float = 4.0,
    max_frames: int = 40,
    threshold: float = 5.0,
    batch: int = 5,
    workers: int = 3,
) -> dict:
    times = pick_keyframes(video, duration, interval, max_frames, threshold, cache=work / f"thumbs_{interval}.npy")
    log(f"Ngữ cảnh hình ảnh: chọn {len(times)} khung hình khác biệt để đọc chữ")
    frames_dir = work / "frames"
    frames = []
    for t in times:
        f = frames_dir / f"frame_{int(t):05d}.jpg"
        if not f.exists():
            media.extract_frame(video, t, f)
        frames.append({"t": t, "file": f})

    batches = [frames[i : i + batch] for i in range(0, len(frames), batch)]

    def run(b: list[dict]) -> dict:
        label = f"khung {ts(b[0]['t'])}–{ts(b[-1]['t'])}"
        return llm.ask(
            visual_prompt([ts(x["t"]) for x in b]),
            system=VISUAL_SYSTEM,
            schema=VISUAL_SCHEMA,
            images=[x["file"] for x in b],
            effort="medium",
            label=label,
        )

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(run, batches), 1):
            results.append(res)
            log(f"  đã đọc {i}/{len(batches)} nhóm khung hình")

    merged: dict[str, dict[str, int]] = {k: {} for k in KEYS}
    for res in results:
        for k in KEYS:
            for item in res.get(k) or []:
                item = " ".join(str(item).split())
                if not item:
                    continue
                # Gộp không phân biệt hoa/thường, giữ cách viết gặp đầu tiên.
                key = next((x for x in merged[k] if x.lower() == item.lower()), item)
                merged[k][key] = merged[k].get(key, 0) + 1
    out = {k: [x for x, _ in sorted(v.items(), key=lambda kv: -kv[1])] for k, v in merged.items()}
    out["frames"] = [{"t": f["t"], "file": f["file"].name} for f in frames]
    log(
        "  tìm thấy: "
        + ", ".join(f"{len(out[k])} {name}" for k, name in
                    zip(KEYS, ("tên người", "tổ chức", "sản phẩm", "thuật ngữ", "dòng chữ slide")))
    )
    return out
