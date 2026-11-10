"""Nhận dạng giọng nói qua API speech-to-text của OpenRouter (mặc định ElevenLabs Scribe v2), không cần GPU.

Audio được chia thành các phần khoảng 20 phút, cắt ở chỗ lặng nhất gần mốc chia để không cắt ngang từ.
Mỗi phần được nén mp3 32 kbps rồi gửi riêng, chạy song song. Kết quả từng phần lưu trong `_work/asr_api/`,
nên nếu tool bị dừng giữa chừng thì lần chạy sau không phải trả tiền lại cho phần đã xong.
Kết quả trả về cùng dạng với `asr.transcribe`: các đoạn cỡ câu, có mốc thời gian và độ tin cậy từng từ.
"""

from __future__ import annotations

import base64
import json
import math
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .asr import is_suspicious, load_wav, resegment
from .llm import openrouter_key
from .media import ffmpeg_exe
from .util import log, read_json, sha, ts, write_json

DEFAULT_MODEL = "elevenlabs/scribe-v2"
URL = "https://openrouter.ai/api/v1/audio/transcriptions"
CHUNK_S = 20 * 60          # độ dài mỗi phần gửi đi
SEARCH_S = 20.0            # tìm chỗ lặng trong khoảng ±SEARCH_S quanh mốc chia
PAUSE_SPLIT_S = 1.5        # khoảng lặng giữa hai từ dài hơn mức này thì tách đoạn trước khi chia câu


def _cut_points(audio: Path, duration: float) -> list[float]:
    """Mốc chia các phần, mỗi mốc đặt vào quãng 0,5 giây lặng nhất gần mốc chia đều."""
    n = max(1, math.ceil(duration / CHUNK_S))
    if n == 1:
        return [0.0, duration]
    x = load_wav(audio)
    hop = 1600  # 0,1 giây
    m = len(x) // hop
    rms = np.sqrt((x[: m * hop].reshape(m, hop) ** 2).mean(axis=1))
    smooth = np.convolve(rms, np.ones(5) / 5, mode="same")
    cuts = [0.0]
    for k in range(1, n):
        t = duration * k / n
        lo, hi = int((t - SEARCH_S) * 10), int((t + SEARCH_S) * 10)
        lo, hi = max(lo, 0), min(hi, m - 1)
        cuts.append(round((lo + int(np.argmin(smooth[lo:hi]))) / 10, 2) if hi > lo else t)
    return cuts + [duration]


def _encode(audio: Path, a: float, b: float) -> bytes:
    r = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-v", "error", "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}", "-i", str(audio),
         "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "32k", "-f", "mp3", "pipe:1"],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Nén audio thất bại: {r.stderr.decode(errors='replace')[-800:]}")
    return r.stdout


def _post(key: str, body: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "X-Title": "vtranscript"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:800]
            if e.code < 500 and e.code != 429:
                raise RuntimeError(f"API nhận dạng giọng nói lỗi HTTP {e.code}: {msg}") from e
            err = f"HTTP {e.code}: {msg}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = str(e)
        wait = 10 * 2 ** attempt
        log(f"  ! Lỗi gọi API ({err[:200]}), thử lại sau {wait} giây")
        time.sleep(wait)
    raise RuntimeError(f"API nhận dạng giọng nói lỗi liên tục: {err[:800]}")


def _chunk(key: str, audio: Path, a: float, b: float, model: str, language: str | None, cache_dir: Path) -> dict:
    cache = cache_dir / f"{sha(model, language, round(a, 2), round(b, 2), audio.stat().st_size)}.json"
    if cache.exists():
        return read_json(cache)
    data = _encode(audio, a, b)
    body = {"model": model, "input_audio": {"data": base64.b64encode(data).decode(), "format": "mp3"},
            "response_format": "verbose_json", "timestamp_granularities": ["word", "segment"]}
    if language:
        body["language"] = language
    t0 = time.time()
    resp = _post(key, body)
    if resp.get("error"):
        raise RuntimeError(f"API nhận dạng giọng nói lỗi: {str(resp['error'])[:800]}")
    resp["_offset"] = a
    write_json(cache, resp)
    cost = float((resp.get("usage") or {}).get("cost") or 0)
    log(f"  phần {ts(a)}–{ts(b)}: {len(resp.get('words') or [])} từ, {time.time() - t0:.0f} giây, ${cost:.3f}")
    return resp


def _words(resp: dict) -> list[dict]:
    off = resp["_offset"]
    out = []
    for w in resp.get("words") or []:
        tok = (w.get("word") or "").strip()
        if not tok or w.get("type") == "audio_event":
            continue
        conf = w.get("confidence")
        out.append({"s": round(off + float(w["start"]), 3), "e": round(off + float(w["end"]), 3), "w": " " + tok,
                    "p": round(float(conf), 3) if conf is not None else 1.0})
    return out


def _segments(words: list[dict]) -> list[dict]:
    """Gom từ thành đoạn ở các khoảng lặng dài, rồi chia thành đoạn cỡ câu (≤15 giây) như bản Whisper."""
    groups: list[list[dict]] = []
    for w in words:
        if groups and w["s"] - groups[-1][-1]["e"] < PAUSE_SPLIT_S:
            groups[-1].append(w)
        else:
            groups.append([w])
    segs = [{"start": g[0]["s"], "end": g[-1]["e"], "text": "".join(w["w"] for w in g).strip(), "words": g}
            for g in groups]
    return resegment(segs)


def transcribe_api(audio: Path, duration: float, model: str | None = None, language: str | None = "vi",
                   workers: int = 4) -> dict:
    model = model or DEFAULT_MODEL
    key = openrouter_key()
    cuts = _cut_points(audio, duration)
    spans = list(zip(cuts[:-1], cuts[1:]))
    log(f"ASR qua API: {model} (OpenRouter), {len(spans)} phần, ngôn ngữ {language or 'tự nhận'}")
    cache_dir = Path(audio).parent / "asr_api"
    cache_dir.mkdir(exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(spans)))) as ex:
        resps = list(ex.map(lambda ab: _chunk(key, Path(audio), ab[0], ab[1], model, language, cache_dir), spans))
    words = [w for r in resps for w in _words(r)]
    words.sort(key=lambda w: w["s"])
    segments = _segments(words)
    cost = sum(float((r.get("usage") or {}).get("cost") or 0) for r in resps)
    sus = [s for s in segments if is_suspicious(s)]
    if sus:
        log(f"  ! {len(sus)} đoạn giống câu ảo giác, để bước chuẩn hóa xử lý: "
            + "; ".join(f"{ts(s['start'])} «{s['text'][:50]}»" for s in sus[:5]))
    log(f"ASR xong: {len(segments)} đoạn, {len(words)} từ trong {(time.time() - t0) / 60:.1f} phút, ${cost:.3f}")
    return {
        "model": model,
        "device": "api",
        "compute_type": None,
        "batch_size": None,
        "language": resps[0].get("language") or language,
        "language_probability": resps[0].get("language_confidence"),
        "duration": duration,
        "hotwords": None,
        "segments": segments,
        "cost_usd": round(cost, 4),
        "repairs": [],  # cách sửa ảo giác bằng cách chạy lại Whisper trên máy không áp dụng cho API
    }
