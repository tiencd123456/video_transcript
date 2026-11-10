"""Nhận dạng giọng nói bằng faster-whisper (mặc định large-v3, chạy GPU nếu có).

Mỗi lần nhận dạng chạy trong một tiến trình con: nếu GPU hết bộ nhớ, CUDA context của tiến trình đó hỏng
(các lệnh CUDA sau đều lỗi "invalid device ordinal"), nên phải thử lại với batch nhỏ hơn ở tiến trình mới.
"""

from __future__ import annotations

import ctypes
import glob
import json
import os
import re
import site
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from .util import log, read_json, sha, ts, write_json


def _preload_cuda_libs() -> None:
    """ctranslate2 cần cuBLAS/cuDNN của CUDA 12. Khi cài qua pip (nvidia-cublas-cu12, nvidia-cudnn-cu12),
    các .so nằm trong site-packages chứ không nằm trong LD_LIBRARY_PATH — nạp trước bằng đường dẫn đầy đủ."""
    roots = [*site.getsitepackages(), site.getusersitepackages()]
    for pattern in ("nvidia/cublas/lib/libcublasLt.so*", "nvidia/cublas/lib/libcublas.so*", "nvidia/cudnn/lib/libcudnn*.so*"):
        for root in roots:
            for lib in sorted(glob.glob(os.path.join(root, pattern))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def load_wav(path: Path) -> np.ndarray:
    """Đọc WAV PCM 16-bit mono 16 kHz (do bước tách audio tạo ra) thành float32 [-1, 1].

    Tự đọc thay vì để faster-whisper giải mã bằng PyAV: tránh lệ thuộc phiên bản PyAV (PyAV 19 bỏ tham số
    `metadata_errors` mà faster-whisper 1.2 vẫn truyền vào).
    """
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1 or w.getframerate() != 16000:
            raise ValueError(f"WAV phải là PCM 16-bit mono 16 kHz: {path}")
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def _gpu_mem_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return int(out.split()[0])
    except Exception:  # noqa: BLE001
        return None


def auto_batch_size() -> int:
    """large-v3 int8_float16: batch 4 cần ~3,1 GB VRAM, batch 8 vượt 4 GB."""
    mem = _gpu_mem_mib()
    if mem is None:
        return 4
    if mem < 3500:
        return 0
    if mem < 7000:
        return 4
    if mem < 12000:
        return 8
    return 16


# ----------------------------------------------------------------------------------------------
# Tiến trình con: chạy đúng một lần nhận dạng
# ----------------------------------------------------------------------------------------------


def _run_once(cfg: dict) -> dict:
    """Nhận dạng audio từ giây `offset` đến hết. Mỗi đoạn nhận dạng xong được ghi ngay (JSON Lines, mốc thời gian
    tuyệt đối) vào file `progress` — nếu tiến trình chết giữa chừng (GPU hết bộ nhớ…), lần sau chạy tiếp từ đó."""
    _preload_cuda_libs()
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    device, compute_type, bs = cfg["device"], cfg["compute_type"], cfg["batch_size"]
    duration = cfg["duration"]
    off = float(cfg.get("offset") or 0.0)
    model = WhisperModel(cfg["model"], device=device, compute_type=compute_type, cpu_threads=os.cpu_count() or 4)
    kwargs = dict(
        language=cfg["language"],
        beam_size=cfg["beam_size"],
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=300),
        word_timestamps=True,
        # Không nối ngữ cảnh đoạn trước -> tránh vòng lặp "ảo giác" lặp câu, rất hay gặp với audio họp dài.
        condition_on_previous_text=False,
        hotwords=cfg["hotwords"] or None,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    samples = load_wav(Path(cfg["audio"]))
    if off > 0:
        samples = samples[int(off * 16000) :]
    if bs and bs > 1:
        segments_iter, info = BatchedInferencePipeline(model=model).transcribe(samples, batch_size=bs, **kwargs)
    else:
        segments_iter, info = model.transcribe(samples, hallucination_silence_threshold=2.0, **kwargs)

    t0 = time.time()
    next_report = off + 120.0
    with open(cfg["progress"], "a", encoding="utf-8") as prog:
        for seg in segments_iter:
            rec = _seg_record(seg, off)
            prog.write(json.dumps(rec, ensure_ascii=False) + "\n")
            prog.flush()
            end = rec["end"]
            if end >= next_report:
                speed = (end - off) / max(1e-6, time.time() - t0)
                eta = (duration - end) / max(1e-6, speed)
                log(f"  ASR {ts(end)}/{ts(duration)} ({end / duration:.0%}) — x{speed:.1f} realtime, "
                    f"còn ~{eta / 60:.0f} phút")
                next_report = end + 240
    return {"language": info.language, "language_probability": round(info.language_probability, 3)}


def _seg_record(seg, off: float) -> dict:
    return {
        "start": round(seg.start + off, 3),
        "end": round(seg.end + off, 3),
        "text": seg.text.strip(),
        "avg_logprob": round(seg.avg_logprob, 3),
        "no_speech_prob": round(seg.no_speech_prob, 3),
        "compression_ratio": round(seg.compression_ratio, 3),
        "words": [
            {"s": round(w.start + off, 3), "e": round(w.end + off, 3), "w": w.word, "p": round(w.probability, 3)}
            for w in (seg.words or [])
        ],
    }


def _run_repair(cfg: dict) -> dict:
    """Nhận dạng lại các quãng nghi ảo giác bằng chế độ tuần tự (không batch): có temperature fallback khi
    câu ra bị lặp/khó tin, và `hallucination_silence_threshold` bỏ chữ "bịa" ở chỗ im lặng."""
    _preload_cuda_libs()
    from faster_whisper import WhisperModel

    model = WhisperModel(cfg["model"], device=cfg["device"], compute_type=cfg["compute_type"],
                         cpu_threads=os.cpu_count() or 4)
    samples = load_wav(Path(cfg["audio"]))
    out = []
    for a, b in cfg["windows"]:
        clip = samples[int(a * 16000) : int(b * 16000)]
        segs, _ = model.transcribe(
            clip, language=cfg["language"], beam_size=cfg["beam_size"], vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200), word_timestamps=True,
            condition_on_previous_text=False, temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold=2.4, log_prob_threshold=-1.0, no_speech_threshold=0.6,
            hallucination_silence_threshold=2.0,
        )
        out.append([_seg_record(s, a) for s in segs])
    return {"windows": out}


def _worker_main(cfg_path: str) -> int:
    cfg = read_json(Path(cfg_path))
    write_json(Path(cfg["out"]), _run_repair(cfg) if cfg.get("mode") == "repair" else _run_once(cfg))
    return 0


# Câu Whisper hay "bịa" ra ở chỗ im lặng/nhạc (học từ phụ đề YouTube) — nhất là với tiếng Việt.
HALLUCINATION_RE = re.compile(
    r"subscribe|ghiền mì gõ|đăng ký kênh|bỏ lỡ (những|các) video|vietsub|phụ đề (bởi|được)|"
    r"cảm ơn (các bạn|mọi người) đã (theo dõi|xem|lắng nghe)|hẹn gặp lại (các bạn )?(trong|ở) (những |các )?video",
    re.I,
)


def is_suspicious(seg: dict) -> bool:
    return seg.get("compression_ratio", 0) > 2.4 or bool(HALLUCINATION_RE.search(seg.get("text", "")))


def repair_hallucinations(asr: dict, audio: Path) -> dict:
    """Ở chế độ batch, faster-whisper không thử lại khi câu ra bị lặp/ảo giác, nên có khi cả quãng 20-30 giây
    lời nói thật bị thay bằng "Hãy subscribe cho kênh…". Tìm các đoạn đó, nhận dạng lại từng quãng bằng chế độ
    tuần tự; câu nhận dạng lại vẫn là ảo giác thì bỏ."""
    segs = asr["segments"]
    bad = [i for i, s in enumerate(segs) if is_suspicious(s)]
    asr["repairs"] = []
    if not bad:
        return asr
    groups: list[list[int]] = []
    for i in bad:
        if groups and i == groups[-1][-1] + 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    windows = []
    for g in groups:
        prev_end = segs[g[0] - 1]["end"] if g[0] > 0 else 0.0
        next_start = segs[g[-1] + 1]["start"] if g[-1] + 1 < len(segs) else asr["duration"]
        a = max(prev_end, segs[g[0]]["start"] - 0.3)
        b = min(next_start, segs[g[-1]]["end"] + 0.3)
        windows.append([round(a, 3), round(b, 3)])
    log(f"Sửa ảo giác ASR: nhận dạng lại {len(windows)} quãng nghi ngờ "
        f"({sum(b - a for a, b in windows):.0f} giây) bằng chế độ tuần tự…")

    pkg_root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with tempfile.TemporaryDirectory(prefix="vtranscript-repair-") as tmp:
        cfg = {"mode": "repair", "audio": str(Path(audio).resolve()), "model": asr["model"], "device": asr["device"],
               "compute_type": asr["compute_type"], "language": asr.get("language"), "beam_size": 5,
               "windows": windows, "out": str(Path(tmp) / "out.json")}
        write_json(Path(tmp) / "cfg.json", cfg)
        proc = subprocess.run([sys.executable, "-m", "vtranscript.asr", str(Path(tmp) / "cfg.json")],
                              stderr=subprocess.PIPE, text=True, env=env)
        if proc.returncode != 0:
            log(f"  ! Không nhận dạng lại được, giữ nguyên (bước chuẩn hóa sẽ xử lý): {proc.stderr[-500:]}")
            return asr
        redone = read_json(Path(cfg["out"]))["windows"]

    new_segs: list[dict] = []
    gi = {g[0]: k for k, g in enumerate(groups)}
    skip: set[int] = {i for g in groups for i in g}
    for i, s in enumerate(segs):
        if i in gi:
            k = gi[i]
            kept = [r for r in redone[k] if r["text"] and not is_suspicious(r)]
            old = " ".join(segs[j]["text"] for j in groups[k])
            new = " ".join(r["text"] for r in kept)
            asr["repairs"].append({"start": windows[k][0], "end": windows[k][1], "old": old, "new": new})
            log(f"  {ts(windows[k][0])}: «{old[:60]}…» → «{new[:80] or '(bỏ — không có lời nói)'}»")
            new_segs.extend(resegment(kept))
        elif i not in skip:
            new_segs.append(s)
    for i, s in enumerate(new_segs):
        s["id"] = i
    asr["segments"] = new_segs
    return asr


def resegment(segments: list[dict], max_dur: float = 15.0, min_dur: float = 3.0) -> list[dict]:
    """Chế độ batch của faster-whisper trả về đoạn dài ~30 giây. Tách lại thành đoạn cỡ câu dựa vào mốc thời gian
    từng từ: ngắt sau dấu kết câu hoặc ở khoảng lặng; đoạn quá dài thì ngắt ở khoảng lặng lớn nhất."""
    out: list[dict] = []

    def emit(seg: dict, words: list[dict]) -> None:
        text = "".join(w["w"] for w in words).strip()
        if text:
            out.append({**seg, "start": words[0]["s"], "end": words[-1]["e"], "text": text, "words": words})

    for seg in segments:
        words = seg.get("words") or []
        if not words or seg["end"] - seg["start"] <= max_dur:
            out.append(seg)
            continue
        cur: list[dict] = []
        for i, w in enumerate(words):
            cur.append(w)
            nxt = words[i + 1] if i + 1 < len(words) else None
            if nxt is None:
                break
            dur = w["e"] - cur[0]["s"]
            gap = nxt["s"] - w["e"]
            end_sent = w["w"].rstrip()[-1:] in ".?!…"
            if dur >= min_dur and (end_sent or gap >= 0.6):
                emit(seg, cur)
                cur = []
            elif dur >= max_dur:
                # Ngắt ở khoảng lặng lớn nhất trong nửa sau của đoạn đang gom.
                half = len(cur) // 2
                k = max(range(half, len(cur)), key=lambda j: (words_gap(cur, j, nxt)))
                emit(seg, cur[: k + 1])
                cur = cur[k + 1 :]
        if cur:
            emit(seg, cur)
    for i, s in enumerate(out):
        s["id"] = i
    return out


def words_gap(cur: list[dict], j: int, nxt: dict) -> float:
    after = cur[j + 1]["s"] if j + 1 < len(cur) else nxt["s"]
    return after - cur[j]["e"]


# ----------------------------------------------------------------------------------------------
# Tiến trình chính
# ----------------------------------------------------------------------------------------------


def transcribe(
    audio: Path,
    duration: float,
    model_name: str = "large-v3",
    language: str | None = "vi",
    device: str = "auto",
    compute_type: str = "auto",
    hotwords: str | None = None,
    beam_size: int = 5,
    batch_size: int = -1,
) -> dict:
    if device == "auto":
        device = "cuda" if _gpu_mem_mib() else "cpu"
    if compute_type == "auto":
        compute_type = "int8_float16" if device == "cuda" else "int8"
    if batch_size < 0:
        batch_size = auto_batch_size() if device == "cuda" else 8
    log(f"ASR: model={model_name} device={device} compute={compute_type} batch={batch_size} lang={language or 'auto'}")
    if hotwords:
        log(f"  hotwords: {hotwords[:200]}{'…' if len(hotwords) > 200 else ''}")

    sizes = [batch_size] + [b for b in (4, 2, 0) if b < batch_size]
    pkg_root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    # Tiến độ dở dang nằm cạnh file audio (thư mục _work) để cả khi tool bị tắt hẳn, lần chạy sau vẫn làm tiếp.
    key = sha(model_name, language, hotwords, beam_size, compute_type, os.path.getsize(audio))
    progress = Path(audio).parent / f"asr_partial_{key}.jsonl"
    done = _read_progress(progress)
    if done:
        log(f"  tiếp tục ASR dở dang từ {ts(done[-1]['end'])} ({len(done)} đoạn đã có)")
    t0 = time.time()
    info: dict = {}
    level, failures = 0, 0
    with tempfile.TemporaryDirectory(prefix="vtranscript-asr-") as tmp:
        while True:
            offset = done[-1]["end"] if done else 0.0
            if offset >= duration - 1.0:
                break
            bs = sizes[level]
            # Ghi lại file tiến độ chỉ gồm các dòng hợp lệ: dòng cuối có thể bị ghi dở khi tiến trình trước chết,
            # nếu để nguyên thì dòng mới sẽ bị nối vào sau nó.
            progress.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in done), encoding="utf-8")
            cfg = {"audio": str(Path(audio).resolve()), "duration": duration, "model": model_name, "language": language,
                   "device": device, "compute_type": compute_type, "hotwords": hotwords, "beam_size": beam_size,
                   "batch_size": bs, "offset": offset, "progress": str(progress), "out": str(Path(tmp) / "out.json")}
            cfg_path = Path(tmp) / "cfg.json"
            write_json(cfg_path, cfg)
            proc = subprocess.Popen([sys.executable, "-m", "vtranscript.asr", str(cfg_path)],
                                    stderr=subprocess.PIPE, text=True, env=env)
            tail: list[str] = []
            assert proc.stderr is not None
            for line in proc.stderr:
                sys.stderr.write(line)
                tail = (tail + [line])[-40:]
            rc = proc.wait()
            done = _read_progress(progress)
            if rc == 0:
                info = read_json(Path(cfg["out"]))
                break
            err = "".join(tail)
            gpu_err = any(k in err.lower() for k in ("out of memory", "cuda", "cublas", "cudnn"))
            failures += 1
            if not gpu_err or failures > 8:
                raise RuntimeError(f"ASR thất bại (batch_size={bs}, đã chạy tới {ts(offset)}):\n{err[-1500:]}")
            progressed = (done[-1]["end"] if done else 0.0) - offset
            # Chạy được một quãng rồi mới lỗi -> nhiều khả năng chỉ thiếu bộ nhớ nhất thời: giữ batch, chạy tiếp.
            # Lỗi ngay -> hạ batch.
            if progressed < 120 and level + 1 < len(sizes):
                level += 1
            log(f"  ! Lỗi GPU với batch_size={bs} — chạy tiếp từ {ts(done[-1]['end'] if done else 0.0)} "
                f"với batch_size={sizes[level]} ở tiến trình mới")

    segments = resegment([s for s in done if s["text"]])
    progress.unlink(missing_ok=True)
    log(f"ASR xong: {len(segments)} đoạn trong {(time.time() - t0) / 60:.1f} phút")
    return {
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "batch_size": sizes[level],
        "language": info.get("language", language),
        "language_probability": info.get("language_probability"),
        "duration": duration,
        "hotwords": hotwords,
        "segments": segments,
    }


def _read_progress(path: Path) -> list[dict]:
    """Đọc các đoạn đã nhận dạng (bỏ dòng cuối nếu bị ghi dở khi tiến trình chết)."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break
    out.sort(key=lambda s: s["start"])
    return out


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1]))
