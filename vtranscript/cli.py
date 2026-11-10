"""vtranscript — lấy transcript từ video, chuẩn hóa kỹ bằng LLM, tóm tắt và in phụ đề vào video.

Đầu ra (trong thư mục output/<tên video>/):
  01_transcript_goc.txt / .srt          transcript gốc từ ASR
  02_transcript_chuan_hoa.txt / .srt    transcript đã chuẩn hóa (chính tả, tên riêng, thuật ngữ...)
  02_bao_cao_chuan_hoa.md               báo cáo: từ điển, từng chỗ đã sửa, chỗ cần kiểm tra lại
  03_tom_tat.md                         tổng hợp nội dung & ý chính (kèm ảnh slide/màn hình trong 03_tom_tat_hinh/)
  04_video_phu_de.mp4                   video đã in phụ đề từ bản chuẩn hóa
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from . import __version__, media
from .util import log, nfc, read_json, ts, write_json, write_text

STEPS = ("audio", "visual", "asr", "normalize", "summary", "video")
LANG_NAMES = {"vi": "tiếng Việt", "en": "tiếng Anh"}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vtranscript",
        description="Transcript video -> chuẩn hóa (chính tả, tên riêng, thuật ngữ) -> tóm tắt -> video có phụ đề.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("video", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None, help="Thư mục đầu ra (mặc định output/<tên video>)")
    p.add_argument("--lang", default="vi", help="Mã ngôn ngữ nói trong video (vi, en, ...)")
    p.add_argument("--profile", default=None,
                   help="Bộ model đặt sẵn (đè lên .env): best, budget, cheap, tên hồ sơ tự tạo trong ./profiles/ hoặc "
                        "~/.config/vtranscript/profiles/, hoặc đường dẫn một file .env")
    p.add_argument("--reuse-from", type=Path, default=None,
                   help="Dùng lại dữ liệu trung gian của một lần chạy khác trên CÙNG video (thư mục output của lần đó): "
                        "audio, khung hình, kết quả ASR/đọc khung hình nếu cùng model, và các lượt gọi LLM đã cache. "
                        "Dùng khi thử model khác để không trả tiền lại cho các bước không đổi")
    g = p.add_argument_group("ASR")
    g.add_argument("--asr-backend", default="auto", choices=["auto", "api", "local"],
                   help="api: gọi API speech-to-text của OpenRouter (không cần GPU); local: faster-whisper trên máy. "
                        "auto: api nếu có OPENROUTER_API_KEY, không thì local")
    g.add_argument("--asr-model", default=None,
                   help="Mặc định: elevenlabs/scribe-v2 (api) / large-v3 (local)")
    g.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    g.add_argument("--compute-type", default="auto")
    g.add_argument("--batch-size", type=int, default=-1, help="-1 = tự chọn theo VRAM; 0 = chạy tuần tự (chậm hơn, ít tốn VRAM)")
    g.add_argument("--no-hotwords", action="store_true", help="Không mồi tên riêng/thuật ngữ cho ASR")
    g = p.add_argument_group("Chuẩn hóa")
    g.add_argument("--glossary", type=Path, default=None,
                   help="File từ điển: mỗi dòng 'Cách viết chuẩn = biến thể 1, biến thể 2'")
    g.add_argument("--llm-backend", default="auto", choices=["auto", "openrouter", "claude-cli", "anthropic"])
    g.add_argument("--llm-model", default=None,
                   help="Model chính: sửa từng cụm, vòng soát, tóm tắt. Mặc định: deepseek/deepseek-v4.1-flash "
                        "(openrouter) / opus (claude-cli) / claude-opus-5-5 (anthropic)")
    g.add_argument("--llm-strong-model", default=None,
                   help="Model lập từ điển tên riêng/thuật ngữ (quyết định chất lượng chuẩn hóa). Mặc định: "
                        "anthropic/claude-opus-5.5 (openrouter); backend khác: như --llm-model")
    g.add_argument("--llm-vision-model", default=None,
                   help="Model đọc chữ trên khung hình. Mặc định: anthropic/claude-sonnet-5.5 (openrouter); "
                        "backend khác: như --llm-model")
    g.add_argument("--workers", type=int, default=4, help="Số lượt gọi LLM song song")
    g.add_argument("--no-visual", action="store_true", help="Không đọc chữ trên màn hình/slide làm ngữ cảnh")
    g.add_argument("--max-frames", type=int, default=50)
    g.add_argument("--no-review", action="store_true", help="Bỏ vòng soát toàn cục")
    g.add_argument("--verbatim", action="store_true", help="Giữ nguyên từ đệm (à, ờ, kiểu như là...)")
    g.add_argument("--summary-lang", default="tiếng Việt")
    g.add_argument("--no-summary-images", action="store_true",
                   help="Không chèn ảnh màn hình (slide...) vào bản tóm tắt")
    g = p.add_argument_group("Video")
    g.add_argument("--skip-video", action="store_true")
    g.add_argument("--font", default=None, help="Tên font phụ đề (mặc định Arial nếu có, không thì DejaVu Sans)")
    g.add_argument("--font-size", type=int, default=None)
    g.add_argument("--crf", type=int, default=22)
    g.add_argument("--preset", default="veryfast")
    p.add_argument("--redo", choices=STEPS, default=None, help="Chạy lại từ bước này (bỏ cache của bước đó trở đi)")
    return p.parse_args(argv)


def build_hotwords(user_gl: list[dict], visual: dict | None, limit: int = 300) -> str:
    """Tên riêng/thuật ngữ mồi cho Whisper. Giữ ngắn — prompt quá dài làm Whisper dễ ảo giác.

    Ưu tiên: từ điển người dùng > tổ chức/sản phẩm > thuật ngữ > vài tên người dài (tên ngắn kiểu "Min"
    có thể kéo Whisper viết nhầm từ thông dụng như "mình"; tên người được sửa kỹ ở bước chuẩn hóa).
    """
    def ok(x: str) -> bool:
        # Bỏ tiêu đề slide viết hoa toàn bộ dài dòng (không giúp nhận dạng), giữ viết tắt ngắn như KPI, UID.
        return len(x) <= 24 and not (x.isupper() and len(x) > 8)

    cands = [u["canonical"] for u in user_gl]
    if visual:
        # Thứ tự trong mỗi nhóm đã theo tần suất xuất hiện trên màn hình.
        cands += [x for x in visual.get("organizations", []) if ok(x)][:6]
        cands += [x for x in visual.get("terms", []) if ok(x)][:30]
        cands += [x for x in visual.get("products", []) if ok(x)][:6]
        cands += [x for x in visual.get("people", []) if len(x) >= 6][:4]
    out, seen, n = [], set(), 0
    for c in cands:
        key = "".join(c.lower().split())
        if key in seen or n + len(c) + 2 > limit:
            continue
        seen.add(key)
        out.append(c)
        n += len(c) + 2
    return ", ".join(out)


def raw_txt(asr: dict, video_name: str) -> str:
    lines = [f"# Transcript gốc (ASR {asr['model']}) — {video_name}", ""]
    lines += [f"[{ts(s['start'])}] {s['text']}" for s in asr["segments"]]
    return "\n".join(lines) + "\n"


def norm_txt(segments: list[dict], video_name: str) -> str:
    from .summarize import paragraphs

    lines = [f"# Transcript đã chuẩn hóa — {video_name}", ""]
    for t, para in paragraphs(segments, max_chars=600):
        lines += [f"[{ts(t)}] {para}", ""]
    return "\n".join(lines)


def pick_font(work: Path, requested: str | None) -> tuple[str, Path | None]:
    fonts = work / "fonts"
    if requested:
        return requested, None
    win = Path("/mnt/c/Windows/Fonts")
    if (win / "arial.ttf").exists():
        fonts.mkdir(parents=True, exist_ok=True)
        for f in ("arial.ttf", "arialbd.ttf"):
            if (win / f).exists() and not (fonts / f).exists():
                shutil.copy(win / f, fonts / f)
        return "Arial", fonts
    for d in (Path("/usr/share/fonts"), Path("/usr/local/share/fonts")):
        if list(d.rglob("Arial*.ttf")) if d.exists() else []:
            return "Arial", None
    return "DejaVu Sans", None


def reuse_work(src: Path, work: Path, vision_model: str | None, asr_model: str) -> None:
    """Liên kết (hardlink) dữ liệu trung gian của lần chạy `src` sang `work`.

    An toàn vì mọi file trong _work đều được ghi kiểu tạo file tạm rồi đổi tên, không ghi đè lên file cũ. Cache
    ASR qua API và cache từng lượt gọi LLM có khóa theo model + nội dung nên lấy hết; còn kết quả đọc khung hình
    và ASR thô chỉ lấy khi cùng model với lần chạy này."""
    srcw = src.resolve() / "_work"
    if not srcw.is_dir():
        raise SystemExit(f"--reuse-from: không thấy {srcw}")
    if srcw == work.resolve():
        return

    def link(f: Path, dst: Path) -> bool:
        if dst.exists() or not f.is_file():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(f, dst)
        except OSError:
            shutil.copy2(f, dst)
        return True

    got = []
    if link(srcw / "audio_16k.wav", work / "audio_16k.wav"):
        got.append("audio")
    for f in srcw.glob("thumbs_*.npy"):
        link(f, work / f.name)
    n = {d: sum(link(f, work / d / f.name) for f in (srcw / d).glob("*")) for d in
         ("frames", "asr_api", "normalize", "summary") if (srcw / d).is_dir()}
    if n.get("frames"):
        got.append(f"{n['frames']} khung hình")
    if n.get("asr_api"):
        got.append(f"{n['asr_api']} phần ASR đã cache")
    if n.get("normalize", 0) + n.get("summary", 0):
        got.append(f"{n.get('normalize', 0) + n.get('summary', 0)} lượt gọi LLM đã cache")
    v = srcw / "visual_context.json"
    if v.is_file() and vision_model:
        vm = read_json(v).get("model")
        if vm == vision_model and link(v, work / v.name):
            got.append(f"ngữ cảnh hình ảnh ({vm})")
        elif vm != vision_model:
            log(f"  đọc lại khung hình: lần trước dùng {vm or 'model không rõ'}, lần này {vision_model}")
    r = srcw / "asr_raw.json"
    if r.is_file() and read_json(r).get("model") == asr_model and link(r, work / r.name):
        got.append(f"ASR thô ({asr_model})")
    log(f"Dùng lại từ {src}: {', '.join(got) or 'không có gì'}")


def main(argv=None) -> int:
    from . import config

    a = parse_args(argv)
    try:
        cfg_files = config.load(a.profile)
        config.apply_to_args(a)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    video = a.video.resolve()
    if not video.exists() and video.parent.is_dir():
        # Tên file tiếng Việt có thể lưu dạng Unicode NFD (macOS/zip) trong khi chuỗi gõ vào là NFC.
        want = nfc(video.name)
        video = next((f for f in video.parent.iterdir() if nfc(f.name) == want), video)
    if not video.exists():
        print(f"Không thấy file video: {video}", file=sys.stderr)
        return 2
    out = (a.output or Path("output") / nfc(video.stem)).resolve()
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    lang_name = LANG_NAMES.get(a.lang, a.lang)

    redo_from = STEPS.index(a.redo) if a.redo else len(STEPS)

    def fresh(step: str, *files: Path) -> bool:
        """True nếu bước đã có kết quả và không bị yêu cầu chạy lại."""
        return STEPS.index(step) < redo_from and all(f.exists() for f in files)

    info = media.probe(video)
    log(f"Video: {video.name} — {ts(info['duration'])}, {info['width']}x{info['height']}, audio={info['has_audio']}")
    if cfg_files:
        log("Cấu hình: " + " + ".join(config.describe(f) for f in cfg_files))
    if not info["has_audio"]:
        print("Video không có âm thanh.", file=sys.stderr)
        return 2

    from .normalize import load_user_glossary

    user_gl = load_user_glossary(a.glossary)
    pool = None

    def get_pool():
        nonlocal pool
        if pool is None:
            from .llm import LLMPool

            pool = LLMPool(a.llm_backend, {"main": a.llm_model, "strong": a.llm_strong_model,
                                           "vision": a.llm_vision_model})
        return pool

    def get_llm(role: str = "main"):
        return get_pool().get(role)

    asr_backend = a.asr_backend if a.asr_backend != "auto" else (
        "api" if os.environ.get("OPENROUTER_API_KEY") else "local")
    if asr_backend == "api":
        from .asr_api import DEFAULT_MODEL as API_ASR_MODEL

        asr_model = a.asr_model or API_ASR_MODEL
    else:
        asr_model = a.asr_model or "large-v3"
    if a.reuse_from:
        reuse_work(a.reuse_from, work, None if a.no_visual else get_pool().models["vision"], asr_model)
    if asr_backend == "local" and not fresh("asr", work / "asr_raw.json"):
        import importlib.util

        if importlib.util.find_spec("faster_whisper") is None:
            print("Chưa cài ASR chạy trên máy (faster-whisper). Hoặc điền OPENROUTER_API_KEY vào .env để gọi ASR qua "
                  "API, hoặc cài thêm: pip install 'vtranscript[local]' (máy có GPU NVIDIA: 'vtranscript[gpu]').",
                  file=sys.stderr)
            return 2

    # 1) Audio
    wav = work / "audio_16k.wav"
    if not fresh("audio", wav):
        log("Tách âm thanh 16 kHz mono…")
        media.extract_audio(video, wav)

    # 2) Ngữ cảnh hình ảnh (chữ trên slide/màn hình) — dùng để mồi ASR và làm nguồn chính tả cho bước chuẩn hóa
    visual = None
    vfile = work / "visual_context.json"
    if not a.no_visual:
        if fresh("visual", vfile):
            visual = read_json(vfile)
            want = get_pool().models["vision"]
            if visual.get("model") and visual["model"] != want:
                log(f"  ! Ngữ cảnh hình ảnh đã có do {visual['model']} đọc, khác model đang chọn ({want}). "
                    "Muốn đọc lại thì thêm --redo visual")
        else:
            from .visual import collect

            try:
                vllm = get_llm("vision")
                visual = collect(video, work, vllm, info["duration"], max_frames=a.max_frames)
                visual["model"] = vllm.model
                write_json(vfile, visual)
            except Exception as e:  # noqa: BLE001 — thiếu ngữ cảnh hình ảnh vẫn chạy tiếp được
                log(f"  ! Bỏ qua ngữ cảnh hình ảnh: {e}")

    # 3) ASR
    afile = work / "asr_raw.json"
    asr_cost = 0.0  # tiền ASR của lần chạy này (0 nếu lấy lại kết quả cũ)
    if fresh("asr", afile):
        asr = read_json(afile)
        if asr.get("model") != asr_model:
            log(f"  ! Transcript gốc đã có do {asr.get('model')} nhận dạng, khác model đang chọn ({asr_model}). "
                "Muốn nhận dạng lại thì thêm --redo asr")
    elif asr_backend == "api":
        from .asr_api import transcribe_api

        asr = transcribe_api(wav, info["duration"], asr_model, a.lang, workers=a.workers)
        asr_cost = asr.get("cost_usd") or 0.0
        write_json(afile, asr)
    else:
        from .asr import transcribe

        hot = None if a.no_hotwords else (build_hotwords(user_gl, visual) or None)
        asr = transcribe(wav, info["duration"], asr_model, a.lang, a.device, a.compute_type, hot,
                         batch_size=a.batch_size)
        write_json(afile, asr)
    if "repairs" not in asr:
        from .asr import repair_hallucinations

        asr = repair_hallucinations(asr, wav)
        write_json(afile, asr)
    write_text(out / "01_transcript_goc.txt", raw_txt(asr, video.name))
    from .subtitles import build_cues, timed_tokens, write_ass, write_srt

    raw_by_id = {s["id"]: s for s in asr["segments"]}
    write_srt(build_cues(timed_tokens(asr["segments"], raw_by_id)), out / "01_transcript_goc.srt")
    log(f"→ {out / '01_transcript_goc.txt'}")

    # 4) Chuẩn hóa
    nfile = work / "normalized.json"
    if fresh("normalize", nfile):
        norm = read_json(nfile)
    else:
        from .normalize import normalize, write_report

        norm = normalize(asr, visual, user_gl, get_llm(), work, lang_name, a.verbatim, a.workers,
                         do_review=not a.no_review, glossary_llm=get_llm("strong"))
        write_json(nfile, norm)
        stats = write_report(norm, out / "02_bao_cao_chuan_hoa.md", video.name, asr["model"], asr.get("repairs"))
        log(f"  sửa {stats['changed']} đoạn, xóa {stats['removed']}, {stats['uncertain']} chỗ cần kiểm tra lại")
    if not (out / "02_bao_cao_chuan_hoa.md").exists():
        from .normalize import write_report

        write_report(norm, out / "02_bao_cao_chuan_hoa.md", video.name, asr["model"], asr.get("repairs"))
    segs = norm["segments"]
    write_text(out / "02_transcript_chuan_hoa.txt", norm_txt(segs, video.name))
    write_json(out / "02_transcript_chuan_hoa.json",
               {"video": video.name, "duration": info["duration"], "glossary": norm["glossary"],
                "segments": [{k: s[k] for k in ("id", "start", "end", "text", "raw")} for s in segs]})
    cues = build_cues(timed_tokens(segs, raw_by_id))
    write_srt(cues, out / "02_transcript_chuan_hoa.srt")
    log(f"→ {out / '02_transcript_chuan_hoa.txt'} ({len(cues)} câu phụ đề)")

    # 5) Tóm tắt
    sfile = out / "03_tom_tat.md"
    if not fresh("summary", sfile):
        from .summarize import summarize

        write_text(sfile, summarize(segs, norm["glossary"], get_llm(), work, a.summary_lang).strip() + "\n")
    from .summary_images import IMG_DIR, add_images, strip_images

    if a.no_summary_images:
        write_text(sfile, strip_images(sfile.read_text(encoding="utf-8")))
        shutil.rmtree(out / IMG_DIR, ignore_errors=True)
    else:
        speech_end = max((s["end"] for s in segs if (s.get("text") or "").strip()), default=info["duration"])
        try:
            add_images(sfile, video, work, min(info["duration"], speech_end + 5))
        except Exception as e:  # noqa: BLE001 — thiếu ảnh không làm hỏng bản tóm tắt
            log(f"  ! Bỏ qua chèn ảnh vào tóm tắt: {e}")
    log(f"→ {sfile}")

    # 6) Video có phụ đề
    vout = out / "04_video_phu_de.mp4"
    if not a.skip_video and not fresh("video", vout):
        font, fonts_dir = pick_font(work, a.font)
        ass = work / "subtitles.ass"
        write_ass(cues, ass, info["width"], info["height"], font, a.font_size)
        log(f"In phụ đề vào video (font {font}, libx264 crf={a.crf}, preset={a.preset})…")
        state = {"next": 0.0}

        def progress(f: float) -> None:
            if f >= state["next"]:
                log(f"  ghép phụ đề {f:.0%}")
                state["next"] = f + 0.1

        media.burn_subtitles(video, ass, vout, Path("fonts") if fonts_dir else None, info["duration"],
                             crf=a.crf, preset=a.preset, on_progress=progress)
    if vout.exists():
        log(f"→ {vout}")

    log(f"ASR: {asr['model']} " + (f"${asr_cost:.3f}" if asr_cost else "(dùng lại kết quả đã có, không tốn thêm)"))
    llm_cost = 0.0
    if pool is not None and pool.used:
        parts = ", ".join(f"{x.model} {x.calls} lượt ${x.cost_usd:.3f}" for x in pool.used)
        llm_cost = sum(x.cost_usd for x in pool.used)
        note = " (claude-cli: theo gói đăng ký, không tính thêm tiền)" if pool.backend == "claude-cli" else ""
        log(f"LLM: {parts}{note}")
    log(f"Chi phí lần chạy này (ước tính): ${asr_cost + llm_cost:.2f}")
    # Sổ chi phí: mỗi lần chạy một dòng, chỉ tính tiền thật sự tiêu ở lần đó (phần lấy từ cache không tính).
    import json
    import time

    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "profile": a.profile,
           "asr": {"model": asr.get("model"), "usd": round(asr_cost, 4)},
           "llm": [{"model": x.model, "calls": x.calls, "usd": round(x.cost_usd, 4)} for x in (pool.used if pool else [])]}
    with open(work / "chi_phi.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"Xong. Kết quả trong: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
