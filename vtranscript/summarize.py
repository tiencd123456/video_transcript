"""Tổng hợp nội dung & ý chính từ transcript đã chuẩn hóa."""

from __future__ import annotations

from pathlib import Path

from . import prompts as P
from .llm import LLM
from .util import log, read_json, sha, ts, write_json

PART_CHARS = 150_000  # transcript dài hơn mức này thì tóm tắt từng phần rồi gộp


def paragraphs(segments: list[dict], max_chars: int = 500, gap: float = 2.0) -> list[tuple[float, str]]:
    """Gộp các đoạn ASR ngắn thành đoạn văn có mốc thời gian (dễ đọc, ít token hơn)."""
    out: list[tuple[float, str]] = []
    cur, start, last_end = [], 0.0, None
    for s in segments:
        t = (s.get("text") or "").strip()
        if not t:
            continue
        n = sum(len(x) + 1 for x in cur)
        if cur and (
            (last_end is not None and s["start"] - last_end > gap)
            or n > max_chars * 1.6
            or (n > max_chars and cur[-1][-1:] in ".?!…")
        ):
            out.append((start, " ".join(cur)))
            cur = []
        if not cur:
            start = s["start"]
        cur.append(t)
        last_end = s["end"]
    if cur:
        out.append((start, " ".join(cur)))
    return out


def summarize(segments: list[dict], glossary: dict | None, llm: LLM, work: Path, out_lang: str = "tiếng Việt") -> str:
    lines = [f"[{ts(t)}] {txt}" for t, txt in paragraphs(segments)]
    context = P.glossary_block(glossary) if glossary else ""
    system = P.SUMMARY_SYSTEM.format(out_lang=out_lang)
    cache = work / "summary"
    cache.mkdir(parents=True, exist_ok=True)

    def ask(prompt: str, label: str) -> str:
        f = cache / f"{label}_{sha(system, prompt, llm.model)}.json"
        if f.exists():
            return read_json(f)["text"]
        text = llm.ask(prompt, system=system, effort="high", label=label)
        write_json(f, {"text": text})
        return text

    total = sum(len(x) + 1 for x in lines)
    if total <= PART_CHARS:
        log(f"Tóm tắt: transcript {total:,} ký tự, 1 lượt")
        return ask(P.summary_prompt(context, lines), "summary")

    # Rất dài: tóm tắt từng phần (giữ chi tiết + mốc giờ) rồi gộp.
    parts, cur, n = [], [], 0
    for x in lines:
        if cur and n + len(x) > PART_CHARS:
            parts.append(cur)
            cur, n = [], 0
        cur.append(x)
        n += len(x) + 1
    parts.append(cur)
    log(f"Tóm tắt: transcript {total:,} ký tự, chia {len(parts)} phần")
    partials = [
        ask(P.summary_prompt(context, p) + f"\n\n(Đây là PHẦN {i + 1}/{len(parts)} của video.)", f"summary_part{i}")
        for i, p in enumerate(parts)
    ]
    merge = (
        context
        + "Dưới đây là bản tổng hợp của từng phần liên tiếp trong cùng một video. Hãy gộp thành MỘT bản tổng hợp hoàn "
          "chỉnh theo đúng cấu trúc yêu cầu, giữ mốc thời gian và chi tiết quan trọng, bỏ trùng lặp.\n\n"
        + "\n\n---\n\n".join(partials)
    )
    return ask(merge, "summary_merge")
