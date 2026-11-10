"""Chuẩn hóa transcript: từ điển -> hiệu đính từng cụm -> soát toàn cục -> báo cáo thay đổi."""

from __future__ import annotations

import difflib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import prompts as P
from .llm import LLM, LLMError
from .util import clean_spaces, fold, log, nfc, read_json, sha, ts, write_json

LOW_CONF = 0.45  # xác suất từ (Whisper) dưới ngưỡng này được đánh dấu ‹ › để LLM soi kỹ


# ----------------------------------------------------------------------------------------------
# Từ điển người dùng
# ----------------------------------------------------------------------------------------------


def load_user_glossary(path: Path | None) -> list[dict]:
    """Định dạng mỗi dòng:  Cách viết chuẩn = biến thể 1, biến thể 2   (vế sau dấu = là tùy chọn; # là chú thích)."""
    if not path:
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        canon, _, rest = line.partition("=")
        variants = [v.strip() for v in rest.split(",") if v.strip()]
        out.append({"canonical": nfc(canon.strip()), "variants": [nfc(v) for v in variants]})
    return out


# ----------------------------------------------------------------------------------------------
# Chuẩn bị đầu vào
# ----------------------------------------------------------------------------------------------


def mark_low_confidence(seg: dict) -> str:
    words = seg.get("words") or []
    if not words:
        return seg["text"]
    out, run = [], []
    for w in words:
        tok = w["w"].strip()
        if not tok:
            continue
        if w["p"] < LOW_CONF:
            run.append(tok)
            continue
        if run:
            out.append(P.LOW_CONF_OPEN + " ".join(run) + P.LOW_CONF_CLOSE)
            run = []
        out.append(tok)
    if run:
        out.append(P.LOW_CONF_OPEN + " ".join(run) + P.LOW_CONF_CLOSE)
    return " ".join(out)


def _strip_marks(text: str) -> str:
    return clean_spaces(nfc(text.replace(P.LOW_CONF_OPEN, "").replace(P.LOW_CONF_CLOSE, "")))


def _chunks(segs: list[dict], max_chars: int) -> list[list[dict]]:
    out, cur, n = [], [], 0
    for s in segs:
        if cur and n + len(s["text"]) > max_chars:
            out.append(cur)
            cur, n = [], 0
        cur.append(s)
        n += len(s["text"]) + 12
    if cur:
        out.append(cur)
    return out


# ----------------------------------------------------------------------------------------------
# Vòng 1: từ điển
# ----------------------------------------------------------------------------------------------


def _merge_glossaries(parts: list[dict]) -> dict:
    if len(parts) == 1:
        return parts[0]
    merged = {
        "topic": " ".join(p.get("topic", "") for p in parts[:1]),
        "setting": parts[0].get("setting", ""),
        "style_notes": " ".join(dict.fromkeys(p.get("style_notes", "") for p in parts if p.get("style_notes"))),
        "speakers": list(dict.fromkeys(x for p in parts for x in p.get("speakers", []))),
        "entries": [],
    }
    index: dict[str, dict] = {}
    for p in parts:
        for e in p.get("entries", []):
            k = e["canonical"].lower()
            if k in index:
                index[k]["variants"] = list(dict.fromkeys(index[k]["variants"] + e.get("variants", [])))
            else:
                index[k] = dict(e)
                merged["entries"].append(index[k])
    return merged


def build_glossary(marked: list[str], visual, user_gl, llm: LLM, cache_dir: Path, lang_name: str) -> dict:
    # Transcript rất dài thì chia phần (~120k ký tự) rồi gộp từ điển.
    parts, cur, n = [], [], 0
    for line in marked:
        if cur and n + len(line) > 120_000:
            parts.append(cur)
            cur, n = [], 0
        cur.append(line)
        n += len(line) + 1
    parts.append(cur)

    results = []
    for i, lines in enumerate(parts):
        prompt = P.glossary_prompt(lang_name, lines, visual, user_gl)
        key = sha(P.GLOSSARY_SYSTEM, prompt, llm.model)
        f = cache_dir / f"glossary_{i}_{key}.json"
        if f.exists():
            results.append(read_json(f))
            continue
        log(f"  Vòng 1 — lập từ điển tên riêng/thuật ngữ{f' (phần {i + 1}/{len(parts)})' if len(parts) > 1 else ''}…")
        res = llm.ask(prompt, system=P.GLOSSARY_SYSTEM, schema=P.GLOSSARY_SCHEMA, effort="high", label="từ điển")
        write_json(f, res)
        results.append(res)
    gl = _merge_glossaries(results)
    # Từ điển người dùng luôn đè lên kết quả tự động.
    for u in user_gl:
        hit = next((e for e in gl["entries"] if e["canonical"].lower() == u["canonical"].lower()), None)
        if hit:
            hit["canonical"] = u["canonical"]
            hit["variants"] = list(dict.fromkeys(hit.get("variants", []) + u["variants"]))
            hit["confidence"] = "high"
        else:
            gl["entries"].append(
                {"canonical": u["canonical"], "variants": u["variants"], "type": "other", "confidence": "high",
                 "note": "từ điển người dùng"}
            )
    return gl


# ----------------------------------------------------------------------------------------------
# Vòng 2: hiệu đính từng cụm
# ----------------------------------------------------------------------------------------------


def correct_chunks(segs, marked_by_id, context, llm, cache_dir, system, workers, max_chars) -> tuple[dict, list, list]:
    chunks = _chunks(segs, max_chars)
    by_index = {s["id"]: i for i, s in enumerate(segs)}
    log(f"  Vòng 2 — hiệu đính {len(segs)} đoạn, chia {len(chunks)} cụm, {workers} luồng song song…")

    def job(ci: int):
        chunk = chunks[ci]
        i0, i1 = by_index[chunk[0]["id"]], by_index[chunk[-1]["id"]]
        before = [f"[{s['id']}] {s['text']}" for s in segs[max(0, i0 - 6) : i0]]
        after = [f"[{s['id']}] {s['text']}" for s in segs[i1 + 1 : i1 + 5]]
        items = [{"id": s["id"], "text": marked_by_id[s["id"]]} for s in chunk]
        prompt = P.correct_prompt(context, before, items, after)
        key = sha(system, prompt, llm.model)
        f = cache_dir / f"chunk_{ci:03d}_{key}.json"
        if f.exists():
            return ci, read_json(f)
        want = [s["id"] for s in chunk]
        res = None
        for attempt in range(2):
            p = prompt
            if attempt:
                p += (f"\n\nLƯU Ý: lần trước bạn trả thiếu/thừa đoạn. Bắt buộc trả về đúng {len(want)} đoạn với các id: "
                      f"{want}.")
            try:
                res = llm.ask(p, system=system, schema=P.CORRECT_SCHEMA, effort="high", label=f"cụm {ci + 1}")
            except LLMError as e:
                log(f"  ! cụm {ci + 1} lỗi hẳn, giữ nguyên bản gốc: {e}")
                return ci, {"segments": [], "uncertain": []}  # không cache để lần chạy sau thử lại
            got = [s.get("id") for s in res.get("segments", [])]
            if sorted(set(got)) == want:
                break
        write_json(f, res)
        return ci, res

    out: dict[int, str] = {}
    uncertain: list[dict] = []
    fallback: list[int] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ci, res in ex.map(job, range(len(chunks))):
            done += 1
            ids = {s["id"] for s in chunks[ci]}
            for s in res.get("segments", []):
                if s.get("id") in ids and s["id"] not in out:
                    out[s["id"]] = _strip_marks(s.get("text", ""))
            for s in chunks[ci]:
                if s["id"] not in out:
                    out[s["id"]] = s["text"]
                    fallback.append(s["id"])
            for u in res.get("uncertain", []):
                if u.get("id") in ids:
                    u["source"] = "vòng 2"
                    uncertain.append(u)
            if done % 5 == 0 or done == len(chunks):
                log(f"    {done}/{len(chunks)} cụm xong")
    return out, uncertain, fallback


# ----------------------------------------------------------------------------------------------
# Vòng 3: soát toàn cục
# ----------------------------------------------------------------------------------------------


def review(segs, fixed: dict, context, llm, cache_dir, lang_name, workers, window_chars=24_000):
    system = P.REVIEW_SYSTEM.format(lang_name=lang_name)
    rows_all = []
    for s in segs:
        new = fixed[s["id"]]
        row = f"[{s['id']}] {new if new else '(đã xóa)'}"
        if fold(new) != fold(s["text"]) or new == "":
            row += f"\n      ASR: {s['text']}"
        rows_all.append((s["id"], row))
    windows, cur, n = [], [], 0
    for sid, row in rows_all:
        if cur and n + len(row) > window_chars:
            windows.append(cur)
            cur, n = [], 0
        cur.append((sid, row))
        n += len(row) + 1
    if cur:
        windows.append(cur)
    log(f"  Vòng 3 — soát toàn cục (nhất quán, lỗi sót, sửa quá tay) trên {len(windows)} cửa sổ…")

    def ask(rows, name: str, label: str) -> dict:
        prompt = P.review_prompt(context, [r for _, r in rows])
        key = sha(system, prompt, llm.model)
        f = cache_dir / f"review_{name}_{key}.json"
        if f.exists():
            return read_json(f)
        res = llm.ask(prompt, system=system, schema=P.REVIEW_SCHEMA, effort="high", label=label)
        write_json(f, res)
        return res

    def job(wi: int) -> list[tuple[list, dict]]:
        """Soát một cửa sổ. Nếu lỗi (thường do model trả lời quá dài) thì chia đôi rồi soát lại từng nửa.
        Một cửa sổ hỏng không làm mất kết quả của các cửa sổ khác."""
        rows = windows[wi]
        try:
            return [(rows, ask(rows, f"{wi:02d}", f"soát {wi + 1}"))]
        except LLMError as e:
            if len(rows) < 2:
                log(f"  ! Bỏ qua cửa sổ soát {wi + 1}: {e}")
                return []
            log(f"  ! Cửa sổ soát {wi + 1} lỗi ({e}), chia đôi rồi soát lại")
        half = len(rows) // 2
        out = []
        for part, tag in ((rows[:half], "a"), (rows[half:], "b")):
            try:
                out.append((part, ask(part, f"{wi:02d}{tag}", f"soát {wi + 1}{tag}")))
            except LLMError as e:
                log(f"  ! Bỏ qua nửa cửa sổ soát {wi + 1}{tag}: {e}")
        return out

    applied, skipped, uncertain = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = [x for part in ex.map(job, range(len(windows))) for x in part]
    for rows, res in results:
        ids = {sid for sid, _ in rows}
        for e in res.get("edits", []):
            sid, find, rep = e.get("id"), e.get("find", ""), e.get("replace", "")
            if sid not in ids or not find:
                skipped.append(e)
                continue
            cur_text = fixed[sid]
            if find in cur_text:
                new = cur_text.replace(find, rep, 1)
            else:
                m = re.search(re.escape(find), cur_text, re.I)
                if not m:
                    skipped.append(e)
                    continue
                new = cur_text[: m.start()] + rep + cur_text[m.end() :]
            fixed[sid] = clean_spaces(nfc(new))
            applied.append(e)
        for u in res.get("uncertain", []):
            if u.get("id") in ids:
                u["source"] = "vòng 3"
                uncertain.append(u)
    log(f"    áp dụng {len(applied)} chỉnh sửa, bỏ qua {len(skipped)} (không khớp chuỗi)")
    return applied, skipped, uncertain


# ----------------------------------------------------------------------------------------------
# Điều phối
# ----------------------------------------------------------------------------------------------


def normalize(
    raw: dict,
    visual: dict | None,
    user_gl: list[dict],
    llm: LLM,
    work: Path,
    lang_name: str = "tiếng Việt",
    verbatim: bool = False,
    workers: int = 4,
    chunk_chars: int = 2800,
    do_review: bool = True,
    glossary_llm: LLM | None = None,
) -> dict:
    """`glossary_llm`: model (mạnh) riêng cho bước lập từ điển; mặc định dùng `llm`."""
    glossary_llm = glossary_llm or llm
    cache_dir = work / "normalize"
    cache_dir.mkdir(parents=True, exist_ok=True)
    segs = raw["segments"]
    marked_by_id = {s["id"]: mark_low_confidence(s) for s in segs}
    n_low = sum(v.count(P.LOW_CONF_OPEN) for v in marked_by_id.values())
    log(f"Chuẩn hóa: {len(segs)} đoạn, {n_low} cụm từ ASR kém tự tin được đánh dấu để soi kỹ")

    gl = build_glossary([f"[{s['id']}] {marked_by_id[s['id']]}" for s in segs], visual, user_gl, glossary_llm, cache_dir,
                        lang_name)
    log(f"    từ điển: {len(gl['entries'])} mục; chủ đề: {gl.get('topic', '')[:160]}")
    context = P.glossary_block(gl) + P.user_glossary_block(user_gl) + P.visual_block(visual, 80)

    system = P.correct_system(lang_name, verbatim)
    fixed, uncertain, fallback = correct_chunks(segs, marked_by_id, context, llm, cache_dir, system, workers, chunk_chars)
    if fallback:
        log(f"  ! {len(fallback)} đoạn LLM không trả về — giữ nguyên bản gốc: {fallback[:20]}")

    applied, skipped = [], []
    if do_review:
        try:
            applied, skipped, unc3 = review(segs, fixed, context, llm, cache_dir, lang_name, workers)
            uncertain += unc3
        except LLMError as e:
            log(f"  ! Vòng soát toàn cục lỗi, bỏ qua: {e}")

    # Đoạn bị viết lại khác hẳn lời gốc -> có thể sửa quá tay, đưa vào danh sách cần kiểm tra.
    flagged = {u.get("id") for u in uncertain}
    for s in segs:
        a, b = fold(s["text"]), fold(fixed[s["id"]])
        if b and len(a) >= 15 and s["id"] not in flagged:
            r = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
            if r < 0.4:
                uncertain.append({"id": s["id"], "text": fixed[s["id"]][:80],
                                  "note": f"bản sửa khác nhiều so với ASR (giống {r:.0%}) — nên nghe lại", "source": "tự động"})

    out_segs = []
    for s in segs:
        out_segs.append({"id": s["id"], "start": s["start"], "end": s["end"], "raw": s["text"],
                         "text": fixed[s["id"]], "words": s.get("words", [])})
    return {
        "glossary": gl,
        "segments": out_segs,
        "uncertain": uncertain,
        "review_applied": applied,
        "review_skipped": skipped,
        "fallback_ids": fallback,
        "llm": {"backend": llm.name, "model": llm.model, "glossary_model": glossary_llm.model},
    }


# ----------------------------------------------------------------------------------------------
# Báo cáo
# ----------------------------------------------------------------------------------------------


def word_diff(a: str, b: str) -> str:
    """Hiển thị khác biệt cấp từ: ~~cũ~~ **mới**."""
    aw, bw = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, [fold(x) or x for x in aw], [fold(x) or x for x in bw], autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            # Có thể khác dấu câu/hoa thường/dấu thanh — vẫn hiển thị bản mới.
            for x, y in zip(aw[i1:i2], bw[j1:j2]):
                out.append(y if x == y else f"~~{x}~~ **{y}**")
        else:
            if i2 > i1:
                out.append("~~" + " ".join(aw[i1:i2]) + "~~")
            if j2 > j1:
                out.append("**" + " ".join(bw[j1:j2]) + "**")
    return " ".join(out)


def write_report(result: dict, path: Path, video_name: str, asr_model: str, asr_repairs: list | None = None) -> dict:
    segs = result["segments"]
    by_id = {s["id"]: s for s in segs}
    changed = [s for s in segs if s["text"] != s["raw"] and s["text"]]
    removed = [s for s in segs if not s["text"]]
    gl = result["glossary"]
    ll = result["llm"]
    gm = ll.get("glossary_model") or ll["model"]
    L = [
        f"# Báo cáo chuẩn hóa transcript — {video_name}",
        "",
        f"- ASR: `{asr_model}` · LLM: `{ll['backend']}` / `{ll['model']}`"
        + (f" (từ điển: `{gm}`)" if gm != ll["model"] else ""),
        f"- Tổng số đoạn: **{len(segs)}** · đã sửa: **{len(changed)}** · đã xóa (ảo giác/lặp): **{len(removed)}** · "
        f"cần kiểm tra lại: **{len(result['uncertain'])}**",
        f"- Vòng soát toàn cục áp dụng thêm **{len(result['review_applied'])}** chỉnh sửa.",
    ]
    if result["fallback_ids"]:
        L.append(f"- ⚠️ {len(result['fallback_ids'])} đoạn LLM không xử lý được, giữ nguyên bản gốc: {result['fallback_ids']}")
    L += ["", "## Bối cảnh nhận diện", "", gl.get("topic", ""), "", gl.get("setting", "")]
    if gl.get("style_notes"):
        L += ["", f"*Quy ước viết:* {gl['style_notes']}"]
    if gl.get("speakers"):
        L += ["", "*Người nói/tham gia:* " + ", ".join(gl["speakers"])]

    L += ["", "## Từ điển tên riêng & thuật ngữ đã áp dụng", "",
          "| Cách viết chuẩn | Loại | Các biến thể ASR nghe nhầm | Tin cậy | Ghi chú |", "|---|---|---|---|---|"]
    for e in gl.get("entries", []):
        v = ", ".join(e.get("variants") or []) or "—"
        L.append(f"| **{e['canonical']}** | {e.get('type', '')} | {v} | {e.get('confidence', '')} | "
                 f"{(e.get('note') or '').replace('|', '/')} |")

    L += ["", "## ⚠️ Cần kiểm tra lại (chưa chắc chắn)", ""]
    if result["uncertain"]:
        L += ["| Thời điểm | Cụm từ | Ghi chú | Câu (đã chuẩn hóa) |", "|---|---|---|---|"]
        for u in sorted(result["uncertain"], key=lambda x: x.get("id", 0)):
            s = by_id.get(u.get("id"))
            if not s:
                continue
            L.append(f"| {ts(s['start'])} | {u.get('text', '')} | {(u.get('note') or '').replace('|', '/')} | "
                     f"{s['text'].replace('|', '/')} |")
    else:
        L.append("Không có.")

    if asr_repairs:
        L += ["", "## Quãng ASR bị ảo giác đã nhận dạng lại", "",
              "Whisper chèn câu bịa (vd. \"Hãy subscribe cho kênh…\") hoặc lặp câu ở các quãng dưới đây; tool đã nhận dạng "
              "lại riêng từng quãng ở chế độ kỹ hơn.", "", "| Thời điểm | ASR lần đầu | Nhận dạng lại |", "|---|---|---|"]
        for r in asr_repairs:
            old_t = r["old"][:120].replace("|", "/") + ("…" if len(r["old"]) > 120 else "")
            L.append(f"| {ts(r['start'])}–{ts(r['end'])} | ~~{old_t}~~ | {r['new'].replace('|', '/') or '(không có lời nói — bỏ)'} |")

    if result["review_applied"]:
        L += ["", "## Chỉnh sửa ở vòng soát toàn cục", "", "| Thời điểm | Sửa | Lý do |", "|---|---|---|"]
        for e in result["review_applied"]:
            s = by_id[e["id"]]
            L.append(f"| {ts(s['start'])} | ~~{e['find']}~~ → **{e['replace'] or '(xóa)'}** | {e.get('reason', '')} |")

    if removed:
        L += ["", "## Đoạn đã xóa (ảo giác / lặp của ASR)", ""]
        for s in removed:
            L.append(f"- `{ts(s['start'])}` ~~{s['raw']}~~")

    L += ["", "## Chi tiết từng thay đổi", "", "Ký hiệu: ~~bản gốc ASR~~ **bản chuẩn hóa**", ""]
    for s in changed:
        L.append(f"- `{ts(s['start'])}` {word_diff(s['raw'], s['text'])}")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
    return {"changed": len(changed), "removed": len(removed), "uncertain": len(result["uncertain"])}
