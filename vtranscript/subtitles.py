"""Tạo phụ đề từ transcript đã chuẩn hóa.

Văn bản chuẩn hóa khác bản ASR (sửa từ, gộp "kê pi ai" -> "KPI", xóa từ đệm...), nên không dùng lại mốc thời gian
của đoạn ASR một cách thô. Thay vào đó căn từng từ của bản chuẩn hóa với từng từ ASR (có mốc thời gian riêng) bằng
so khớp chuỗi trên dạng đã bỏ dấu; phần không khớp thì nội suy theo độ dài ký tự. Sau đó chia câu phụ đề theo chuẩn
thường dùng: tối đa 2 dòng × ~42 ký tự, tối đa ~6 giây/câu, ưu tiên ngắt ở dấu câu.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from .util import fold, ts

MAX_LINE = 42
MAX_LINES = 2
MAX_DUR = 6.0
MIN_DUR = 1.0
GAP_BREAK = 1.5
SENT_END = re.compile(r"[.!?…]['\")\]]*$")
CLAUSE_END = re.compile(r"[,;:–—]['\")\]]*$")
# Từ thường mở đầu một cụm/mệnh đề: ngắt dòng ngay TRƯỚC chúng ít khi cắt đôi từ ghép (tiếng Việt viết tách âm tiết
# nên không biết ranh giới từ; ngắt bừa dễ ra "đàm / phán", "tiềm / năng").
BREAK_BEFORE = frozenset(
    "là và thì mà nhưng nên vì để của cho với khi nếu rồi hoặc hay các những về trong trên từ đến tới theo như bởi do "
    "sau trước tại tức kiểu giống còn nhưng and but or so because that which to for with when if".split()
)


def _break_penalty(prev_word: str, next_word: str) -> float:
    """Độ "xấu" khi ngắt giữa hai từ: 0 sau dấu kết câu, thấp sau dấu phẩy, vừa trước từ nối, cao nếu không có gì."""
    if SENT_END.search(prev_word):
        return 0.0
    if CLAUSE_END.search(prev_word):
        return 0.8
    if next_word.lower().strip("\"'(“") in BREAK_BEFORE:
        return 2.5
    return 5.0


# ----------------------------------------------------------------------------------------------
# Căn thời gian cấp từ
# ----------------------------------------------------------------------------------------------


def _spread(tokens: list[str], start: float, end: float) -> list[tuple[str, float, float]]:
    """Chia khoảng [start, end] cho các token theo độ dài ký tự."""
    if not tokens:
        return []
    end = max(end, start)
    weights = [max(1, len(t)) for t in tokens]
    total = sum(weights)
    out, cur = [], start
    for t, w in zip(tokens, weights):
        nxt = cur + (end - start) * w / total
        out.append((t, cur, nxt))
        cur = nxt
    return out


def align_segment(text: str, seg: dict) -> list[tuple[str, float, float]]:
    """Trả về [(token, start, end)] cho văn bản `text` dựa trên word timestamps của đoạn ASR `seg`."""
    toks = text.split()
    if not toks:
        return []
    words = [w for w in (seg.get("words") or []) if w["w"].strip()]
    if not words:
        return _spread(toks, seg["start"], seg["end"])
    a = [fold(w["w"]) or w["w"].strip().lower() for w in words]
    b = [fold(t) or t.lower() for t in toks]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    timed: list[tuple[str, float, float] | None] = [None] * len(toks)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                w = words[i1 + k]
                timed[j1 + k] = (toks[j1 + k], w["s"], w["e"])
        elif op == "replace":
            for x in _spread(toks[j1:j2], words[i1]["s"], words[i2 - 1]["e"]):
                timed[j1] = x
                j1 += 1
        # "insert" (từ mới, không có trong ASR) nội suy ở dưới; "delete" (từ bị bỏ) không cần gì.
    # Nội suy cho các token chèn thêm: nằm giữa token có thời gian ở hai bên.
    i = 0
    while i < len(toks):
        if timed[i] is not None:
            i += 1
            continue
        j = i
        while j < len(toks) and timed[j] is None:
            j += 1
        left = timed[i - 1][2] if i > 0 else seg["start"]
        right = timed[j][1] if j < len(toks) else seg["end"]
        # Không có khoảng trống thì cho chiếm tạm một chút thời gian (bước "đơn điệu" bên dưới sẽ đẩy từ sau lùi lại).
        right = max(right, left + 0.08 * (j - i))
        for k, x in enumerate(_spread(toks[i:j], left, right)):
            timed[i + k] = x
        i = j
    # Đảm bảo đơn điệu tăng và nằm trong khoảng hợp lý.
    out, last = [], seg["start"]
    for t, s, e in timed:  # type: ignore[misc]
        s = max(s, last)
        e = max(e, s + 0.02)
        out.append((t, s, e))
        last = s
    return out


def timed_tokens(segments: list[dict], raw_by_id: dict[int, dict], key: str = "text") -> list[tuple[str, float, float]]:
    out = []
    for s in segments:
        txt = s.get(key) or ""
        if not txt.strip():
            continue
        out.extend(align_segment(txt, raw_by_id.get(s["id"], s)))
    # Đảm bảo đơn điệu giữa các đoạn
    fixed, last = [], 0.0
    for t, a, b in out:
        a = max(a, last)
        b = max(b, a + 0.02)
        fixed.append((t, a, b))
        last = a
    return fixed


# ----------------------------------------------------------------------------------------------
# Chia câu phụ đề
# ----------------------------------------------------------------------------------------------


def _text_len(tokens) -> int:
    return sum(len(t[0]) for t in tokens) + max(0, len(tokens) - 1)


def _span(tokens) -> float:
    return tokens[-1][2] - tokens[0][1]


def _sentences(tokens: list, max_chars: int, max_dur: float) -> list[list]:
    """Tách thành từng câu (sau dấu kết câu hoặc ở khoảng lặng dài), rồi gộp câu rất ngắn ("Ok.", "Đúng không?")
    với câu kề bên nếu cả hai vừa một câu phụ đề."""
    units: list[list] = []
    cur: list = []
    for tok in tokens:
        if cur and (tok[1] - cur[-1][2] > GAP_BREAK or SENT_END.search(cur[-1][0])):
            units.append(cur)
            cur = []
        cur.append(tok)
    if cur:
        units.append(cur)
    merged: list[list] = []
    for u in units:
        if merged:
            prev = merged[-1]
            gap = u[0][1] - prev[-1][2]
            comb = prev + u
            fits = _text_len(comb) <= max_chars and _span(comb) <= max_dur
            if fits and ((_text_len(prev) < 30 and gap <= 1.0) or (_text_len(u) < 15 and gap <= 0.5)):
                merged[-1] = comb
                continue
        merged.append(u)
    return merged


def _split_sentence(u: list, max_chars: int, max_dur: float) -> list[list]:
    """Chia một câu dài thành ít câu phụ đề nhất, các phần dài gần đều nhau, ngắt ở dấu phẩy/khoảng lặng,
    tránh để sót một mẩu cụt ngủn (vd. chỉ còn "BingX.") — quy hoạch động trên các điểm ngắt."""
    n = len(u)
    pre = [0]
    for t in u:
        pre.append(pre[-1] + len(t[0]) + 1)

    def length(i: int, j: int) -> int:
        return pre[j] - pre[i] - 1

    def fits(i: int, j: int) -> bool:
        return j - i == 1 or (length(i, j) <= max_chars and u[j - 1][2] - u[i][1] <= max_dur)

    if fits(0, n):
        return [u]
    k = max(2, -(-length(0, n) // max_chars), int(-(-_span(u) // max_dur)))
    ideal = length(0, n) / k

    def break_cost(j: int) -> float:  # ngắt giữa token j-1 và j
        gap = u[j][1] - u[j - 1][2]
        return _break_penalty(u[j - 1][0], u[j][0]) - min(1.5, 2 * gap)

    inf = float("inf")
    dp, back = [0.0] + [inf] * n, [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            if not fits(i, j):
                break
            L = length(i, j)
            c = dp[i] + 2.0 + 2.0 * ((L - ideal) / ideal) ** 2 + (3.0 if L < 20 else 0.0)
            if j < n:
                c += break_cost(j)
            if c < dp[j]:
                dp[j], back[j] = c, i
    pieces, j = [], n
    while j > 0:
        pieces.append(u[back[j] : j])
        j = back[j]
    return pieces[::-1]


def build_cues(tokens: list[tuple[str, float, float]], max_line: int = MAX_LINE, max_lines: int = MAX_LINES,
               max_dur: float = MAX_DUR) -> list[dict]:
    max_chars = max_line * max_lines
    cues: list[list] = []
    for sent in _sentences(tokens, max_chars, max_dur):
        cues.extend(_split_sentence(sent, max_chars, max_dur))

    out = []
    for c in cues:
        out.append({"start": c[0][1], "end": c[-1][2], "lines": split_lines([t[0] for t in c], max_line)})
    # Thời lượng tối thiểu + kéo dài nhẹ cho dễ đọc, không chồng lấn câu sau.
    for i, cue in enumerate(out):
        nxt = out[i + 1]["start"] if i + 1 < len(out) else cue["end"] + 3
        chars = sum(len(x) for x in cue["lines"])
        # Đủ thời gian đọc (~17 ký tự/giây), tối thiểu 1 giây — nhưng không đè lên câu sau.
        want = max(cue["end"] + 0.25, cue["start"] + max(MIN_DUR, chars / 17))
        cue["end"] = max(cue["end"], min(want, nxt - 0.04))
    return out


def split_lines(words: list[str], max_line: int = MAX_LINE) -> list[str]:
    text = " ".join(words)
    if len(text) <= max_line or len(words) < 2:
        return [text]
    # Chọn điểm ngắt sao cho hai dòng cân nhau, ưu tiên sau dấu câu; dòng trên có thể dài hơn đôi chút.
    best, best_score = 1, float("inf")
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        score = abs(len(a) - len(b)) + 1.6 * _break_penalty(words[i - 1], words[i])
        if max(len(a), len(b)) > max_line:
            score += 40 * (max(len(a), len(b)) - max_line)
        if score < best_score:
            best, best_score = i, score
    return [" ".join(words[:best]), " ".join(words[best:])]


# ----------------------------------------------------------------------------------------------
# Ghi file
# ----------------------------------------------------------------------------------------------


def write_srt(cues: list[dict], path: Path) -> None:
    blocks = []
    for i, c in enumerate(cues, 1):
        blocks.append(f"{i}\n{ts(c['start'], ms=True)} --> {ts(c['end'], ms=True)}\n" + "\n".join(c["lines"]))
    Path(path).write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def _ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(s: str) -> str:
    return s.replace("\\", "/").replace("{", "(").replace("}", ")")


def write_ass(cues: list[dict], path: Path, width: int, height: int, font: str = "Arial",
              font_size: int | None = None) -> None:
    fs = font_size or max(16, round(height * 0.043))
    outline = max(2, round(fs * 0.22))  # với BorderStyle=3 đây là độ rộng lề của hộp nền
    margin_v = round(height * 0.045)
    margin_lr = round(width * 0.06)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{fs},&H00FFFFFF,&H000000FF,&H60000000,&H00000000,0,0,0,0,100,100,0,0,3,{outline},0,2,{margin_lr},{margin_lr},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [
        f"Dialogue: 0,{_ass_time(c['start'])},{_ass_time(c['end'])},Default,,0,0,0,,"
        + r"\N".join(_ass_escape(x) for x in c["lines"])
        for c in cues
    ]
    Path(path).write_text(header + "\n".join(lines) + "\n", encoding="utf-8-sig")
