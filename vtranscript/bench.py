"""Công cụ thử model: xem số dư key OpenRouter và so chất lượng các lần chạy với một bản tham chiếu.

  vtranscript-bench balance
  vtranscript-bench compare output/<bản tham chiếu> output/_thu/<lần thử> [...]
(chưa cài gói thì thay `vtranscript-bench` bằng `python -m vtranscript.bench`)

"Lệch" = tỉ lệ từ khác bản chuẩn hóa của lần chạy tham chiếu (thường là cấu hình best), sau khi bỏ dấu câu,
hoa/thường, tiếng ngập ngừng và số (vì "hai trăm" và "200" là như nhau). Bản tham chiếu không phải sự thật
tuyệt đối: chính nó còn khoảng 2% lỗi, nên chỉ so được lần nào tốt hơn hay kém hơn, cần nghe lại các phút lệch nhiều.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

from . import __version__, config
from .util import read_json, ts

FILLERS = set("à ờ ừ ừm ơ ờm hừm hmm uh um ạ ừa ồ ô ê ủa á a ơi dạ vâng".split())
NUM_WORDS = set("không một mốt hai ba bốn tư năm lăm nhăm sáu bảy tám chín mười mươi trăm nghìn ngàn triệu tỷ tỉ "
                "linh lẻ rưỡi".split())
HALLUCINATION = re.compile(r"subscribe|Ghiền Mì Gõ|theo dõi và hẹn gặp|video tiếp theo|bỏ lỡ những video|"
                           r"Cảm ơn các bạn đã|đăng ký kênh", re.I)


def words(text: str) -> list[str]:
    s = unicodedata.normalize("NFC", text.lower())
    s = re.sub(r"(.)\1{2,}", r"\1", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return [t for t in s.split() if t not in FILLERS and t not in NUM_WORDS and not re.search(r"\d", t)]


def diff_count(ref: list[str], hyp: list[str]) -> int:
    n = 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
        if op != "equal":
            n += max(i2 - i1, j2 - j1)
    return n


def pct(ref: str, hyp: str) -> float:
    r = words(ref)
    return 100 * diff_count(r, words(hyp)) / max(len(r), 1)


class Run:
    def __init__(self, d: Path):
        self.dir = d
        w = d / "_work"
        self.norm = read_json(w / "normalized.json")
        self.segs = self.norm["segments"]
        asr = read_json(w / "asr_raw.json") if (w / "asr_raw.json").exists() else {}
        vis = read_json(w / "visual_context.json") if (w / "visual_context.json").exists() else {}
        llm = self.norm.get("llm") or {}
        self.models = {"asr": asr.get("model"), "vision": vis.get("model"),
                       "glossary": llm.get("glossary_model"), "main": llm.get("model")}
        self.cost = None  # None = lần chạy chưa có sổ chi phí (chạy trước khi có tính năng này)
        if (w / "chi_phi.jsonl").exists():
            self.cost = 0.0
            for line in (w / "chi_phi.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.cost += r["asr"]["usd"] + sum(x["usd"] for x in r["llm"])
        s = d / "03_tom_tat.md"
        self.summary = s.read_text(encoding="utf-8") if s.exists() else ""

    def text(self, a: float = 0, b: float = 1e9, key: str = "text") -> str:
        return " ".join(s[key] for s in self.segs if a <= s["start"] < b)


def compare(ref_dir: Path, run_dirs: list[Path]) -> str:
    ref = Run(ref_dir)
    runs = [ref] + [Run(d) for d in run_dirs]
    dur = max(s["end"] for s in ref.segs)
    ref_txt = ref.text()
    # Thuật ngữ để đếm: các mục trong từ điển của bản tham chiếu, xuất hiện ít nhất 3 lần ở bản tham chiếu.
    terms = []
    for e in ref.norm["glossary"].get("entries", []):
        t = e["canonical"]
        c = len(re.findall(r"(?<!\w)" + re.escape(t) + r"(?!\w)", ref_txt, re.I))
        if c >= 3 and t not in [x for x, _ in terms]:
            terms.append((t, c))

    out = [f"# So sánh với bản tham chiếu: {ref_dir}", ""]
    out += ["| | " + " | ".join(r.dir.name for r in runs) + " |", "|---" * (len(runs) + 1) + "|"]

    def row(name, vals):
        out.append(f"| {name} | " + " | ".join(str(v) for v in vals) + " |")

    for k, label in (("asr", "ASR"), ("vision", "Đọc khung hình"), ("glossary", "Lập từ điển"),
                     ("main", "Sửa cụm/soát/tóm tắt")):
        row(label, [r.models[k] or "?" for r in runs])
    row("**Lệch sau chuẩn hóa**", ["(tham chiếu)"] + [f"**{pct(ref_txt, r.text()):.1f}%**" for r in runs[1:]])
    row("Lệch bản thô (ASR)", [f"{pct(ref_txt, r.text(key='raw')):.1f}%" for r in runs])
    lost_rows = []
    for r in runs:
        lost, where = 0, []
        for a in range(0, int(dur), 30):
            n_ref, n_run = len(words(ref.text(a, a + 30))), len(words(r.text(a, a + 30)))
            if n_ref >= 15 and n_run < 0.4 * n_ref:
                lost += n_ref - n_run
                where.append(ts(a)[3:])
        lost_rows.append((lost, where))
    total = len(words(ref_txt))
    row("Nội dung bị mất", [f"{100 * x / total:.1f}%" + (f" ({', '.join(w[:4])})" if w else "") for x, w in lost_rows])
    row("Mục từ điển", [len(r.norm["glossary"].get("entries", [])) for r in runs])
    row("Chỗ cần kiểm tra lại", [len(r.norm.get("uncertain", [])) for r in runs])
    row("Đoạn không sửa được (LLM lỗi)", [len(r.norm.get("fallback_ids", [])) for r in runs])
    row("Câu ảo giác còn sót", [sum(1 for s in r.segs if HALLUCINATION.search(s["text"])) for r in runs])
    row("Tóm tắt (số mục / ký tự)", [f"{len(re.findall(r'^#{2,3} ', r.summary, re.M))} / {len(r.summary):,}"
                                     for r in runs])
    row("Chi phí đã tiêu (không tính cache)", [f"${r.cost:.2f}" if r.cost is not None else "?" for r in runs])

    if terms:
        out += ["", "## Số lần xuất hiện tên riêng, thuật ngữ", "",
                "| | " + " | ".join(r.dir.name for r in runs) + " |", "|---" * (len(runs) + 1) + "|"]
        for t, c in sorted(terms, key=lambda x: -x[1])[:25]:
            vals = [len(re.findall(r"(?<!\w)" + re.escape(t) + r"(?!\w)", r.text(), re.I)) for r in runs]
            mark = " ⚠" if any(v < 0.6 * c for v in vals[1:]) else ""
            row(t + mark, vals)

    for r in runs[1:]:
        mins = []
        for a in range(0, int(dur), 60):
            rt = ref.text(a, a + 60)
            if len(words(rt)) >= 40:
                mins.append((pct(rt, r.text(a, a + 60)), a))
        mins.sort(reverse=True)
        if not mins or mins[0][0] == 0:
            continue
        out += ["", f"## {r.dir.name}: các phút lệch nhiều nhất (nên nghe lại)", ""]
        for p, a in mins[:3]:
            out += [f"- **{ts(a)}** lệch {p:.0f}%",
                    f"    - tham chiếu: {ref.text(a, a + 60)[:300]}",
                    f"    - lần thử: {r.text(a, a + 60)[:300]}"]
    return "\n".join(out) + "\n"


def balance() -> str:
    from .llm import openrouter_key

    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": f"Bearer {openrouter_key()}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        d = json.load(resp)["data"]
    lim, rem = d.get("limit"), d.get("limit_remaining")
    return (f"Key OpenRouter: đã dùng ${d.get('usage', 0):.2f}"
            + (f", giới hạn ${lim:.2f}, còn ${rem:.2f}" if lim is not None else ", không đặt giới hạn"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vtranscript-bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("balance", help="Số dư key OpenRouter")
    c = sub.add_parser("compare", help="So các lần chạy với bản tham chiếu")
    c.add_argument("ref", type=Path, help="Thư mục output của lần chạy tham chiếu")
    c.add_argument("runs", type=Path, nargs="+", help="Thư mục output của các lần thử")
    c.add_argument("--out", type=Path, default=None, help="Ghi bảng so sánh ra file .md")
    a = p.parse_args(argv)
    try:
        config.load()
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    if a.cmd == "balance":
        print(balance())
        return 0
    md = compare(a.ref, a.runs)
    print(md)
    if a.out:
        a.out.write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
