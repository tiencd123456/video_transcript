"""Tiện ích dùng chung: log, định dạng thời gian, đọc/ghi JSON, chuẩn hóa chuỗi để so khớp."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[{int(elapsed // 60):02d}:{int(elapsed % 60):02d}] {msg}", file=sys.stderr, flush=True)


def ts(seconds: float, ms: bool = False) -> str:
    """1234.5 -> '00:20:34' (hoặc '00:20:34,500' khi ms=True, dạng SRT)."""
    seconds = max(0.0, seconds)
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, millis = divmod(rem, 1000)
    base = f"{h:02d}:{m:02d}:{s:02d}"
    return f"{base},{millis:03d}" if ms else base


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(json.dumps(p, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


_PUNCT_RE = re.compile(r"[^\w]+", re.UNICODE)


def fold(text: str) -> str:
    """Hạ chữ thường, bỏ dấu tiếng Việt và dấu câu — dùng để so khớp mờ giữa bản gốc và bản sửa."""
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return _PUNCT_RE.sub("", text)


def clean_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?…%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)
