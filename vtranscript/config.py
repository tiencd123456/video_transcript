"""Cấu hình từ file `.env` và hồ sơ model (`<tên>.env`).

Thứ tự ưu tiên, cao xuống thấp:
  1. tham số dòng lệnh (`--llm-model ...`);
  2. biến môi trường đã đặt sẵn trong shell (`export VT_LLM_MODEL=...`);
  3. hồ sơ chọn bằng `--profile <tên>`;
  4. file `.env`;
  5. mặc định trong code.

File `.env` giữ API key và model mặc định của máy này. Tool dùng file đầu tiên tìm thấy trong danh sách:
  - đường dẫn trong biến `VT_ENV_FILE`;
  - `.env` ở thư mục hiện tại;
  - `.env` ở thư mục mã nguồn (khi chạy từ bản checkout);
  - `~/.config/vtranscript/.env` (khi đã cài thành lệnh).

Hồ sơ `--profile <tên>` được tìm theo thứ tự: `./profiles/`, `~/.config/vtranscript/profiles/`, rồi các hồ sơ có
sẵn trong gói (`vtranscript/profiles/`: best, budget, cheap). Hồ sơ tự tạo trùng tên sẽ đè hồ sơ có sẵn.
"""

from __future__ import annotations

import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
BUILTIN_PROFILES = PKG_DIR / "profiles"
# Thư mục mã nguồn: chỉ có nghĩa khi chạy từ bản checkout (cài -e), không phải khi cài từ wheel.
SOURCE_ROOT = PKG_DIR.parent if (PKG_DIR.parent / "pyproject.toml").is_file() else None


def user_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "vtranscript"


# Biến trong .env/hồ sơ -> tên tham số dòng lệnh (thuộc tính argparse).
ARG_VARS = {
    "VT_ASR_BACKEND": "asr_backend",
    "VT_ASR_MODEL": "asr_model",
    "VT_LLM_BACKEND": "llm_backend",
    "VT_LLM_MODEL": "llm_model",
    "VT_LLM_STRONG_MODEL": "llm_strong_model",
    "VT_LLM_VISION_MODEL": "llm_vision_model",
}
CHOICES = {
    "asr_backend": ("auto", "api", "local"),
    "llm_backend": ("auto", "openrouter", "claude-cli", "anthropic"),
}


def parse_env_file(path: Path) -> dict[str, str]:
    """Đọc file dạng KEY=giá trị. Bỏ dòng trống, dòng `#`, tiền tố `export`, dấu nháy và chú thích cuối dòng."""
    out: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, val = line.partition("=")
        if not sep:
            continue
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        else:
            val = val.split(" #", 1)[0].rstrip()
        out[key] = val
    return out


def profile_dirs() -> list[Path]:
    dirs = [Path.cwd() / "profiles", user_dir() / "profiles", BUILTIN_PROFILES]
    out: list[Path] = []
    for d in dirs:
        if d.is_dir() and d.resolve() not in [x.resolve() for x in out]:
            out.append(d)
    return out


def list_profiles() -> list[str]:
    return sorted({p.stem for d in profile_dirs() for p in d.glob("*.env")})


def profile_path(name: str) -> Path:
    p = Path(name).expanduser()
    if p.suffix == ".env" and p.is_file():
        return p
    for d in profile_dirs():
        if (d / f"{name}.env").is_file():
            return d / f"{name}.env"
    raise ValueError(f"Không có hồ sơ '{name}'. Hồ sơ có sẵn: {', '.join(list_profiles()) or '(chưa có)'}")


def env_file() -> Path | None:
    if os.environ.get("VT_ENV_FILE"):
        f = Path(os.environ["VT_ENV_FILE"]).expanduser()
        if not f.is_file():
            raise ValueError(f"VT_ENV_FILE={f} không tồn tại")
        return f
    for d in (Path.cwd(), SOURCE_ROOT, user_dir()):
        if d and (d / ".env").is_file():
            return d / ".env"
    return None


def describe(path: Path) -> str:
    """Đường dẫn ngắn để in log: tên hồ sơ có sẵn, đường dẫn tương đối hoặc ~/…"""
    path = path.resolve()
    if path.parent == BUILTIN_PROFILES:
        return f"hồ sơ có sẵn '{path.stem}'"
    for base, prefix in ((Path.cwd(), ""), (Path.home(), "~/")):
        if path.is_relative_to(base):
            return prefix + str(path.relative_to(base))
    return str(path)


def load(profile: str | None = None) -> list[Path]:
    """Nạp .env và hồ sơ vào os.environ (không đè biến đã đặt sẵn). Trả về các file đã nạp."""
    files, values = [], {}
    if f := env_file():
        values.update(parse_env_file(f))
        files.append(f)
    if profile:
        f = profile_path(profile)
        values.update(parse_env_file(f))  # hồ sơ đè .env
        files.append(f)
    for k, v in values.items():
        if v and k not in os.environ:
            os.environ[k] = v
    return files


def apply_to_args(a) -> None:
    """Điền các tham số dòng lệnh chưa đặt (None hoặc 'auto') từ biến môi trường VT_*."""
    for var, attr in ARG_VARS.items():
        val = os.environ.get(var, "").strip()
        if not val or getattr(a, attr) not in (None, "auto"):
            continue
        if attr in CHOICES and val not in CHOICES[attr]:
            raise ValueError(f"{var}={val} không hợp lệ, chọn một trong: {', '.join(CHOICES[attr])}")
        setattr(a, attr, val)
