"""Lớp gọi LLM. Ba backend:

- openrouter : gọi model bất kỳ qua OpenRouter (OPENROUTER_API_KEY) — rẻ nhất khi phối nhiều model.
- claude-cli : gọi `claude -p` (Claude Code headless) — dùng luôn tài khoản Claude Code đang đăng nhập.
- anthropic  : gọi Claude API qua SDK `anthropic` (ANTHROPIC_API_KEY hoặc profile `ant auth login`).

Cả ba đều hỗ trợ đầu ra JSON theo schema và đầu vào ảnh (để đọc chữ trên slide).

Mỗi bước có thể dùng một model riêng (vai trò):
- main   : sửa từng cụm, vòng soát, tóm tắt — nhiều token nhất nên dùng model rẻ.
- strong : lập từ điển tên riêng/thuật ngữ — 1 lượt gọi nhưng quyết định chất lượng chuẩn hóa.
- vision : đọc chữ trên khung hình.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .util import log


class LLMError(RuntimeError):
    pass


class LLM:
    name = "base"

    def __init__(self, model: str, timeout: int = 1200, retries: int = 2):
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.cost_usd = 0.0
        self.calls = 0
        self._lock = threading.Lock()

    def ask(
        self,
        prompt: str,
        *,
        system: str,
        schema: dict | None = None,
        images: list[Path] | None = None,
        effort: str = "high",
        label: str = "",
    ) -> Any:
        """Trả về dict (khi có schema) hoặc str. Tự thử lại khi lỗi tạm thời / JSON hỏng."""
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                out = self._ask(prompt, system=system, schema=schema, images=images, effort=effort)
                if schema is not None and not isinstance(out, dict):
                    out = parse_json(out)
                with self._lock:
                    self.calls += 1
                return out
            except Exception as e:  # noqa: BLE001 — mọi lỗi đều thử lại, hết lượt thì báo
                last = e
                if attempt < self.retries:
                    wait = 5 * (attempt + 1)
                    log(f"  ! LLM lỗi{' (' + label + ')' if label else ''}: {str(e)[:300]} — thử lại sau {wait}s")
                    time.sleep(wait)
        raise LLMError(f"LLM thất bại sau {self.retries + 1} lần{' (' + label + ')' if label else ''}: {last}")

    def _ask(self, prompt, *, system, schema, images, effort):  # pragma: no cover - abstract
        raise NotImplementedError

    def _add_cost(self, usd: float) -> None:
        with self._lock:
            self.cost_usd += usd or 0.0


class ClaudeCLI(LLM):
    name = "claude-cli"

    def __init__(self, model: str = "opus", **kw):
        super().__init__(model, **kw)
        self.exe = shutil.which("claude")
        if not self.exe:
            raise LLMError("Không tìm thấy lệnh `claude` (Claude Code) trong PATH.")
        # Chạy trong thư mục trống để Claude không đọc CLAUDE.md / cấu hình của project nào.
        self.sandbox = Path(tempfile.mkdtemp(prefix="vtranscript-llm-"))

    def _ask(self, prompt, *, system, schema, images, effort):
        cmd = [
            self.exe, "-p", "--model", self.model, "--output-format", "json",
            "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "",
            "--system-prompt", system,
        ]
        if effort:
            cmd += ["--effort", effort]
        if schema is not None:
            cmd += ["--json-schema", json.dumps(schema, ensure_ascii=False)]
        if images:
            dirs = sorted({str(Path(p).resolve().parent) for p in images})
            cmd += ["--tools", "Read", "--allowedTools", "Read"]
            for d in dirs:
                cmd += ["--add-dir", d]
            listing = "\n".join(f"- {Path(p).resolve()}" for p in images)
            prompt = (
                "Trước tiên dùng tool Read để mở và xem LẦN LƯỢT TẤT CẢ các ảnh sau (không bỏ ảnh nào):\n"
                f"{listing}\n\n{prompt}"
            )
        else:
            cmd += ["--tools", ""]
        env = dict(os.environ)
        env.pop("CLAUDECODE", None)  # cho phép gọi lồng khi tool được chạy từ trong một phiên Claude Code
        r = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout, cwd=self.sandbox, env=env
        )
        if r.returncode != 0 and not r.stdout.strip():
            raise LLMError(f"claude thoát mã {r.returncode}: {r.stderr.strip()[-800:]}")
        try:
            data = json.loads(r.stdout)
        except json.JSONDecodeError as e:
            raise LLMError(f"Không đọc được output của claude: {r.stdout[-500:]}") from e
        self._add_cost(float(data.get("total_cost_usd") or 0))
        if data.get("is_error"):
            raise LLMError(f"claude báo lỗi: {str(data.get('result'))[:800]}")
        if schema is not None:
            so = data.get("structured_output")
            if isinstance(so, dict):
                return so
            return parse_json(data.get("result") or "")
        return (data.get("result") or "").strip()


class AnthropicAPI(LLM):
    name = "anthropic"
    PRICES = {"opus": (4, 20), "sonnet": (2, 10), "haiku": (0.10, 0.50), "fable": (8, 40)}

    def __init__(self, model: str = "claude-opus-5-5", **kw):
        super().__init__(model, **kw)
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("Chưa cài SDK: pip install anthropic") from e
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=self.timeout, max_retries=3)

    def _ask(self, prompt, *, system, schema, images, effort):
        content: list[dict] = []
        for p in images or []:
            data = base64.standard_b64encode(Path(p).read_bytes()).decode("ascii")
            mt = "image/png" if str(p).lower().endswith(".png") else "image/jpeg"
            content.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}})
        content.append({"type": "text", "text": prompt})
        output_config: dict = {"effort": effort or "high"}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=64000,
            system=system,
            messages=[{"role": "user", "content": content}],
            thinking={"type": "adaptive"},
            output_config=output_config,
            # Nếu bộ lọc an toàn từ chối nhầm, API tự chạy lại trên model dự phòng phù hợp.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            raise LLMError(f"Model từ chối yêu cầu: {getattr(msg.stop_details, 'explanation', '')}")
        if msg.stop_reason == "max_tokens":
            raise LLMError("Output bị cắt do max_tokens")
        u = msg.usage
        # Ước tính chi phí theo bảng giá niêm yết ($/MTok vào, ra) — chỉ để tham khảo, chưa tính cache.
        pin, pout = next((v for k, v in self.PRICES.items() if k in self.model), self.PRICES["opus"])
        self._add_cost((u.input_tokens * pin + u.output_tokens * pout) / 1e6)
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        return parse_json(text) if schema is not None else text


def openrouter_key() -> str:
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise LLMError("Chưa đặt OPENROUTER_API_KEY.")
    return key if key.startswith("sk-or-") else "sk-or-v1-" + key


class OpenRouter(LLM):
    """Gọi model bất kỳ trên OpenRouter (API kiểu OpenAI chat completions). Key: OPENROUTER_API_KEY."""

    name = "openrouter"
    BASE = "https://openrouter.ai/api/v1"
    _models: dict | None = None  # thông tin model (giá, tham số hỗ trợ), tải một lần
    _models_lock = threading.Lock()

    def __init__(self, model: str = "deepseek/deepseek-v4.1-flash", **kw):
        super().__init__(model, **kw)
        self.key = openrouter_key()
        info = self._model_info(model)
        self.params = set(info.get("supported_parameters") or [])
        self.vision = "image" in (info.get("architecture") or {}).get("input_modalities", [])
        top = info.get("top_provider") or {}
        self.max_out = min(int(top.get("max_completion_tokens") or 32000), 32000)

    def _http(self, method: str, path: str, body: dict | None = None) -> dict:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            self.BASE + path, method=method,
            data=json.dumps(body, ensure_ascii=False).encode() if body is not None else None,
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                     "X-Title": "vtranscript"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise LLMError(f"OpenRouter HTTP {e.code}: {e.read().decode(errors='replace')[:600]}") from e

    def _model_info(self, model: str) -> dict:
        with OpenRouter._models_lock:
            if OpenRouter._models is None:
                OpenRouter._models = {m["id"]: m for m in self._http("GET", "/models")["data"]}
        if model not in OpenRouter._models:
            raise LLMError(f"OpenRouter không có model '{model}'. Xem danh sách: https://openrouter.ai/models")
        return OpenRouter._models[model]

    def _ask(self, prompt, *, system, schema, images, effort):
        if images and not self.vision:
            raise LLMError(f"Model {self.model} không nhận ảnh — chọn model khác cho bước đọc khung hình.")
        content: list[dict] = [{"type": "text", "text": prompt}]
        for p in images or []:
            data = base64.standard_b64encode(Path(p).read_bytes()).decode("ascii")
            mt = "image/png" if str(p).lower().endswith(".png") else "image/jpeg"
            content.append({"type": "image_url", "image_url": {"url": f"data:{mt};base64,{data}"}})
        sys_text = system
        body: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": sys_text}, {"role": "user", "content": content}],
            "max_tokens": self.max_out,
            "usage": {"include": True},
        }
        if schema is not None:
            if "structured_outputs" in self.params or "response_format" in self.params:
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": "result", "strict": True, "schema": schema}}
                body["provider"] = {"require_parameters": True}
            # Nhắc thêm trong lời nhắc để model yếu về structured output vẫn trả đúng dạng.
            body["messages"][0]["content"] = (
                f"{sys_text}\n\nChỉ trả về MỘT đối tượng JSON hợp lệ theo JSON Schema sau, không kèm chữ nào khác:\n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            )
        if effort and "reasoning" in self.params:
            body["reasoning"] = {"effort": {"max": "high", "xhigh": "high"}.get(effort, effort)}
        data = self._http("POST", "/chat/completions", body)
        if data.get("error"):
            raise LLMError(f"OpenRouter lỗi: {str(data['error'])[:600]}")
        u = data.get("usage") or {}
        self._add_cost(float(u.get("cost") or 0))
        with self._lock:
            self.tokens_in = getattr(self, "tokens_in", 0) + int(u.get("prompt_tokens") or 0)
            self.tokens_out = getattr(self, "tokens_out", 0) + int(u.get("completion_tokens") or 0)
        ch = (data.get("choices") or [{}])[0]
        if ch.get("finish_reason") == "length":
            raise LLMError("Output bị cắt do max_tokens")
        text = ((ch.get("message") or {}).get("content") or "").strip()
        if not text:
            raise LLMError(f"OpenRouter trả về rỗng (finish_reason={ch.get('finish_reason')})")
        return parse_json(text) if schema is not None else text


def parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        if i < 0 or j <= i:
            raise LLMError(f"Không tìm thấy JSON trong phản hồi: {text[:300]}")
        out = json.loads(text[i : j + 1])
    if not isinstance(out, dict):
        raise LLMError("Phản hồi JSON không phải object")
    return out


# Model mặc định theo backend và vai trò. OpenRouter: phối model theo kết quả thử trên video 68 phút
# (xem README). Từ điển do Opus lập (bước quyết định chất lượng); đọc khung hình dùng Sonnet (đọc đủ nhất
# sau Opus); phần sửa từng cụm do DeepSeek làm, với từ điển tốt vẫn sát bản Opus.
DEFAULT_MODELS = {
    "openrouter": {"main": "deepseek/deepseek-v4.1-flash", "strong": "anthropic/claude-opus-5.5",
                   "vision": "anthropic/claude-sonnet-5.5"},
    "claude-cli": {"main": "opus"},
    "anthropic": {"main": "claude-opus-5-5"},
}
BACKENDS = {"openrouter": OpenRouter, "claude-cli": ClaudeCLI, "anthropic": AnthropicAPI}


def resolve_backend(backend: str) -> str:
    if backend != "auto":
        if backend not in BACKENDS:
            raise LLMError(f"Backend không hỗ trợ: {backend}")
        return backend
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if shutil.which("claude"):
        return "claude-cli"
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    raise LLMError(
        "Không có backend LLM: đặt OPENROUTER_API_KEY, cài Claude Code (lệnh `claude`) hoặc đặt ANTHROPIC_API_KEY."
    )


def default_model(backend: str, role: str = "main") -> str:
    d = DEFAULT_MODELS[backend]
    return d.get(role) or d["main"]


def make_llm(backend: str, model: str | None) -> LLM:
    backend = resolve_backend(backend)
    return BACKENDS[backend](model or default_model(backend))


class LLMPool:
    """Tạo LLM theo vai trò khi cần (lazy); vai trò trùng model thì dùng chung một đối tượng."""

    def __init__(self, backend: str, models: dict[str, str | None]):
        self.backend = resolve_backend(backend)
        main = models.get("main") or default_model(self.backend, "main")
        # Backend chỉ có một model mặc định (claude-cli, anthropic): vai trò không chỉ định thì theo model chính.
        self.models = {r: models.get(r) or (default_model(self.backend, r) if r in DEFAULT_MODELS[self.backend]
                                            else main) for r in ("main", "strong", "vision")}
        self.models["main"] = main
        self._by_model: dict[str, LLM] = {}
        self._lock = threading.Lock()

    def get(self, role: str = "main") -> LLM:
        model = self.models[role]
        with self._lock:
            if model not in self._by_model:
                self._by_model[model] = BACKENDS[self.backend](model)
                log(f"LLM ({role}): {self.backend} / {model}")
            return self._by_model[model]

    @property
    def used(self) -> list[LLM]:
        return list(self._by_model.values())
