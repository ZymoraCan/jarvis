# config/__init__.py
import json
import platform
import sys
from pathlib import Path

DEFAULT_LOCAL_MODEL = "Orion-zhen/Qwen2.5-14B-Instruct-Uncensored"
DEFAULT_LOCAL_CHAT_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"

_CONFIG_PATH = Path(__file__).parent / "api_keys.json"


def _detect_os() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "mac"
    if system == "linux":
        return "linux"
    return "windows"


DEFAULT_CONFIG = {
    "os_system": _detect_os(),
    "model_type": "gemini",
    "local_endpoint": DEFAULT_LOCAL_CHAT_ENDPOINT,
    "local_model": DEFAULT_LOCAL_MODEL,
    "model_path": "",
    "gemini_api_key": "",
    "python_path": sys.executable,
    "microphone_device": "",
    "web_agent_enabled": True,
    "safe_mode": True,
    "default_browser": "brave",
}


def _write_config(data: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(
        json.dumps(data, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


def get_config_path() -> Path:
    return _CONFIG_PATH


def get_config() -> dict:
    """Load config/api_keys.json, filling missing fields with safe defaults."""
    data = {}
    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}

    cfg = {**DEFAULT_CONFIG, **data}
    changed = cfg != data
    if changed:
        _write_config(cfg)
    return cfg


def save_config(data: dict) -> dict:
    cfg = {**DEFAULT_CONFIG, **(data or {})}
    _write_config(cfg)
    return cfg


def update_config(updates: dict) -> dict:
    cfg = get_config()
    cfg.update(updates or {})
    return save_config(cfg)


def get_os() -> str:
    """Returns: 'windows' | 'mac' | 'linux'"""
    return get_config().get("os_system", "windows").lower()


def is_windows() -> bool: return get_os() == "windows"
def is_mac()     -> bool: return get_os() == "mac"
def is_linux()   -> bool: return get_os() == "linux"


def get_model_type() -> str:
    """Returns: 'local' or 'gemini'. Unknown values fall back to gemini."""
    value = str(get_config().get("model_type", "gemini")).lower().strip()
    return value if value in {"local", "gemini"} else "gemini"


def get_gemini_key() -> str:
    """Returns Gemini API key or an empty string."""
    return str(get_config().get("gemini_api_key", "") or "").strip()


def get_local_model() -> str:
    """Returns local model id/name."""
    cfg = get_config()
    return str(cfg.get("local_model") or DEFAULT_LOCAL_MODEL).strip()


def get_model_path() -> str:
    """Optional local filesystem path for the Transformers model."""
    return str(get_config().get("model_path", "") or "").strip()


def _strip_chat_suffix(endpoint: str) -> str:
    endpoint = (endpoint or DEFAULT_LOCAL_CHAT_ENDPOINT).strip().rstrip("/")
    suffix = "/chat/completions"
    if endpoint.endswith(suffix):
        endpoint = endpoint[: -len(suffix)]
    return endpoint


def get_local_endpoint() -> str:
    """OpenAI-compatible base URL, normalized for openai.OpenAI(base_url=...)."""
    return _strip_chat_suffix(str(get_config().get("local_endpoint") or DEFAULT_LOCAL_CHAT_ENDPOINT))


def get_local_chat_endpoint() -> str:
    """Full /v1/chat/completions URL for health checks and plain HTTP clients."""
    base = get_local_endpoint().rstrip("/")
    return f"{base}/chat/completions"


def get_local_server_base() -> str:
    """Server root URL, e.g. http://127.0.0.1:8000."""
    base = get_local_endpoint().rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def get_python_path() -> str:
    return str(get_config().get("python_path") or sys.executable)


def get_microphone_device() -> str:
    return str(get_config().get("microphone_device", "") or "").strip()


def is_web_agent_enabled() -> bool:
    return bool(get_config().get("web_agent_enabled", True))


def is_safe_mode() -> bool:
    return bool(get_config().get("safe_mode", True))


def get_default_browser() -> str:
    browser = str(get_config().get("default_browser") or "brave").lower().strip()
    return browser or "brave"


def get_ollama_endpoint() -> str:
    """Backwards compatibility alias. This project no longer requires Ollama."""
    return get_local_endpoint()


def get_ollama_model() -> str:
    """Backwards compatibility alias. This project no longer requires Ollama."""
    return get_local_model()
