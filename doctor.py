import importlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import requests

from config import (
    get_config,
    get_config_path,
    get_gemini_key,
    get_local_server_base,
    get_model_type,
)


REQUIRED_PACKAGES = {
    "openai": "openai",
    "requests": "requests",
    "ddgs": "ddgs",
    "PIL": "pillow",
    "google.genai": "google-genai",
    "google.generativeai": "google-generativeai",
    "playwright.async_api": "playwright",
    "pyautogui": "pyautogui",
    "pyperclip": "pyperclip",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "mss": "mss",
    "torch": "torch",
    "transformers": "transformers",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "sounddevice": "sounddevice",
    "send2trash": "send2trash",
}


def _line(ok: bool, label: str, detail: str = "") -> str:
    status = "OK" if ok else "WARN"
    suffix = f" - {detail}" if detail else ""
    return f"[{status}] {label}{suffix}"


def _import_ok(module_name: str) -> tuple[bool, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "")
        return True, str(version) if version else "installed"
    except Exception as exc:
        return False, str(exc)


def _check_packages() -> list[str]:
    lines = ["Packages:"]
    for module_name, package_name in REQUIRED_PACKAGES.items():
        ok, detail = _import_ok(module_name)
        lines.append("  " + _line(ok, package_name, detail))
    return lines


def _check_local_server() -> list[str]:
    lines = ["Local Model Server:"]
    base = get_local_server_base().rstrip("/")
    health_url = f"{base}/health"
    lines.append(f"  Endpoint: {health_url}")
    try:
        response = requests.get(health_url, timeout=3)
        lines.append("  " + _line(response.ok, "health endpoint", f"HTTP {response.status_code}"))
        if response.ok:
            data = response.json()
            lines.append("  " + _line(True, "server_running", str(data.get("server_running"))))
            lines.append("  " + _line(bool(data.get("model_loaded")), "model_loaded", str(data.get("model_loaded"))))
            lines.append(f"  Model: {data.get('model_name')}")
            lines.append(f"  Device: {data.get('device')}")
            cuda = data.get("cuda", {})
            lines.append(f"  CUDA: {cuda.get('available')} ({cuda.get('name', 'n/a')})")
            if data.get("error"):
                lines.append("  " + _line(False, "server_error", data.get("error")))
    except Exception as exc:
        lines.append("  " + _line(False, "health endpoint", str(exc)))
    return lines


def _check_gemini() -> list[str]:
    lines = ["Gemini:"]
    key = get_gemini_key()
    lines.append("  " + _line(bool(key), "api key", "present" if key else "not configured"))
    if not key:
        lines.append("  " + _line(False, "connection", "skipped; no api key"))
        return lines

    try:
        from google import genai

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents="Reply with exactly: ok",
        )
        text = (getattr(response, "text", "") or "").strip()
        lines.append("  " + _line(bool(text), "connection", text[:80] or "empty response"))
    except Exception as exc:
        detail = str(exc)
        lower = detail.lower()
        if "429" in lower or "resource_exhausted" in lower or "quota" in lower or "rate limit" in lower:
            lines.append("  " + _line(False, "connection", "quota/limit dolu (429 RESOURCE_EXHAUSTED)"))
            lines.append("  " + _line(True, "safe fallback", "aktif pencere/session bilgisi kullanılabilir; Vision/brain yokken otomatik tıklama-yazma durdurulur"))
        else:
            lines.append("  " + _line(False, "connection", detail[:240]))
    return lines


def _check_playwright() -> list[str]:
    lines = ["Playwright:"]
    ok, detail = _import_ok("playwright.async_api")
    lines.append("  " + _line(ok, "python package", detail))
    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "--version"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        out = (result.stdout or result.stderr).strip()
        lines.append("  " + _line(result.returncode == 0, "cli", out or f"exit={result.returncode}"))
    except Exception as exc:
        lines.append("  " + _line(False, "cli", str(exc)))
    root = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "ms-playwright"
    chromium_dirs = list(root.glob("chromium-*")) if root.exists() else []
    lines.append("  " + _line(bool(chromium_dirs), "chromium browser", str(chromium_dirs[0]) if chromium_dirs else "not installed"))
    return lines


def _check_microphone() -> list[str]:
    lines = ["Microphone:"]
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_count = sum(1 for dev in devices if dev.get("max_input_channels", 0) > 0)
        lines.append("  " + _line(input_count > 0, "input devices", str(input_count)))
    except Exception as exc:
        lines.append("  " + _line(False, "sounddevice query", str(exc)))
    return lines


def _check_torch() -> list[str]:
    lines = ["Torch / CUDA:"]
    try:
        import torch

        lines.append("  " + _line(True, "torch", getattr(torch, "__version__", "installed")))
        cuda = torch.cuda.is_available()
        lines.append("  " + _line(cuda, "cuda_available", str(cuda)))
        if cuda:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            vram = round(props.total_memory / (1024 ** 3), 2)
            lines.append(f"  Device: {idx} - {props.name} ({vram} GB VRAM)")
    except Exception as exc:
        lines.append("  " + _line(False, "torch", str(exc)))
    return lines


def run_doctor() -> str:
    base_dir = Path(__file__).resolve().parent
    cfg_path = get_config_path()
    cfg = get_config()
    lines = [
        "JARVIS Doctor",
        "=" * 60,
        _line(sys.version_info[:2] in ((3, 11), (3, 12)), "Python version", sys.version.split()[0]),
        _line(cfg_path.exists(), "config file", str(cfg_path)),
        _line(get_model_type() in ("gemini", "local"), "active model_type", get_model_type()),
        f"Config: {json.dumps({k: ('***' if k == 'gemini_api_key' and v else v) for k, v in cfg.items()}, ensure_ascii=False)}",
        _line(bool(get_gemini_key()), "Gemini key", "present" if get_gemini_key() else "not configured"),
        _line((base_dir / "actions").exists(), "actions folder", str(base_dir / "actions")),
    ]

    lines.extend(_check_packages())
    lines.extend(_check_gemini())
    if get_model_type() == "local":
        lines.extend(_check_torch())
        lines.extend(_check_local_server())
    else:
        lines.extend(["Local Model Server:", "  [OK] optional in current Gemini mode"])
    lines.extend(_check_playwright())
    lines.extend(_check_microphone())
    return "\n".join(lines)


if __name__ == "__main__":
    print(run_doctor())
