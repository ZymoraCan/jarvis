from __future__ import annotations

import json
import platform
import time
from pathlib import Path

STATE_PATH = Path(__file__).resolve().parent / "memory" / "session_state.json"

DEFAULT_STATE = {
    "last_target_app": "",
    "last_target_window_title": "",
    "last_target_contact_or_page": "",
    "last_successful_action": "",
    "last_screen_summary": "",
    "last_vision_app_guess": "",
    "last_visible_page_or_screen": "",
    "last_visible_buttons": [],
    "last_visible_input_fields": [],
    "last_screen_analysis_time": 0,
    "last_screen_risk_level": "",
    "last_screen_can_continue": "",
    "last_screen_confidence": "",
    "last_context_conflict": "",
    "last_known_input_field": "",
    "last_action_time": 0,
    "current_task_context": "",
}

APP_HINTS = {
    "whatsapp": ("whatsapp", "whats app"),
    "chrome": ("chrome", "google chrome"),
    "google chrome": ("chrome", "google chrome"),
    "brave": ("brave",),
    "edge": ("edge", "microsoft edge"),
    "youtube": ("youtube",),
    "file explorer": ("file explorer", "explorer"),
    "explorer": ("file explorer", "explorer"),
    "settings": ("settings", "ayarlar"),
}

PROCESS_APP_HINTS = {
    "chrome": "chrome",
    "msedge": "edge",
    "brave": "brave",
    "firefox": "firefox",
    "whatsapp": "whatsapp",
    "explorer": "file explorer",
    "applicationframehost": "settings",
}

CONTINUATION_HINTS = (
    "bir de",
    "bunu da",
    "aynı kişiye",
    "ayni kisiye",
    "oradan devam",
    "oradan",
    "devam et",
    "şimdi şuna",
    "simdi suna",
    "açık olan",
    "acik olan",
    "aynı ekranda",
    "ayni ekranda",
    "aynı sayfada",
    "ayni sayfada",
    "aynı klasörde",
    "ayni klasorde",
    "şunu yaz",
    "sunu yaz",
)


def load_task_state() -> dict:
    try:
        if STATE_PATH.exists():
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {**DEFAULT_STATE, **data}
    except Exception:
        pass
    return dict(DEFAULT_STATE)


def save_task_state(state: dict) -> dict:
    data = {**DEFAULT_STATE, **(state or {})}
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def update_task_state(**updates) -> dict:
    state = load_task_state()
    for key, value in updates.items():
        if key in DEFAULT_STATE:
            state[key] = value if value is not None else ""
    state["last_action_time"] = time.time()
    return save_task_state(state)


def clear_task_state() -> dict:
    return save_task_state(dict(DEFAULT_STATE))


def get_active_window_context() -> dict:
    title = ""
    app = ""
    process_name = ""
    system = platform.system().lower()

    try:
        import pygetwindow as gw

        win = gw.getActiveWindow()
        if win:
            title = win.title or ""
    except Exception:
        title = ""

    if system == "windows":
        try:
            import ctypes
            import psutil

            hwnd = ctypes.windll.user32.GetForegroundWindow()
            pid = ctypes.c_ulong()
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            process_name = Path(psutil.Process(pid.value).name()).stem.lower()
            app = PROCESS_APP_HINTS.get(process_name, process_name)
        except Exception:
            pass

    if system == "windows" and not title:
        try:
            import subprocess

            script = (
                "Get-Process | Where-Object {$_.MainWindowTitle} | "
                "Sort-Object StartTime -Descending | Select-Object -First 1 "
                "ProcessName,MainWindowTitle | ConvertTo-Json -Compress"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.stdout.strip():
                data = json.loads(result.stdout)
                title = data.get("MainWindowTitle", "") or title
                process_name = data.get("ProcessName", "") or process_name
                app = PROCESS_APP_HINTS.get(process_name.lower(), process_name) or app
        except Exception:
            pass

    if not app:
        app = infer_app_from_title(title)

    return {
        "active_window_title": title,
        "active_app": app,
        "process_name": process_name,
        "os": platform.system(),
        "checked_at": time.time(),
    }


def infer_app_from_title(title: str) -> str:
    lower = (title or "").lower()
    for app, hints in APP_HINTS.items():
        if any(hint in lower for hint in hints):
            return app
    return ""


def summarize_current_screen(take_screenshot: bool = False) -> str:
    ctx = get_active_window_context()
    title = ctx.get("active_window_title") or "unknown window"
    app = ctx.get("active_app") or "unknown app"
    summary = f"Active app: {app}; window title: {title}"

    if take_screenshot:
        try:
            import pyautogui

            path = Path.home() / "Desktop" / "jarvis_context_screen.png"
            pyautogui.screenshot().save(str(path))
            summary += f"; screenshot: {path}"
        except Exception as exc:
            summary += f"; screenshot unavailable: {exc}"

    return summary


def screen_context_check(target_app: str = "", target_page: str = "", take_screenshot: bool = False) -> dict:
    ctx = get_active_window_context()
    ctx["screen_summary"] = summarize_current_screen(take_screenshot=take_screenshot)
    ctx["target_app"] = target_app or ""
    ctx["target_page"] = target_page or ""
    return ctx


def analyze_current_screen_with_vision(force: bool = False, max_age_seconds: int = 45) -> dict:
    state = load_task_state()
    last_time = float(state.get("last_screen_analysis_time") or 0)
    if not force and last_time and time.time() - last_time <= max_age_seconds:
        return {
            "active_app_guess": state.get("last_vision_app_guess", ""),
            "visible_page_or_screen": state.get("last_visible_page_or_screen", ""),
            "visible_text_summary": state.get("last_screen_summary", ""),
            "possible_input_fields": state.get("last_visible_input_fields", []),
            "possible_buttons": state.get("last_visible_buttons", []),
            "risk_level": state.get("last_screen_risk_level", ""),
            "can_continue_current_task": state.get("last_screen_can_continue", ""),
            "reason": "Using recent cached screen analysis.",
        }

    try:
        from actions.screen_processor import analyze_screen_once

        analysis = analyze_screen_once()
    except Exception as exc:
        analysis = {
            "active_app_guess": "",
            "visible_page_or_screen": "",
            "visible_text_summary": f"Screen analysis failed: {exc}",
            "possible_input_fields": [],
            "possible_buttons": [],
            "risk_level": "unknown",
            "can_continue_current_task": False,
            "reason": "Gemini Vision analysis could not run; using title/process fallback only.",
        }

    unavailable = (
        not analysis.get("active_app_guess")
        and not analysis.get("possible_buttons")
        and not analysis.get("possible_input_fields")
        and str(analysis.get("risk_level", "")).lower() == "unknown"
        and not bool(analysis.get("can_continue_current_task", False))
    )
    confidence = screen_context_confidence(analysis=analysis)
    updates = {
        "last_screen_analysis_time": time.time(),
        "last_screen_risk_level": analysis.get("risk_level", ""),
        "last_screen_can_continue": analysis.get("can_continue_current_task", ""),
        "last_screen_confidence": confidence.get("confidence", ""),
        "last_context_conflict": confidence.get("reason", "") if confidence.get("confidence") == "conflict" else "",
    }
    if confidence.get("confidence") in {"high", "medium"} and not unavailable:
        updates.update({
            "last_vision_app_guess": analysis.get("active_app_guess", ""),
            "last_visible_page_or_screen": analysis.get("visible_page_or_screen", ""),
            "last_screen_summary": analysis.get("visible_text_summary", ""),
            "last_visible_buttons": analysis.get("possible_buttons", []),
            "last_visible_input_fields": analysis.get("possible_input_fields", []),
        })
    update_task_state(**updates)
    return analysis


def should_reuse_current_screen(target_app: str = "", target_page: str = "", max_age_seconds: int = 600) -> tuple[bool, str]:
    state = load_task_state()
    ctx = get_active_window_context()
    active_title = (ctx.get("active_window_title") or "").lower()
    active_app = (ctx.get("active_app") or "").lower()
    target = (target_app or "").lower().strip()
    page = (target_page or "").lower().strip()
    browser_apps = {"chrome", "edge", "brave", "firefox"}

    if target:
        hints = APP_HINTS.get(target, (target,))
        target_norm = _normalize_app_name(target)
        if target_norm in browser_apps and active_app in browser_apps and active_app != target_norm:
            return False, f"Active browser is {active_app}, not {target_norm}."
        if target_norm and target_norm == active_app or any(hint in active_title for hint in hints):
            if not page or page in active_title:
                return True, f"{target_app} already appears active."

    recent = time.time() - float(state.get("last_action_time") or 0) <= max_age_seconds
    if recent and not target and state.get("last_target_app"):
        return True, "Recent task context is available."

    return False, "Current screen does not match the requested target."


def _normalize_app_name(value: str) -> str:
    lower = (value or "").lower().strip()
    if not lower:
        return ""
    for app, hints in APP_HINTS.items():
        if lower == app or any(hint in lower for hint in hints):
            return app
    if "youtube" in lower:
        return "youtube"
    if "valorant" in lower or "game" in lower or "oyun" in lower:
        return "game"
    return lower.split()[0]


def screen_context_confidence(analysis: dict | None = None, target_app: str = "") -> dict:
    state = load_task_state()
    active = get_active_window_context()
    if analysis is None:
        analysis = analyze_current_screen_with_vision(force=False)

    active_app = _normalize_app_name(active.get("active_app", ""))
    vision_app = _normalize_app_name(str((analysis or {}).get("active_app_guess", "")))
    session_app = _normalize_app_name(target_app or str(state.get("last_target_app", "")))
    risk = str((analysis or {}).get("risk_level", "unknown")).lower()

    conflicts = []
    if active_app and vision_app and active_app != vision_app:
        if not (active_app in {"brave", "chrome", "edge", "firefox"} and vision_app == "youtube"):
            conflicts.append(f"active window is {active_app}, vision sees {vision_app}")
    if session_app and vision_app and session_app != vision_app:
        if not (session_app in {"brave", "chrome", "edge", "firefox"} and vision_app == "youtube"):
            conflicts.append(f"session target is {session_app}, vision sees {vision_app}")
    if session_app and active_app and session_app != active_app:
        if not (active_app in {"brave", "chrome", "edge", "firefox"} and session_app == "youtube"):
            conflicts.append(f"session target is {session_app}, active window is {active_app}")

    if risk in {"high", "critical"}:
        confidence = "conflict"
        conflicts.append(f"vision risk is {risk}")
    elif conflicts:
        confidence = "conflict"
    elif active_app and vision_app and (active_app == vision_app or vision_app == "youtube" and active_app in {"brave", "chrome", "edge", "firefox"}):
        confidence = "high"
    elif active_app or vision_app or session_app:
        confidence = "medium"
    else:
        confidence = "low"

    can_continue = bool((analysis or {}).get("can_continue_current_task", False))
    can_act = confidence in {"high", "medium"} and risk not in {"high", "critical"} and (can_continue or confidence == "high")
    reason = "; ".join(conflicts) if conflicts else f"context confidence is {confidence}"

    return {
        "confidence": confidence,
        "reason": reason,
        "active_window_app": active_app,
        "vision_app_guess": vision_app,
        "session_target_app": session_app,
        "can_act_safely": bool(can_act),
        "risk_level": risk or "unknown",
    }


def format_screen_status() -> str:
    analysis = analyze_current_screen_with_vision(force=True)
    confidence = screen_context_confidence(analysis=analysis)
    active = get_active_window_context()
    unavailable = (
        not analysis.get("active_app_guess")
        and not analysis.get("possible_buttons")
        and not analysis.get("possible_input_fields")
        and confidence.get("risk_level") == "unknown"
    )
    visual_line = (
        "Gemini Vision şu an kullanılamıyor; yalnızca aktif pencere/process/session bilgisi gösteriliyor."
        if unavailable
        else analysis.get("visible_text_summary") or "ekran analizi yapılamadı"
    )
    return (
        f"Aktif pencere: {active.get('active_app') or 'bilinmiyor'} - {active.get('active_window_title') or 'başlık yok'}\n"
        f"Görsel analiz: {visual_line}\n"
        f"Muhtemel uygulama: {analysis.get('active_app_guess') or 'bilinmiyor'}\n"
        f"Güven seviyesi: {confidence.get('confidence')} ({confidence.get('reason')})\n"
        f"Devam edilebilir mi: {confidence.get('can_act_safely')}"
    )


def is_continuation_command(text: str) -> bool:
    lower = (text or "").lower().strip()
    return any(hint in lower for hint in CONTINUATION_HINTS)


def continuation_context_for_prompt(text: str) -> str:
    if not is_continuation_command(text):
        return text

    state = load_task_state()
    screen = summarize_current_screen(take_screenshot=False)
    vision = analyze_current_screen_with_vision(force=True)
    context = {
        "last_target_app": state.get("last_target_app", ""),
        "last_target_window_title": state.get("last_target_window_title", ""),
        "last_target_contact_or_page": state.get("last_target_contact_or_page", ""),
        "last_successful_action": state.get("last_successful_action", ""),
        "last_screen_summary": state.get("last_screen_summary", ""),
        "last_known_input_field": state.get("last_known_input_field", ""),
        "last_visible_buttons": state.get("last_visible_buttons", []),
        "last_visible_input_fields": state.get("last_visible_input_fields", []),
        "current_task_context": state.get("current_task_context", ""),
        "current_screen": screen,
        "vision_screen_analysis": vision,
    }
    return (
        "[SESSION CONTEXT]\n"
        f"{json.dumps(context, ensure_ascii=False)}\n\n"
        "[USER CONTINUATION COMMAND]\n"
        f"{text}\n\n"
        "Use the existing screen/task context if it is still valid. "
        "Do not reopen the same app or restart the workflow unless the current screen is not suitable. "
        "For risky actions, ask for confirmation."
    )


def safe_click_or_ask(description: str, click_func) -> str:
    if not description:
        return "I need a visible target description before clicking."
    analysis = analyze_current_screen_with_vision(force=True)
    confidence = screen_context_confidence(analysis=analysis)
    if not confidence.get("can_act_safely"):
        return f"Ekran bağlamı net değil, devam edemiyorum: {confidence.get('reason')}"
    risk = str(analysis.get("risk_level", "")).lower()
    if risk in {"high", "critical"}:
        return "This screen looks risky. Please confirm before I click anything here."
    coords = click_func(description)
    if not coords:
        return "I cannot clearly see the target element. Please confirm the correct screen before I click."
    x, y = coords
    try:
        import pyautogui

        pyautogui.click(x, y)
        update_task_state(
            last_successful_action=f"screen_click:{description}",
            last_known_input_field=description,
            last_screen_summary=summarize_current_screen(False),
        )
        return f"Clicked visible target: {description} at ({x}, {y})"
    except Exception as exc:
        return f"Click failed: {exc}"
