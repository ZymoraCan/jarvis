import re

from config import get_model_type, is_safe_mode, is_web_agent_enabled
from doctor import run_doctor


_APP_NAMES = {
    "chrome": "Chrome",
    "google chrome": "Chrome",
    "edge": "Edge",
    "firefox": "Firefox",
    "opera": "Opera",
    "opera gx": "Opera",
    "vscode": "Visual Studio Code",
    "visual studio code": "Visual Studio Code",
    "notepad": "Notepad",
    "defter": "Notepad",
    "calculator": "Calculator",
    "hesap makinesi": "Calculator",
    "explorer": "File Explorer",
    "dosya gezgini": "File Explorer",
    "steam": "Steam",
    "spotify": "Spotify",
    "discord": "Discord",
    "telegram": "Telegram",
    "whatsapp": "WhatsApp",
}

_CONFIRM_WORDS = ("onayla", "onaylıyorum", "confirm", "confirmed", "evet", "yes")


def _confirmed(text: str) -> bool:
    return any(word in text for word in _CONFIRM_WORDS)


def _after_keyword(text: str, keywords: tuple[str, ...]) -> str:
    for keyword in keywords:
        idx = text.find(keyword)
        if idx >= 0:
            return text[idx + len(keyword):].strip(" :,-")
    return ""


def _safe_block(reason: str) -> str:
    return f"Safe mode blocked this action: {reason}. Add 'onayla' if you really want to continue."


def _folder_from_text(text: str) -> str:
    if "download" in text or "indirilen" in text:
        return "downloads"
    if "document" in text or "belge" in text:
        return "documents"
    if "masaüst" in text or "desktop" in text:
        return "desktop"
    return "desktop"


def route_local_command(user_text: str, player=None) -> str | None:
    """Deterministic local-first router. Returns None when LLM chat should handle it."""
    text = (user_text or "").strip()
    low = text.lower()
    if not low:
        return None

    if low in {"doctor", "doktor", "health", "health check", "sistem kontrol"}:
        return run_doctor()

    if any(word in low for word in ("shutdown", "kapat bilgisayarı", "bilgisayarı kapat", "restart", "yeniden başlat")):
        if is_safe_mode() and not _confirmed(low):
            return _safe_block("PC shutdown/restart requires confirmation")
        from actions.computer_settings import computer_settings

        action = "restart" if "restart" in low or "yeniden" in low else "shutdown"
        return computer_settings({"action": action, "confirmed": "yes"}, player=player)

    if "ekran görüntüsü" in low or "screenshot" in low:
        from actions.computer_control import computer_control

        return computer_control({"action": "screenshot"}, player=player)

    if "masaüst" in low and any(word in low for word in ("liste", "göster", "show", "list")):
        from actions.file_controller import file_controller

        return file_controller({"action": "list", "path": "desktop"}, player=player)

    if any(word in low for word in ("dosyaları listele", "klasörü listele", "list files", "show files")):
        from actions.file_controller import file_controller

        return file_controller({"action": "list", "path": _folder_from_text(low)}, player=player)

    if low.startswith(("dosya ara", "file ara", "find file", "dosya bul")):
        from actions.file_controller import file_controller

        query = _after_keyword(low, ("dosya ara", "file ara", "find file", "dosya bul"))
        if not query:
            return "Aranacak dosya adını yazmalısın."
        return file_controller({"action": "find", "path": "home", "name": query}, player=player)

    if any(word in low for word in ("sil", "delete", "çöp")) and ("dosya" in low or "file" in low):
        if is_safe_mode() and not _confirmed(low):
            return _safe_block("file deletion requires confirmation")
        return "Silme işlemi için net dosya yolu/adı gerekli. Örnek: 'Desktop test.txt sil onayla'."

    if low.startswith(("webde ara", "internette ara", "google'da ara", "google ara", "search web")):
        if not is_web_agent_enabled():
            return "Web agent config içinde kapalı."
        from actions.browser_control import browser_control

        query = _after_keyword(low, ("webde ara", "internette ara", "google'da ara", "google ara", "search web"))
        if not query:
            return "Aranacak metni yazmalısın."
        return browser_control({"action": "search", "query": query, "engine": "google"}, player=player)

    if low.startswith(("youtube'da aç", "youtube aç", "youtube play")):
        from actions.youtube_video import youtube_video

        query = _after_keyword(low, ("youtube'da aç", "youtube aç", "youtube play"))
        if not query:
            return "YouTube'da ne açacağımı yazmalısın."
        return youtube_video({"action": "play", "query": query}, player=player)

    if "hava durumu" in low or "weather" in low:
        from actions.weather_report import weather_action

        city = _after_keyword(low, ("hava durumu", "weather in", "weather"))
        city = city or "Istanbul"
        return weather_action({"city": city}, player=player)

    if "ses" in low or "volume" in low:
        from actions.computer_settings import computer_settings

        if "art" in low or "up" in low:
            return computer_settings({"action": "volume_up"}, player=player)
        if "azalt" in low or "down" in low:
            return computer_settings({"action": "volume_down"}, player=player)
        if "mute" in low or "sessiz" in low:
            return computer_settings({"action": "mute"}, player=player)

    if get_model_type() == "local" and any(word in low for word in ("ekranı analiz", "ekranda ne", "kamerada ne", "ne görüyorsun")):
        return "Ekran/kamera analizi Gemini gerektiriyor. Local mode'da cloud API çağrısı yapılmadı."

    if any(word in low for word in ("aç", "ac", "open", "başlat", "baslat", "launch")):
        from actions.open_app import open_app

        for key, app_name in _APP_NAMES.items():
            if key in low:
                return open_app({"app_name": app_name}, player=player)

        match = re.search(r"(?:aç|ac|open|başlat|baslat|launch)\s+(.+)$", low)
        if match:
            app = match.group(1).strip()
            if app:
                return open_app({"app_name": app}, player=player)

    return None
