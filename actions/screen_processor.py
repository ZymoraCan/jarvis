from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

try:
    import pytesseract
    _TESSERACT = True
except ImportError:
    _TESSERACT = False

import numpy as np
import sounddevice as sd

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    import mss
    import mss.tools
    _MSS = True
except ImportError:
    _MSS = False

try:
    import PIL.Image
    import PIL.ImageEnhance
    import PIL.ImageFilter
    import PIL.ImageGrab
    _PIL = True
except ImportError:
    _PIL = False

from google import genai
from google.genai import types as gtypes
from config import get_model_type

def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


_BASE        = _base_dir()
_CONFIG_PATH = _BASE / "config" / "api_keys.json"


def _load_config() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_config_key(key: str, value) -> None:
    try:
        cfg = _load_config()
        cfg[key] = value
        _CONFIG_PATH.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    except Exception as e:
        print(f"[Vision] ⚠️  Could not save config key '{key}': {e}")


def _get_api_key() -> str:
    key = _load_config().get("gemini_api_key", "")
    if not key:
        raise RuntimeError("gemini_api_key not found in config.")
    return key


def _get_os() -> str:
    return _load_config().get("os_system", "windows").lower()

_LIVE_MODEL         = "models/gemini-2.5-flash-native-audio-preview-12-2025"
_CHANNELS           = 1
_RECEIVE_SAMPLE_RATE = 24_000
_CHUNK_SIZE         = 1_024

_IMG_MAX_W = 640
_IMG_MAX_H = 360
_JPEG_Q    = 60

_SYSTEM_PROMPT = (
    "You are JARVIS, an advanced AI assistant. "
    "Analyze the provided image with precision and intelligence. "
    "Be concise and direct — maximum two sentences unless the user's question "
    "requires more detail. "
    "Address the user respectfully. "
    "Always call the appropriate tool; never simulate results."
)

_SCREEN_ANALYSIS_DEFAULT = {
    "active_app_guess": "",
    "visible_page_or_screen": "",
    "visible_text_summary": "",
    "possible_input_fields": [],
    "possible_buttons": [],
    "risk_level": "unknown",
    "can_continue_current_task": False,
    "reason": "",
    "ocr_texts": [],
    "ocr_candidates": [],
}

_OCR_ANALYSIS_DEFAULT = {
    "found_texts": [],
    "possible_input_fields": [],
    "possible_buttons": [],
    "possible_titles": [],
    "notes": "",
    "raw_text": "",
    "debug_screenshot_path": "",
    "processed_screenshot_path": "",
    "image_size": "",
    "image_mode": "",
    "screen_size": "",
    "screenshot_size": "",
    "processed_size": "",
    "monitor_mode": "primary",
    "image_to_string_text": "",
    "image_to_data_count": 0,
}


def _is_quota_error(exc: Exception | str) -> bool:
    text = str(exc).lower()
    return "429" in text or "resource_exhausted" in text or "quota" in text or "rate limit" in text


def _save_screen_analysis_to_state(analysis: dict) -> None:
    try:
        from session_state import screen_context_confidence, update_task_state

        confidence = screen_context_confidence(analysis=analysis)
        unavailable = (
            not analysis.get("active_app_guess")
            and not analysis.get("possible_buttons")
            and not analysis.get("possible_input_fields")
            and str(analysis.get("risk_level", "")).lower() == "unknown"
            and not bool(analysis.get("can_continue_current_task", False))
        )
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
    except Exception as exc:
        print(f"[Vision] Could not save screen analysis to session state: {exc}")


def _compress(img_bytes: bytes, source_format: str = "PNG") -> tuple[bytes, str]:
    if not _PIL:
        return img_bytes, f"image/{source_format.lower()}"

    try:
        img = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((_IMG_MAX_W, _IMG_MAX_H), PIL.Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_Q, optimize=False)
        return buf.getvalue(), "image/jpeg"
    except Exception as e:
        print(f"[Vision] ⚠️  Image compress failed: {e}")
        return img_bytes, f"image/{source_format.lower()}"

def _capture_screen() -> tuple[bytes, str]:

    if not _MSS:
        raise RuntimeError("mss is not installed. Run: pip install mss")

    with mss.mss() as sct:
        monitors = sct.monitors          # [0] = all combined, [1..n] = real screens
        target   = monitors[1] if len(monitors) > 1 else monitors[0]
        shot     = sct.grab(target)
        png      = mss.tools.to_png(shot.rgb, shot.size)

    return _compress(png, "PNG")


def _capture_ocr_screen_image(monitor_mode: str = "primary"):
    if not _PIL:
        raise RuntimeError("Pillow is required for OCR screenshot capture.")
    if monitor_mode == "all":
        return PIL.ImageGrab.grab(all_screens=True).convert("RGB")
    return PIL.ImageGrab.grab().convert("RGB")


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
    raise ValueError("Gemini Vision did not return valid JSON.")


def _sanitize_for_json_parse(text: str) -> str:
    text = (text or "").encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return "".join(ch for ch in text if ch >= " " or ch in "{}[],:\"")


def _sanitize_ocr_text(text: str, limit: int = 4000) -> str:
    text = (text or "").encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = text.replace('"', "").replace("'", "")
    text = re.sub(r"\s+", " ", text)
    text = "".join(ch for ch in text if ch >= " ")
    return text.strip()[:limit]


def _safe_list(value, limit: int) -> list:
    if not isinstance(value, list):
        return []
    return [_sanitize_ocr_value(item) for item in value[:limit]]


def _sanitize_ocr_value(value):
    if isinstance(value, str):
        return _sanitize_ocr_text(value)
    if isinstance(value, list):
        return [_sanitize_ocr_value(item) for item in value]
    if isinstance(value, dict):
        return {
            _sanitize_ocr_text(str(key), limit=120): _sanitize_ocr_value(item)
            for key, item in value.items()
        }
    return value


def _extract_ocr_data(image_source, monitor_mode: str = "primary") -> dict:
    output = {
        "texts": [],
        "raw_text": "",
        "status": "ok",
        "reason": "",
        "debug_screenshot_path": "",
        "processed_screenshot_path": "",
        "image_size": "",
        "image_mode": "",
        "screen_size": "",
        "screenshot_size": "",
        "processed_size": "",
        "monitor_mode": monitor_mode,
        "image_to_string_text": "",
        "image_to_data_count": 0,
    }
    if not _PIL:
        output["status"] = "unavailable"
        output["reason"] = "Pillow is required for OCR preprocessing."
        return output
    if not _TESSERACT:
        output["status"] = "unavailable"
        output["reason"] = "pytesseract is not installed."
        return output

    try:
        if isinstance(image_source, PIL.Image.Image):
            img = image_source.convert("RGB")
        else:
            img = PIL.Image.open(io.BytesIO(image_source)).convert("RGB")
        output["image_size"] = f"{img.size[0]}x{img.size[1]}"
        output["screen_size"] = output["image_size"]
        output["screenshot_size"] = output["image_size"]
        output["image_mode"] = img.mode

        debug_path = Path.home() / "Desktop" / "jarvis_ocr_debug.png"
        processed_path = Path.home() / "Desktop" / "jarvis_ocr_processed.png"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(debug_path)
        output["debug_screenshot_path"] = str(debug_path)

        gray = np.asarray(img.convert("L"))
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        notes = [
            f"monitor_mode={output['monitor_mode']}",
            f"original_size={output['image_size']}",
            f"screenshot_size={output['screenshot_size']}",
            f"screen_size={output['screen_size']}",
            f"mode={output['image_mode']}",
            f"mean={mean:.2f}",
            f"std={std:.2f}",
        ]
        if img.size[0] < 1000:
            notes.append("LOW RES SCREENSHOT")
        if mean < 5:
            notes.append("Screenshot appears nearly black.")
        elif mean > 250:
            notes.append("Screenshot appears nearly white.")
        elif std < 3:
            notes.append("Screenshot appears very low contrast / possibly blank.")

        resampling = getattr(PIL.Image, "Resampling", PIL.Image)
        processed = img.convert("L")
        if mean < 80:
            processed = PIL.ImageEnhance.Contrast(processed).enhance(1.2)
            processed = processed.filter(PIL.ImageFilter.SHARPEN)
            notes.append("processed pipeline: grayscale + contrast 1.2 + sharpen + 2x resize.")
        else:
            notes.append("processed pipeline: grayscale + 2x resize.")
        processed = processed.resize((processed.width * 2, processed.height * 2), resampling.BICUBIC)
        output["processed_size"] = f"{processed.size[0]}x{processed.size[1]}"
        notes.append(f"processed_size={output['processed_size']}")
        processed.save(processed_path)
        output["processed_screenshot_path"] = str(processed_path)
    except Exception as exc:
        output["status"] = "error"
        output["reason"] = str(exc)[:220]
        return output

    ocr_config = "--oem 3 --psm 6"
    def run_ocr_pipeline(name: str, ocr_img, scale: float) -> dict:
        pipeline = {"name": name, "texts": [], "raw_text": "", "data_count": 0, "error": ""}
        try:
            pipeline["raw_text"] = _sanitize_ocr_text(pytesseract.image_to_string(ocr_img, config=ocr_config))
        except Exception as exc:
            pipeline["error"] = f"image_to_string error: {str(exc)[:180]}"

        try:
            data = pytesseract.image_to_data(ocr_img, output_type=pytesseract.Output.DICT, config=ocr_config)
        except Exception as exc:
            extra = f"image_to_data error: {str(exc)[:180]}"
            pipeline["error"] = f"{pipeline['error']} {extra}".strip()
            return pipeline

        pipeline["data_count"] = len(data.get("text", []))
        for i in range(pipeline["data_count"]):
            text = (data["text"][i] or "").strip()
            conf_raw = str(data.get("conf", ["-1"])[i])
            try:
                conf = float(conf_raw)
            except Exception:
                conf = -1.0
            if not text or conf < 35:
                continue

            item = {
                "text": _sanitize_ocr_text(text[:160]),
                "x": int(data["left"][i] / scale),
                "y": int(data["top"][i] / scale),
                "width": int(data["width"][i] / scale),
                "height": int(data["height"][i] / scale),
                "confidence": round(conf, 2),
            }
            pipeline["texts"].append(item)

        pipeline["texts"] = pipeline["texts"][:150]
        if not pipeline["raw_text"]:
            pipeline["raw_text"] = _sanitize_ocr_text(" ".join(item["text"] for item in pipeline["texts"]))
        return pipeline

    original_result = run_ocr_pipeline("original", img, 1.0)
    processed_result = run_ocr_pipeline("processed", processed, 2.0)
    chosen = max(
        [original_result, processed_result],
        key=lambda item: (len(item["texts"]), len(item["raw_text"])),
    )

    output["texts"] = chosen["texts"]
    output["raw_text"] = chosen["raw_text"]
    output["image_to_string_text"] = chosen["raw_text"]
    output["image_to_data_count"] = chosen["data_count"]

    for pipeline in (original_result, processed_result):
        if pipeline["error"]:
            notes.append(f"{pipeline['name']} pipeline {pipeline['error']}")
        notes.append(
            f"{pipeline['name']} pipeline chars={len(pipeline['raw_text'])}, boxes={len(pipeline['texts'])}/{pipeline['data_count']}"
        )
    notes.append(f"chosen pipeline: {chosen['name']}")

    if output["image_to_string_text"] and not output["texts"]:
        notes.append("image_to_string found text, but image_to_data produced no confident boxes.")
    output["reason"] = " ".join(notes)
    return output


def _ocr_item_to_region(item: dict, label: str | None = None) -> dict:
    return {
        "label": _sanitize_ocr_text(label or item.get("text", ""), limit=120),
        "x": int(item.get("x", 0) or 0),
        "y": int(item.get("y", 0) or 0),
        "width": int(item.get("width", 0) or 0),
        "height": int(item.get("height", 0) or 0),
        "confidence": float(item.get("confidence", 0) or 0),
    }


def _guess_ocr_regions(found_texts: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    input_keywords = {"email", "password", "username", "message", "type", "write", "input"}
    button_keywords = {
        "login", "send", "ok", "cancel", "continue", "submit", "next",
        "back", "yes", "no", "search",
    }

    possible_input_fields = []
    possible_buttons = []
    possible_titles = []
    confident_items = [
        item for item in found_texts
        if float(item.get("confidence", 0) or 0) >= 50
        and _sanitize_ocr_text(str(item.get("text", "")), limit=120)
    ]
    if not confident_items:
        return possible_input_fields, possible_buttons, possible_titles

    heights = sorted(int(item.get("height", 0) or 0) for item in confident_items)
    median_height = heights[len(heights) // 2] if heights else 0
    max_bottom = max(int(item.get("y", 0) or 0) + int(item.get("height", 0) or 0) for item in confident_items)
    title_y_limit = max(140, int(max_bottom * 0.35))

    for item in found_texts:
        if float(item.get("confidence", 0) or 0) < 50:
            continue
        text = _sanitize_ocr_text(str(item.get("text", "")), limit=120)
        low = text.lower().strip(" :")
        if not low:
            continue
        words = set(re.findall(r"[a-z0-9@._-]+", low))
        if len(text) <= 40 and (words & button_keywords):
            possible_buttons.append(_ocr_item_to_region(item, text))
            continue
        if words & input_keywords:
            possible_input_fields.append(_ocr_item_to_region(item, text.rstrip(":")))
            continue

        height = int(item.get("height", 0) or 0)
        y = int(item.get("y", 0) or 0)
        is_upper_title = text.replace(" ", "").isupper() and len(text) >= 3
        is_large = height >= max(18, int(median_height * 1.25))
        if y <= title_y_limit and (is_large or is_upper_title):
            possible_titles.append(_ocr_item_to_region(item, text))

    return possible_input_fields[:30], possible_buttons[:40], possible_titles[:30]


def _apply_raw_text_input_fallback(possible_input_fields: list[dict], found_texts: list[dict], raw_text: str) -> tuple[list[dict], list[str]]:
    fallback_patterns = {
        "EMAIL": r"\b(?:e-mail|email|mail)\b",
        "PASSWORD": r"\bpassword\b",
        "USERNAME": r"\busername\b",
        "MESSAGE": r"\bmessage\b",
    }
    existing_labels = {
        _sanitize_ocr_text(str(item.get("label", ""))).lower()
        for item in possible_input_fields
    }
    found_words = {
        _sanitize_ocr_text(str(item.get("text", ""))).lower()
        for item in found_texts
    }
    notes = []
    for label, pattern in fallback_patterns.items():
        label_key = label.lower()
        if label_key in existing_labels or label_key in found_words:
            continue
        if not re.search(pattern, raw_text or "", flags=re.IGNORECASE):
            continue
        possible_input_fields.append({
            "label": label,
            "x": None,
            "y": None,
            "width": None,
            "height": None,
            "confidence": None,
            "source": "raw_text_fallback",
        })
        notes.append(f"{label} found in raw_text but no bbox")
    return possible_input_fields[:30], notes


def analyze_screen_once(question: str = "") -> dict:
    result = dict(_SCREEN_ANALYSIS_DEFAULT)

    if get_model_type() != "gemini":
        result["reason"] = "Gemini mode is not active."
        _save_screen_analysis_to_state(result)
        return result

    api_key = _load_config().get("gemini_api_key", "").strip()
    if not api_key:
        result["reason"] = "Gemini API key is not configured."
        _save_screen_analysis_to_state(result)
        return result

    try:
        image_bytes, mime_type = _capture_screen()
        ocr_data = _extract_ocr_data(image_bytes)
        client = genai.Client(api_key=api_key)
        prompt = (
            "Analyze this desktop screenshot for safe UI automation. "
            "Return ONLY one compact JSON object with exactly these keys: "
            "active_app_guess, visible_page_or_screen, visible_text_summary, "
            "possible_input_fields, possible_buttons, risk_level, "
            "can_continue_current_task, reason, ocr_texts, ocr_candidates. "
            "possible_input_fields and possible_buttons must be arrays of short strings. "
            "risk_level must be one of: low, medium, high, critical, unknown. "
            "Set can_continue_current_task false when the target is unclear, a send/submit/payment/password/account action is visible, "
            "or the screenshot does not provide enough confidence. "
            "Use OCR list as authoritative for text positions; Vision is for semantic understanding. "
            "Do not include click actions; only report analysis. "
        )
        prompt += " OCR data JSON: " + json.dumps(ocr_data, ensure_ascii=False)[:12000]
        if question:
            prompt += f" User context: {question[:500]}"

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                gtypes.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
        )
        parsed = _extract_json_object(response.text or "")
        result.update({
            "active_app_guess": str(parsed.get("active_app_guess", ""))[:120],
            "visible_page_or_screen": str(parsed.get("visible_page_or_screen", ""))[:180],
            "visible_text_summary": str(parsed.get("visible_text_summary", ""))[:500],
            "possible_input_fields": list(parsed.get("possible_input_fields") or [])[:12],
            "possible_buttons": list(parsed.get("possible_buttons") or [])[:16],
            "risk_level": str(parsed.get("risk_level", "unknown")).lower()[:20],
            "can_continue_current_task": bool(parsed.get("can_continue_current_task", False)),
            "reason": str(parsed.get("reason", ""))[:300],
            "ocr_texts": list(parsed.get("ocr_texts") or ocr_data.get("texts") or [])[:150],
            "ocr_candidates": list(parsed.get("ocr_candidates") or ocr_data.get("candidates") or [])[:60],
        })
        _save_screen_analysis_to_state(result)
        return result
    except Exception as exc:
        if _is_quota_error(exc):
            result["visible_text_summary"] = "Gemini Vision şu an kullanılamıyor: quota/limit dolu."
            result["reason"] = "Gemini quota/limit dolu; güvenli title/process/session fallback kullanılmalı."
        else:
            result["visible_text_summary"] = "Screen analysis could not be completed."
            result["reason"] = str(exc)[:300]
        result["ocr_texts"] = []
        result["ocr_candidates"] = []
        _save_screen_analysis_to_state(result)
        return result


def analyze_screen_ocr_once(question: str = "", monitor_mode: str = "primary") -> dict:
    """
    OCR-first screen analysis:
    - extracts visible text
    - estimates text coordinates
    - guesses likely input/button regions
    No click/automation is performed here.
    """
    result = dict(_OCR_ANALYSIS_DEFAULT)

    try:
        monitor_mode = (monitor_mode or "primary").strip().lower()
        if monitor_mode not in {"primary", "all"}:
            monitor_mode = "primary"
        screen_image = _capture_ocr_screen_image(monitor_mode=monitor_mode)
        ocr_data = _extract_ocr_data(screen_image, monitor_mode=monitor_mode)
        found_texts = _safe_list(ocr_data.get("texts"), 120)
        possible_input_fields, possible_buttons, possible_titles = _guess_ocr_regions(found_texts)

        text = _sanitize_ocr_text(ocr_data.get("raw_text", ""))
        possible_input_fields, fallback_notes = _apply_raw_text_input_fallback(possible_input_fields, found_texts, text)
        print("OCR RAW:", text[:500])
        notes = _sanitize_ocr_text(" ".join([ocr_data.get("reason", ""), *fallback_notes]), limit=500)

        result.update({
            "found_texts": found_texts,
            "possible_input_fields": possible_input_fields,
            "possible_buttons": possible_buttons,
            "possible_titles": possible_titles,
            "notes": notes,
            "raw_text": text,
            "debug_screenshot_path": ocr_data.get("debug_screenshot_path", ""),
            "processed_screenshot_path": ocr_data.get("processed_screenshot_path", ""),
            "image_size": ocr_data.get("image_size", ""),
            "image_mode": ocr_data.get("image_mode", ""),
            "screen_size": ocr_data.get("screen_size", ""),
            "screenshot_size": ocr_data.get("screenshot_size", ""),
            "processed_size": ocr_data.get("processed_size", ""),
            "monitor_mode": ocr_data.get("monitor_mode", monitor_mode),
            "image_to_string_text": ocr_data.get("image_to_string_text", ""),
            "image_to_data_count": ocr_data.get("image_to_data_count", 0),
        })
        return result
    except Exception as exc:
        result["notes"] = _sanitize_ocr_text(str(exc), limit=300)
        return result


def _cv2_backend() -> int:
    """Return the best OpenCV camera backend for the current OS."""
    if not _CV2:
        return 0
    os_name = _get_os()
    if os_name == "windows":
        return cv2.CAP_DSHOW    
    if os_name == "mac":
        return cv2.CAP_AVFOUNDATION  
    return cv2.CAP_ANY


def _probe_camera(index: int, backend: int, warmup: int = 5) -> bool:

    if not _CV2:
        return False
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return False
    for _ in range(warmup):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False
    return bool(np.mean(frame) > 8)


def _detect_camera_index() -> int:

    backend = _cv2_backend()
    print("[Vision] 🔍 Auto-detecting camera...")
    for idx in range(6):
        if _probe_camera(idx, backend):
            print(f"[Vision] ✅ Camera found at index {idx}")
            _save_config_key("camera_index", idx)
            return idx
        print(f"[Vision] ⚠️  Camera index {idx}: no usable frame")

    print("[Vision] ⚠️  No camera found — defaulting to index 0")
    _save_config_key("camera_index", 0)
    return 0


def _get_camera_index() -> int:
    cfg = _load_config()
    if "camera_index" in cfg:
        return int(cfg["camera_index"])
    return _detect_camera_index()


def _capture_camera() -> tuple[bytes, str]:
    if not _CV2:
        raise RuntimeError("OpenCV (cv2) is not installed. Run: pip install opencv-python")

    index   = _get_camera_index()
    backend = _cv2_backend()
    cap     = cv2.VideoCapture(index, backend)

    if not cap.isOpened():
        raise RuntimeError(f"Camera index {index} could not be opened.")

    for _ in range(10):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Camera returned no frame.")

    if _PIL:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(rgb)
        img.thumbnail((_IMG_MAX_W, _IMG_MAX_H), PIL.Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_Q)
        return buf.getvalue(), "image/jpeg"

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])
    return buf.tobytes(), "image/jpeg"

class _VisionSession:
    def __init__(self):
        self._loop:       Optional[asyncio.AbstractEventLoop] = None
        self._thread:     Optional[threading.Thread]          = None
        self._session                                          = None
        self._out_queue:  Optional[asyncio.Queue]             = None
        self._audio_in:   Optional[asyncio.Queue]             = None
        self._ready_evt:  threading.Event                     = threading.Event()
        self._player                                           = None
        self._lock:       threading.Lock                       = threading.Lock()

    def start(self, player=None, timeout: float = 25.0) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                if player is not None:
                    self._player = player
                return
            self._player = player
            self._thread = threading.Thread(
                target=self._run_event_loop,
                daemon=True,
                name="VisionSessionThread",
            )
            self._thread.start()

        if not self._ready_evt.wait(timeout=timeout):
            raise RuntimeError(f"Vision session did not connect within {timeout}s.")
        print("[Vision] ✅ Session ready")

    def analyze(self, image_bytes: bytes, mime_type: str, user_text: str) -> None:
        if not self._loop or not self._out_queue:
            print("[Vision] ⚠️  Session not started — dropping request")
            return
        asyncio.run_coroutine_threadsafe(
            self._out_queue.put((image_bytes, mime_type, user_text)),
            self._loop,
        )

    def is_ready(self) -> bool:
        return self._session is not None

    def _run_event_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._session_loop())

    async def _session_loop(self) -> None:
        self._out_queue = asyncio.Queue(maxsize=30)
        self._audio_in  = asyncio.Queue()

        client = genai.Client(
            api_key=_get_api_key(),
            http_options={"api_version": "v1beta"},
        )
        config = gtypes.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription={},
            system_instruction=_SYSTEM_PROMPT,
            speech_config=gtypes.SpeechConfig(
                voice_config=gtypes.VoiceConfig(
                    prebuilt_voice_config=gtypes.PrebuiltVoiceConfig(
                        voice_name="Charon"
                    )
                )
            ),
        )

        backoff = 2.0
        while True:
            try:
                print("[Vision] 🔌 Connecting...")
                async with client.aio.live.connect(
                    model=_LIVE_MODEL, config=config
                ) as session:
                    self._session = session
                    self._ready_evt.set()
                    backoff = 2.0  
                    print("[Vision] ✅ Connected")

                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(self._send_loop())
                        tg.create_task(self._recv_loop())
                        tg.create_task(self._play_loop())

            except Exception as eg:
                for exc in eg.exceptions:
                    print(f"[Vision] ⚠️  Session error: {exc}")
            finally:
                self._session = None
                self._ready_evt.clear()

            print(f"[Vision] 🔄 Reconnecting in {backoff:.0f}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 30.0)
            self._ready_evt.set()  

    async def _send_loop(self) -> None:
        while True:
            image_bytes, mime_type, user_text = await self._out_queue.get()
            if not self._session:
                print("[Vision] ⚠️  No session — dropping image")
                continue
            try:
                b64 = base64.b64encode(image_bytes).decode("ascii")
                await self._session.send_client_content(
                    turns={
                        "parts": [
                            {"inline_data": {"mime_type": mime_type, "data": b64}},
                            {"text": user_text},
                        ]
                    },
                    turn_complete=True,
                )
                print(f"[Vision] 📤 Sent {len(image_bytes):,} bytes — '{user_text[:60]}'")
            except Exception as e:
                print(f"[Vision] ⚠️  Send error: {e}")

    async def _recv_loop(self) -> None:
        transcript: list[str] = []
        try:
            async for response in self._session.receive():
                if response.data:
                    await self._audio_in.put(response.data)

                sc = response.server_content
                if not sc:
                    continue

                if sc.output_transcription and sc.output_transcription.text:
                    chunk = sc.output_transcription.text.strip()
                    if chunk:
                        transcript.append(chunk)

                if sc.turn_complete:
                    if transcript and self._player:
                        full = re.sub(r"\s+", " ", " ".join(transcript)).strip()
                        if full:
                            self._player.write_log(f"Jarvis: {full}")
                            print(f"[Vision] 💬 {full}")
                    transcript = []

        except Exception as e:
            print(f"[Vision] ⚠️  Recv error: {e}")
            raise  

    async def _play_loop(self) -> None:
        stream = sd.RawOutputStream(
            samplerate=_RECEIVE_SAMPLE_RATE,
            channels=_CHANNELS,
            dtype="int16",
            blocksize=_CHUNK_SIZE,
        )
        stream.start()
        try:
            while True:
                chunk = await self._audio_in.get()
                await asyncio.to_thread(stream.write, chunk)
        except Exception as e:
            print(f"[Vision] ❌ Play error: {e}")
            raise
        finally:
            stream.stop()
            stream.close()

_session      = _VisionSession()
_session_lock = threading.Lock()
_session_up   = False


def _ensure_session(player=None) -> None:
    global _session_up
    with _session_lock:
        if not _session_up:
            _session.start(player=player)
            _session_up = True
        elif player is not None:
            _session._player = player


def screen_process(
    parameters:     dict,
    response=None,
    player=None,
    session_memory=None,
) -> bool:

    params    = parameters or {}
    user_text = (params.get("text") or params.get("user_text") or "").strip()
    angle     = params.get("angle", "screen").lower().strip()
    mode      = (params.get("mode") or "").strip().lower()
    monitor_mode = (params.get("monitor_mode") or "primary").strip().lower()

    if angle in {"ocr", "ocr_screen"} or mode == "ocr":
        ocr_report = analyze_screen_ocr_once(question=user_text, monitor_mode=monitor_mode)
        print("[Vision OCR] Analysis report:")
        print(json.dumps(ocr_report, ensure_ascii=False, indent=2))
        return True

    if get_model_type() != "gemini":
        print("[Vision] Local mode active; Gemini vision call skipped.")
        return False

    if not _load_config().get("gemini_api_key", "").strip():
        print("[Vision] Gemini API key is not configured; vision call skipped.")
        return False

    if not user_text:
        print("[Vision] ⚠️  No question provided — aborting")
        return False

    print(f"[Vision] ▶ angle={angle!r}  question='{user_text[:80]}'")

    try:
        _ensure_session(player=player)
    except Exception as e:
        print(f"[Vision] ❌ Could not start session: {e}")
        return False

    try:
        if angle == "camera":
            image_bytes, mime_type = _capture_camera()
            print(f"[Vision] 📷 Camera: {len(image_bytes):,} bytes")
        else:
            image_bytes, mime_type = _capture_screen()
            print(f"[Vision] 🖥️  Screen: {len(image_bytes):,} bytes")
    except Exception as e:
        print(f"[Vision] ❌ Capture error: {e}")
        return False

    _session.analyze(image_bytes, mime_type, user_text)
    return True


def warmup_session(player=None) -> None:
    try:
        _ensure_session(player=player)
    except Exception as e:
        print(f"[Vision] ⚠️  Warmup failed: {e}")

if __name__ == "__main__":
    print("[TEST] screen_processor.py")
    print("=" * 52)
    mode = input("angle — screen / camera (default: screen): ").strip().lower() or "screen"
    q    = input("Question (Enter = default): ").strip() or "What do you see? Be brief."

    t0 = time.perf_counter()
    warmup_session()
    print(f"Session ready in {time.perf_counter()-t0:.2f}s\n")

    t1 = time.perf_counter()
    ok = screen_process({"angle": mode, "text": q})
    print(f"Queued in {time.perf_counter()-t1:.3f}s — waiting for audio...")
    time.sleep(10)
    print("Done." if ok else "Failed.")
