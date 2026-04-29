import time
import uuid
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    DEFAULT_LOCAL_MODEL,
    get_local_chat_endpoint,
    get_local_model,
    get_model_path,
)


app = FastAPI(title="Jarvis Local OpenAI-Compatible Server")

_tokenizer = None
_model = None
_model_name = DEFAULT_LOCAL_MODEL
_model_source = ""
_device = "not-loaded"
_last_error = ""
_loaded_at = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_LOCAL_MODEL
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.4


def _cuda_info() -> dict:
    available = torch.cuda.is_available()
    info = {"available": available, "device_count": torch.cuda.device_count() if available else 0}
    if available:
        try:
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            info.update({
                "current_device": index,
                "name": props.name,
                "total_vram_gb": round(props.total_memory / (1024 ** 3), 2),
            })
        except Exception as exc:
            info["error"] = str(exc)
    return info


def _resolve_model_source() -> tuple[str, str]:
    model_name = get_local_model() or DEFAULT_LOCAL_MODEL
    configured_path = get_model_path()
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Configured model_path does not exist: {path}")
        return model_name, str(path)
    return model_name, model_name


def _load_model_once():
    global _tokenizer, _model, _model_name, _model_source, _device, _last_error, _loaded_at

    if _tokenizer is not None and _model is not None:
        return

    _model_name, _model_source = _resolve_model_source()
    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32

    try:
        _last_error = ""
        _tokenizer = AutoTokenizer.from_pretrained(
            _model_source,
            trust_remote_code=True,
            local_files_only=True,
        )
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "local_files_only": True,
        }
        if cuda:
            load_kwargs["device_map"] = "auto"

        _model = AutoModelForCausalLM.from_pretrained(_model_source, **load_kwargs)
        if not cuda:
            _model.to("cpu")
        _model.eval()

        try:
            first_param = next(_model.parameters())
            _device = str(first_param.device)
        except Exception:
            _device = "cuda" if cuda else "cpu"
        _loaded_at = int(time.time())

    except Exception as exc:
        _tokenizer = None
        _model = None
        _last_error = (
            f"{type(exc).__name__}: {exc}. "
            "The model must already exist locally; this server does not download models."
        )
        raise RuntimeError(_last_error) from exc


def _build_prompt(messages: List[ChatMessage]) -> str:
    if _tokenizer and hasattr(_tokenizer, "apply_chat_template"):
        chat = [{"role": m.role, "content": m.content} for m in messages]
        return _tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    lines = [f"{m.role}: {m.content}" for m in messages]
    lines.append("assistant:")
    return "\n".join(lines)


def _health_payload() -> dict:
    return {
        "status": "ok",
        "server_running": True,
        "model_loaded": _model is not None,
        "model_name": get_local_model() or DEFAULT_LOCAL_MODEL,
        "model_source": _model_source or get_model_path() or get_local_model() or DEFAULT_LOCAL_MODEL,
        "device": _device,
        "cuda": _cuda_info(),
        "endpoint": get_local_chat_endpoint(),
        "loaded_at": _loaded_at,
        "error": _last_error,
    }


@app.get("/health")
def health():
    return _health_payload()


@app.get("/v1/models")
def list_models():
    model_name = get_local_model() or DEFAULT_LOCAL_MODEL
    return {
        "object": "list",
        "data": [
            {"id": model_name, "object": "model", "created": int(time.time()), "owned_by": "local"}
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    try:
        _load_model_once()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    prompt = _build_prompt(req.messages)
    target_device = next(_model.parameters()).device
    inputs = _tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(target_device) for key, value in inputs.items()}

    max_new_tokens = max(16, min(int(req.max_tokens or 128), 512))
    temperature = req.temperature if req.temperature is not None else 0.4
    do_sample = temperature > 0

    try:
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": _tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
        with torch.inference_mode():
            output_ids = _model.generate(**gen_kwargs)
    except RuntimeError as exc:
        detail = str(exc)
        if "out of memory" in detail.lower():
            detail = "CUDA/RAM memory is not enough for this model. Close other apps or use a smaller/quantized model."
        raise HTTPException(status_code=500, detail=detail) from exc

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    content = _tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": get_local_model() or DEFAULT_LOCAL_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(generated_ids.shape[0]),
            "total_tokens": int(inputs["input_ids"].shape[1] + generated_ids.shape[0]),
        },
    }
