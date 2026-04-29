#!/usr/bin/env python3
"""
Quick test to verify model provider integration
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import get_model_type, get_local_endpoint, get_local_model, get_gemini_key, get_config

print("=" * 60)
print("[INFO] JARVIS Configuration Check")
print("=" * 60)

try:
    model_type = get_model_type()
    print(f"[OK] Model Type: {model_type.upper()}")
    
    if model_type in ("local", "ollama"):
        endpoint = get_local_endpoint()
        model = get_local_model()
        print(f"[OK] Local Endpoint: {endpoint}")
        print(f"[OK] Local Model: {model}")
        print("\n[NOTE] To use local model server:")
        print("   1. Start an OpenAI-compatible local server")
        print("   2. Ensure the model is loaded on that server")
        print("   3. Start JARVIS!")
        
    elif model_type == "gemini":
        api_key = get_gemini_key()
        if api_key:
            print(f"[OK] Gemini API Key: {'*' * len(api_key)}")
        else:
            print("[WARN] Gemini API Key not set (required for Gemini mode)")
            print("   Get free key from: https://aistudio.google.com/app/apikey")
    else:
        endpoint = get_config().get("local_endpoint", "http://localhost:8000/v1")
        print(f"[WARN] Unknown model_type '{model_type}', fallback endpoint: {endpoint}")
            
except Exception as e:
    print(f"[ERR] Error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
