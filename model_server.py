"""
Legacy compatibility entrypoint.

The old version of this file contained a hardcoded personal Hugging Face cache
path. Keep this shim so older commands still work, but use local_model_server.py
as the only real implementation.

Run:
    python -m uvicorn local_model_server:app --host 127.0.0.1 --port 8000
"""

from local_model_server import app
