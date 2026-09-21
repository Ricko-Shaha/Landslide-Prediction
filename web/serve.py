"""Production entry point for Render and local deployment checks."""
import os
import sys
from pathlib import Path

# Limit numerical-library threads before importing the model on a small instance.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from waitress import serve
from web.app import app
from src.predict import load_bundle


if __name__ == "__main__":
    # Fail at startup if the committed model is missing or cannot be loaded.
    load_bundle()
    serve(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "10000")),
        threads=4,
        connection_limit=64,
        channel_timeout=120,
        max_request_body_size=65536,
    )
