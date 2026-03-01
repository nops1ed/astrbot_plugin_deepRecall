#!/usr/bin/env python3
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("deepRecall Web UI")
print("=" * 60)
print("\nInstalling dependencies (if needed)...")

try:
    from webui.webui import app
except ImportError:
    print("\nInstalling fastapi and uvicorn...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn"])

from webui.webui import app
import uvicorn

print("\n" + "=" * 60)
print("Starting deepRecall Web UI...")
print("Open http://0.0.0.0:8000 in your browser")
print("=" * 60 + "\n")

uvicorn.run(app, host="0.0.0.0", port=8000)
