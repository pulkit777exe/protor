import re
import os
import json
from datetime import datetime


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_default_output_dir() -> str:
    """Get the default output directory based on OS (Downloads/protor folder)"""
    if os.name == 'nt':  # Windows
        return os.path.join(os.environ['USERPROFILE'], 'Downloads', 'protor')
    else:  # Linux/macOS
        return os.path.join(os.path.expanduser('~'), 'Downloads', 'protor')
