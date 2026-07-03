"""Tiny .env loader (no python-dotenv dependency).

Loads KEY=VALUE lines into os.environ without overriding variables that are
already set, so a real environment variable always wins over the file.
Used for OPENROUTER_API_KEY — see .env.example.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> bool:
    """Load `path` into os.environ (setdefault semantics). Returns True if
    the file existed. Supports comments, blank lines, `export KEY=...`, and
    single/double-quoted values."""
    p = Path(path)
    if not p.exists():
        return False
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)
    return True
