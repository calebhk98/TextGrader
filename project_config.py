"""Shared project paths loaded from ``config.json``.

All relative paths in the configuration are resolved from the repository root,
not from the caller's current working directory.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


def _load():
    if not CONFIG_PATH.is_file():
        raise RuntimeError(f"missing project configuration: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


CONFIG = _load()


def project_path(key):
    """Return one configured project path, resolved from the repository root."""
    return (ROOT / CONFIG[key]).resolve()


MANUSCRIPT = project_path("manuscript")
CHAPTERS_DIR = project_path("chapters_dir")
CHARACTERS_DIR = project_path("characters_dir")
CORPUS_DIRS = tuple((ROOT / path).resolve() for path in CONFIG["corpus_dirs"])
READING_TARGETS = CONFIG["reading_targets"]
DIALOGUE_TARGETS = CONFIG["dialogue_targets"]
BANNED_CONSTRUCTIONS = CONFIG.get("banned_constructions")
