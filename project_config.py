"""Load project configuration without embedding any manuscript policy."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


def load_config(path=CONFIG_PATH):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data["_config_dir"] = str(path.parent)
    return data


CONFIG = load_config()


def project_path(key, default=None):
    value = CONFIG.get(key, default)
    return (Path(CONFIG["_config_dir"]) / value).resolve() if value else None


MANUSCRIPT = project_path("manuscript", "MANUSCRIPT.md")
CHAPTERS_DIR = project_path("chapters_dir", "chapters")
CHARACTERS_DIR = project_path("characters_dir", "characters")
CORPUS_DIRS = tuple((Path(CONFIG["_config_dir"]) / value).resolve()
                    for value in CONFIG.get("corpus_dirs", []))
CORPUS_PROFILE = project_path("corpus_profile")
PROJECT_RULES = CONFIG.get("project_rules", {})
# Compatibility exports for individual legacy commands. Empty mappings mean
# no house preference; commands must treat their absence as unavailable.
READING_TARGETS = CONFIG.get("reading_targets", {})
DIALOGUE_TARGETS = CONFIG.get("dialogue_targets", {})
BANNED_CONSTRUCTIONS = PROJECT_RULES.get("banned_phrases", [])
