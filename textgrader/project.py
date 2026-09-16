"""Load project configuration without embedding any manuscript policy.

Nothing in this file, and nothing in the checked-in ``config.json``, describes a
particular book.  Cast lists, chapter exemptions, target bands and formatting
policies all live in the user's own configuration and default to "not set",
which every measure must treat as "report, do not judge".

The configuration path is resolved in this order, so that a ``--config`` passed
to ``grade.py`` reaches the legacy reports it runs as subprocesses rather than
silently leaving them on the repository's own ``config.json``:

1. an explicit argument to :func:`load_config`;
2. the ``TEXTGRADER_CONFIG`` environment variable;
3. ``config.json`` beside this file.
"""

import json
import os
from pathlib import Path

from .paths import ROOT

DEFAULT_CONFIG_PATH = ROOT / "config.json"

#: Set by ``grade.py`` before it launches a bundled report.
CONFIG_ENV_VAR = "TEXTGRADER_CONFIG"


def config_path(path=None):
    if path:
        return Path(path).resolve()
    from_environment = os.environ.get(CONFIG_ENV_VAR)
    return Path(from_environment).resolve() if from_environment else DEFAULT_CONFIG_PATH


def load_config(path=None):
    """Read one configuration file, recording the directory paths resolve from."""

    path = config_path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data["_config_dir"] = str(path.parent)
    data["_config_path"] = str(path)
    return data


CONFIG_PATH = config_path()
CONFIG = load_config(CONFIG_PATH)


def project_path(key, default=None, config=None):
    config = CONFIG if config is None else config
    value = config.get(key, default)
    return (Path(config["_config_dir"]) / value).resolve() if value else None


def measure_settings(name, config=None):
    """Options for one bundled report under ``project_measures``.

    An empty mapping is the documented default and means the report has no
    house policy to enforce: it describes what it finds and returns success.
    """

    config = CONFIG if config is None else config
    settings = config.get("project_measures", {}).get(name, {})
    return dict(settings) if isinstance(settings, dict) else {}


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
PROJECT_MEASURES = CONFIG.get("project_measures", {})
