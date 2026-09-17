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


#: Recognised top-level keys.  Anything beginning with an underscore is skipped
#: wherever keys are checked: JSON has no comments, so ``"_comment"`` is how
#: every example config in this repository documents itself, and ``load_config``
#: injects ``_config_dir`` and ``_config_path`` by the same convention.
TOP_LEVEL_KEYS = frozenset({
    "manuscript", "chapters_dir", "characters_dir",
    "corpus_profile", "corpus_dirs", "corpus_builder",
    "analysis", "text_processing", "nlp", "regex",
    "metrics", "project_rules", "project_measures",
    "metric_commands", "allow_external_metric_commands",
})


def _named(keys):
    """The keys a check applies to: everything not spelled as a comment."""

    return {key for key in keys if not str(key).startswith("_")}

#: Recognised keys inside each nested section that has a fixed shape.
#: ``metrics``, ``project_measures`` and ``corpus_builder`` are not here:
#: ``metrics`` is checked against the live registry by ``grade.py``, which owns
#: it, ``project_measures`` is keyed by report name, and ``corpus_builder``
#: validates itself in ``corpus_builder/config.py``.
SECTION_KEYS = {
    "analysis": frozenset({"comparison_unit", "min_sentences_for_corpus",
                           "min_words_for_corpus", "lexile_frequency_source",
                           "benchmark"}),
    "text_processing": frozenset({"strip_gutenberg", "strip_markdown_headings",
                                  "strip_transcript", "drop_marker_paragraphs",
                                  "normalize_quotes", "segmenter", "language",
                                  "segmenter_clean", "transcript"}),
    "nlp": frozenset({"model", "disable", "max_chars_per_chunk", "max_words"}),
    "regex": frozenset({"engine", "timeout_seconds"}),
    "project_rules": frozenset({"banned_phrases", "em_dash", "quote_style",
                                "hard_line_breaks", "chapter_length"}),
}

#: Keys the shipped ``config.json`` advertises that nothing reads yet.  They
#: are recognised, so they are not typos, but a user who sets one is waiting
#: for an effect that never arrives.  Saying so is the whole point: silence
#: here is indistinguishable from a rule that ran and found nothing.
UNIMPLEMENTED_KEYS = {
    ("project_rules", "hard_line_breaks"),
    ("project_rules", "chapter_length"),
}


def near_miss(name, known):
    """The closest recognised key, when it is close enough to be a typo.

    Public because ``grade.py`` checks ``metrics`` and ``project_measures``
    against the live registry, which is the one key set this module cannot see.

    A small hand-rolled edit distance rather than difflib, because the
    threshold wants to be explicit: one edit in a short key, two in a long
    one. Beyond that a suggestion is a guess, and a wrong guess sends someone
    to rename a key that was never the problem.
    """

    best, best_distance = None, None
    for candidate in known:
        previous = list(range(len(candidate) + 1))
        for index, left in enumerate(name, 1):
            current = [index]
            for position, right in enumerate(candidate, 1):
                current.append(min(previous[position] + 1, current[position - 1] + 1,
                                   previous[position - 1] + (left != right)))
            previous = current
        distance = previous[-1]
        if best_distance is None or distance < best_distance:
            best, best_distance = candidate, distance
    limit = 1 if len(name) <= 6 else 2
    return best if best_distance is not None and best_distance <= limit else None


def config_issues(data):
    """Every recognisable problem with a configuration's shape.

    Returns a list of ``{"key", "kind", "message"}`` and raises nothing. A
    misspelt key used to be read, stored and never looked at again: no
    warning, no note in the report, no non-zero exit. Adding
    ``"totally_made_up_key": "banana"`` to a working config produced
    byte-identical output to leaving it out.

    That matters more here than it would in most tools, because configuration
    IS the extension mechanism. Every project-specific behaviour lives in
    ``project_measures`` and ``project_rules``, the reports are inert without
    it, and ``examples/project_measures.example.json`` is a large file people
    copy and adapt. A misspelt key there is a report that quietly does
    nothing, and nothing distinguished that from a report that ran and found
    nothing.
    """

    issues = []
    if not isinstance(data, dict):
        return issues
    for key in sorted(_named(data) - TOP_LEVEL_KEYS):
        suggestion = near_miss(key, TOP_LEVEL_KEYS)
        issues.append({
            "key": key, "kind": "unknown",
            "message": f"unrecognised configuration key {key!r}; it is read and never used"
                       + (f". Did you mean {suggestion!r}?" if suggestion else ""),
        })
    for section, known in SECTION_KEYS.items():
        values = data.get(section)
        if not isinstance(values, dict):
            continue
        for key in sorted(_named(values) - known):
            suggestion = near_miss(key, known)
            issues.append({
                "key": f"{section}.{key}", "kind": "unknown",
                "message": f"unrecognised key {key!r} under {section!r}; "
                           "it is read and never used"
                           + (f". Did you mean {suggestion!r}?" if suggestion else ""),
            })
        for key in sorted(known):
            if (section, key) in UNIMPLEMENTED_KEYS and values.get(key) is not None:
                issues.append({
                    "key": f"{section}.{key}", "kind": "unimplemented",
                    "message": f"{section}.{key} is recognised but nothing implements it "
                               "yet, so setting it has no effect",
                })
    return issues


def load_config(path=None):
    """Read one configuration file, recording the directory paths resolve from."""

    path = config_path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    # Recorded, not raised and not printed. A bad key is a reported limitation
    # like any other in this tool, so the caller that owns the output decides
    # how to show it; ``grade.py`` turns each one into a result.
    issues = config_issues(data)
    data["_config_dir"] = str(path.parent)
    data["_config_path"] = str(path)
    data["_config_issues"] = issues
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
