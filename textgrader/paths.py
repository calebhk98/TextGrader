"""Where the installed package keeps its own files.

Bundled reference data used to sit in the repository root, mixed in with the
entry-point scripts, and each consumer rebuilt the path with its own
``Path(__file__).resolve().parent.parent``.  One module owns those paths now,
so moving a file is one edit rather than a search.

Nothing here is user data.  A project's own manuscript, chapters, corpus and
profile are named in its configuration and resolved relative to it.
"""

from __future__ import annotations

from pathlib import Path

#: The installed ``textgrader`` package.
PACKAGE_DIR = Path(__file__).resolve().parent

#: The repository or installation root: entry-point scripts live here.
ROOT = PACKAGE_DIR.parent

#: Bundled reference data shipped with TextGrader.
DATA_DIR = ROOT / "data"

#: Worked example configurations. Never defaults.
EXAMPLES_DIR = ROOT / "examples"

#: The bundled 23-book core-metric reference, in the pre-2.0 layout.
PROSE_REFERENCE = DATA_DIR / "prose_reference.json"

#: The historical, Gutenberg-contaminated frequency table. Off unless the
#: configuration asks for it by name; see ``core_metrics.word_frequency``.
WORD_FREQUENCY = DATA_DIR / "word_frequency.json"

#: Reference rates for the absolutes report.
ABSOLUTES_REFERENCE = DATA_DIR / "absolutes_reference.json"


__all__ = ["PACKAGE_DIR", "ROOT", "DATA_DIR", "EXAMPLES_DIR",
           "PROSE_REFERENCE", "WORD_FREQUENCY", "ABSOLUTES_REFERENCE"]
