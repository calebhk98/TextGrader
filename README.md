# TextGrader

TextGrader builds a Markdown manuscript and reports readability, style,
dialogue, repeated constructions, number usage, and optional character-sheet
checks.

## Requirements

- Python 3.8 or newer
- No third-party packages: there is no `pip install` step and no
  `requirements.txt`

## Quick start

**Put chapters in `chapters/` → run `build_manuscript.py` → run `grade.py`.**

1. Add chapters as `chapters/01_title.md`, `chapters/02_title.md`, and so on.
   Files sort by name and must be numbered consecutively.
2. Start each file with the matching spelled-out heading, for example:

   ```markdown
   ## Chapter One: A Title
   ```

3. Build the combined manuscript:

   ```console
   python3 build_manuscript.py
   ```

   This generates `MANUSCRIPT.md`. It is a build artifact and does not need to
   be distributed with the project.
4. Run the complete report:

   ```console
   python3 grade.py
   ```

Use `python3 grade.py --one chapters/01_title.md` for a single chapter, or
`python3 grade.py --help` for the other report modes.

## Optional inputs and configuration

- `characters/` may contain character sheets. If it is empty, character-sheet
  checks simply report that they checked zero files.
- `sample_corpus/` may contain plain-text (`.txt`) reference books. It is
  optional: saved benchmark JSON remains available to measures that have it;
  measures without saved number/construction profiles clearly report that
  corpus comparisons are unavailable.
- `config.json` controls the manuscript, chapter, character, and corpus paths,
  reading-level bands and targets, dialogue targets, and banned constructions.
  Relative paths are resolved from the repository root. Leave
  `banned_constructions` as `null` for the built-in rules, or replace it with a
  list of objects containing `pattern`, `name`, and optional `reason` fields.
