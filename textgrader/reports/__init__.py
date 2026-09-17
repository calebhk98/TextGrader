"""Project reports: house policy, made runnable.

These are a different kind of thing from the metrics in
``textgrader/metrics/``.  A metric measures prose and reports a number in a
structured result; a report here checks a manuscript against an author's own
configured conventions and writes a table for a person to read, exiting 1 when
something it was configured to watch for is present.

They used to live in a top-level ``measures/`` directory of loose scripts, each
opening with a ``sys.path.insert`` so it could import the library sitting next
to it.  They are a package now, so ``grade.py`` runs them as
``python -m textgrader.reports.<name>`` and they import the library normally.

Every one of them is off by default and does nothing without
``project_measures`` in the configuration.  See
``examples/project_measures.example.json``.
"""

#: The targets a report template may name.  Every template names exactly one,
#: and ``tests/test_grade_contract.py`` enforces that.
#:
#: ``{manuscript}``
#:     the single file ``grade.py`` was pointed at.  For reports that measure a
#:     document as a whole, and for the two that take it by name because a bare
#:     positional means a character sheet to them.
#: ``{chapters_dir}``
#:     the configured chapters directory.  For reports that compare chapters
#:     with each other, which need more than one of them to have anything to
#:     compare.
#: ``{manuscript_dir}``
#:     the manuscript's parent directory.  Named here so it stays a recognised
#:     word rather than a typo, and used by nothing: handing it to
#:     ``dialogue_study`` as a corpus is the bug this vocabulary replaced.
TARGETS = ("{manuscript}", "{chapters_dir}", "{manuscript_dir}")

#: Report name -> the arguments ``grade.py`` passes it.
#:
#: Every report used to get ``{manuscript}``, which was over-correction. The
#: bug being fixed then was real - ``dialogue_study`` was handed the
#: manuscript's parent directory as a corpus, and ``prose_check`` and
#: ``verify_citations`` were handed the manuscript as if it were a character
#: sheet - but banning directories outright also cut off the reports that
#: compare chapters with each other. Those received the whole manuscript
#: concatenated, saw exactly one unit of text, and degenerated to a single row.
#:
#: ``register`` was the clearest casualty, because it judges each chapter
#: against the book's own median: with one row the median IS the row, nothing
#: can exceed it, and the report printed a pass in the same words it uses for a
#: real one. Run against the chapters directory, the chapters that actually
#: sit above the line are visible again.
#:
#: Nothing in the reports themselves needed changing. Each already resolves a
#: directory argument through ``resolve_paths`` and already falls back to
#: ``chapters_dir`` when handed nothing. Only the template blocked it.
REPORTS = {
    "absolutes": ("{chapters_dir}",),
    "banned_phrases": ("{chapters_dir}",),
    "check_edits": ("{chapters_dir}",),
    "dialogue_study": ("{manuscript}",),
    "number_report": ("{manuscript}",),
    "prose_check": ("--manuscript", "{manuscript}"),
    "quotable": ("{chapters_dir}",),
    "quote_length": ("{chapters_dir}",),
    "register": ("{chapters_dir}",),
    "style_report": ("{manuscript}",),
    "tics": ("{chapters_dir}",),
    "verify_citations": ("--manuscript", "{manuscript}"),
    "voice_separation": ("{chapters_dir}",),
}

#: Set by ``grade.py`` on every report it launches.  A report that does not see
#: it is being run by hand, and says so on stderr: these print tables for a
#: person to read, and ``grade.py`` is what turns one into a structured result.
VIA_GRADE_ENV_VAR = "TEXTGRADER_VIA_GRADE"


def solo_notice():
    """Say on stderr that this report was run outside ``grade.py``.

    One definition rather than the fourteen identical copies this used to
    have, each naming the environment variable in its own string literal.
    """

    import os
    import sys

    if os.environ.get(VIA_GRADE_ENV_VAR):
        return
    print("  [bundled diagnostic; use grade.py for the structured report]",
          file=sys.stderr)


__all__ = ["REPORTS", "TARGETS", "VIA_GRADE_ENV_VAR", "solo_notice"]
