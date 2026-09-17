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

#: Report name -> the arguments ``grade.py`` passes it.  ``prose_check`` and
#: ``verify_citations`` take a named manuscript because a bare positional means
#: a character sheet to both of them.
REPORTS = {
    "absolutes": ("{manuscript}",),
    "banned_phrases": ("{manuscript}",),
    "check_edits": ("{manuscript}",),
    "dialogue_study": ("{manuscript}",),
    "number_report": ("{manuscript}",),
    "prose_check": ("--manuscript", "{manuscript}"),
    "quotable": ("{manuscript}",),
    "quote_length": ("{manuscript}",),
    "register": ("{manuscript}",),
    "style_report": ("{manuscript}",),
    "tics": ("{manuscript}",),
    "verify_citations": ("--manuscript", "{manuscript}"),
    "voice_separation": ("{manuscript}",),
}

__all__ = ["REPORTS"]
