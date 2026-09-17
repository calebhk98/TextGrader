#!/usr/bin/env python3
"""Analyze English prose and emit structured evidence.

Every measurement in one run describes ONE document.  The text is cleaned,
tokenized, segmented into sentences and paragraphs, and split into dialogue and
narration exactly once, in :mod:`textgrader.document`; the core metrics, the
optional metrics and the corpus builder all read that same object.  Before this
existed, core analysis stripped Gutenberg boilerplate and Markdown headings
while the optional metrics received the raw file, so two numbers in one report
could describe two different texts.

Corpus outliers are evidence, not failures.  Project rules are the author's own
choices and are the only thing reported as a violation.  ``--json`` is the
stable machine-readable representation; the ``summary.top_findings`` list is
the short, severity-ordered list an authoring agent should read first.
"""

import argparse
import importlib
import json
import os
import subprocess
import statistics
import sys
import time
from pathlib import Path

from textgrader.project import (CONFIG_ENV_VAR, near_miss, config_path,
                                load_config, project_path)
from textgrader.document import (COMPARISON_UNITS, DocumentAnalysis, NlpSettings,
                                 TextProcessing, resolve_unit, units_comparable)
from textgrader.core_metrics import measure as core_measure
from textgrader.metrics import REGISTRY
from textgrader.reports import REPORTS
from textgrader.results import Action, MetricResult, Polarity, Report, StatusType
from textgrader.rules import compile_rules
from textgrader import stats

ROOT = Path(__file__).resolve().parent

#: Bundled project reports, and the arguments each is given.  Defined in
#: ``textgrader/reports/__init__.py`` so the package that owns them owns their
#: contract.  ``prose_check`` and ``verify_citations`` used to be handed the
#: manuscript as a bare positional, which those two programs read as a
#: character-sheet file, and ``dialogue_study`` was handed the manuscript's
#: PARENT DIRECTORY as if it were a corpus.
BUNDLED_MEASURES = REPORTS

#: Polarity per core metric: does a HIGH value mean more developed prose.
#: This is not the ``direction`` field, which only says whether the document
#: sits above or below the corpus centre. Four of these read inverted against
#: a tool that orients them - top100, u10, shortruns and simple - and that is
#: not a disagreement about the measurement, only about whether the number was
#: turned the right way up before being averaged with the others.
HIGHER, LOWER, NEUTRAL = Polarity.HIGHER, Polarity.LOWER, Polarity.NEUTRAL

#: metric key -> (name, unit, family, polarity).
#:
#: NEUTRAL is a decision, not a default. More or fewer fronted subordinate
#: clauses is a style choice rather than a competence, so front, and2, andrate
#: and negative are measured and reported and kept OUT of the aggregate rather
#: than assigned an arbitrary direction. The size counts are neutral for the
#: same reason with less room for argument: a longer book is not a better one.
METRIC_NAMES = {
    "fk": ("Flesch-Kincaid grade", "grade", "readability", HIGHER),
    "ari": ("Automated Readability Index", "grade", "readability", HIGHER),
    "lexile": ("Approximate Lexile", "L", "readability", HIGHER),
    "wps": ("Words per sentence", "words/sentence", "sentence_rhythm", HIGHER),
    "slcv": ("Sentence-length variation", "%", "sentence_rhythm", HIGHER),
    "wpp": ("Words per paragraph", "words/paragraph", "paragraph_rhythm", HIGHER),
    "spp": ("Sentences per paragraph", "sentences/paragraph", "paragraph_rhythm", HIGHER),
    "wlen": ("Mean word length", "characters", "lexical", HIGHER),
    "long7": ("Words of 7+ characters", "%", "lexical", HIGHER),
    "sttr": ("Standardized type-token ratio", "%", "lexical", HIGHER),
    "top100": ("Commonest-100 word share", "%", "lexical", LOWER),
    "commas": ("Commas per sentence", "commas/sentence", "punctuation", HIGHER),
    "subord": ("Subordinator-cue sentence share (lexical proxy)", "%", "syntax", HIGHER),
    "relcl": ("Relative-word sentence share (lexical proxy)", "%", "syntax", HIGHER),
    "simple": ("No-clause-cue sentence share (lexical proxy)", "%", "syntax", LOWER),
    "u10": ("Sentences under 10 words", "%", "sentence_rhythm", LOWER),
    "b2035": ("Sentences 20-35 words", "%", "sentence_rhythm", HIGHER),
    "shortruns": ("Sentences in short runs", "%", "sentence_rhythm", LOWER),
    "front": ("Front-loaded cue proxy", "%", "syntax", NEUTRAL),
    "and2": ('Sentences with two or more "and" tokens', "%", "syntax", NEUTRAL),
    "andrate": ('"and" share', "%", "lexical", NEUTRAL),
    "negative": ("Negative-cue sentence share", "%", "discourse", NEUTRAL),
    "_words": ("Word count", "words", "size", NEUTRAL),
    "_sentences": ("Sentence count", "sentences", "size", NEUTRAL),
    "_paragraphs": ("Paragraph count", "paragraphs", "size", NEUTRAL),
    "_transcript": ("Transcript word share", "%", "size", NEUTRAL),
}


# --------------------------------------------------------------- corpus access

def distribution(profile, key):
    """Corpus observations for one metric id, from either profile layout."""

    if not profile:
        return []
    if "distributions" in profile:
        found = profile["distributions"].get(key, [])
        if isinstance(found, dict):
            found = found.get("values", [])
        return [value for value in found if value is not None]
    books = profile.get("books", profile)
    if isinstance(books, dict):
        return [row[key] for row in books.values()
                if isinstance(row, dict) and row.get(key) is not None]
    if isinstance(books, list):
        return [row[key] for row in books if isinstance(row, dict) and row.get(key) is not None]
    return []


def profile_unit(profile):
    if not profile:
        return "unknown"
    unit = profile.get("comparison_unit") or profile.get("metadata", {}).get("comparison_unit")
    return unit if unit in COMPARISON_UNITS else "unknown"


class Comparator:
    """Holds one measurement against the corpus, with every gate applied.

    Three separate things can make a comparison unsafe, and each is reported
    rather than silently absorbed: too little manuscript, too little corpus, and
    a unit mismatch between a chapter and a shelf of novels.
    """

    def __init__(self, profile, document, settings):
        self.profile = profile
        self.document = document
        self.corpus_unit = profile_unit(profile)
        self.min_sentences = settings.get("min_sentences_for_corpus", 40)
        self.min_words = settings.get("min_words_for_corpus", 500)
        self.min_corpus = settings.get("min_corpus_sample", stats.MIN_CORPUS_SAMPLE)
        counts = [book.get("word_count") for book in (profile or {}).get("books", [])
                  if isinstance(book, dict) and book.get("word_count")]
        self.typical_words = int(statistics.median(counts)) if counts else None

    #: A sample-size-dependent metric compares only against observations of a
    #: similar size.  Threefold is generous; entropy drifts measurably well
    #: inside it, and anything tighter would refuse ordinary chapter variation.
    SIZE_RATIO = 3.0

    def size_mismatch(self, sample_size):
        """Why a sample-size-dependent value cannot be held against this corpus."""

        typical = self.typical_words
        mine = self.document.word_count
        if not typical or not mine:
            return None
        ratio = max(typical / mine, mine / typical)
        if ratio <= self.SIZE_RATIO:
            return None
        return (f"this value depends on how much text produced it, and this document is "
                f"{mine:,} words against a corpus whose texts are typically {typical:,}; "
                f"comparison withheld. Build a profile from texts of a similar size, or "
                f"read the value on its own")

    @property
    def document_is_large_enough(self):
        return (self.document.sentence_count >= self.min_sentences
                and self.document.word_count >= self.min_words)

    def document_note(self):
        return (f"document has {self.document.sentence_count} sentences and "
                f"{self.document.word_count} words, below the "
                f"{self.min_sentences}/{self.min_words} needed to place it in a corpus "
                f"distribution; the measurement is reported without a comparison")

    def apply(self, result, key, value, sample_size=None, min_sample=None):
        """Attach corpus statistics to ``result``, or explain their absence."""

        reference = distribution(self.profile, key)
        result.comparison_unit = self.document.comparison_unit
        if value is None:
            result.action = Action.UNAVAILABLE if result.error or result.warning else result.action
            return result
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            # Some findings are labels rather than quantities ("which feature
            # drifts most"). They are reportable and not comparable.
            return result
        if sample_size is not None and min_sample and sample_size < min_sample:
            result.action = Action.INSUFFICIENT_DATA
            result.warning = _join(result.warning,
                                   f"measured from {sample_size} units, below the {min_sample} "
                                   f"this metric needs to be read as a rate")
            return result
        if not reference:
            return result
        if not self.document_is_large_enough:
            result.action = Action.INSUFFICIENT_DATA
            result.warning = _join(result.warning, self.document_note())
            return result
        if not units_comparable(key, self.document.comparison_unit, self.corpus_unit):
            result.warning = _join(
                result.warning,
                f"{key} scales with document length: this profile describes "
                f"'{resolve_unit(self.corpus_unit)}' units and the input is "
                f"'{resolve_unit(self.document.comparison_unit)}', so the comparison is "
                f"withheld. Rebuild the profile from the same kind of unit, or compare a "
                f"rate instead of a count")
            result.action = Action.INSUFFICIENT_DATA
            return result
        comparison = stats.compare(value, reference, min_corpus=self.min_corpus)
        result.corpus = comparison.to_dict()
        result.corpus["profile_name"] = (self.profile.get("corpus_name")
                                         or self.profile.get("metadata", {}).get("corpus_name"))
        result.corpus["corpus_unit"] = self.corpus_unit
        result.direction = comparison.direction
        result.severity = comparison.severity
        result.confidence = comparison.confidence
        if comparison.outlier is None:
            result.status = "uncompared"
            result.status_type = StatusType.INFORMATIONAL
            result.action = Action.INSUFFICIENT_DATA
            result.warning = _join(result.warning, "; ".join(comparison.notes) or None)
        elif comparison.outlier:
            result.status, result.status_type = "outlier", StatusType.CORPUS_OUTLIER
            result.action = Action.REVIEW
        else:
            result.status, result.status_type = "inlier", StatusType.CORPUS_INLIER
            result.action = Action.INFORMATIONAL
        return result


def _join(existing, addition):
    if not addition:
        return existing
    return f"{existing}; {addition}" if existing else addition


# --------------------------------------------------------------- metric running

def metric_enabled(metric_config, name):
    setting = metric_config.get(name)
    if setting is None:
        return False
    return setting is True or (isinstance(setting, dict) and bool(setting.get("enabled", False)))


def metric_options(metric_config, name):
    setting = metric_config.get(name, {})
    options = dict(setting) if isinstance(setting, dict) else {}
    options.pop("enabled", None)
    spec = REGISTRY.get(name)
    if spec:
        merged = dict(spec.defaults)
        merged.update(options)
        return merged
    return options


def options_match_profile(profile, name, options):
    """Whether the corpus was built with the same options as this run.

    Comparing a MATTR computed over a 100-word window with a corpus built on a
    50-word window is comparing two different measurements, so it is refused.
    """

    if not profile or "metric_settings" not in profile:
        return False
    expected = dict(profile["metric_settings"].get(name, {}))
    expected.pop("enabled", None)
    spec = REGISTRY.get(name)
    if spec:
        base = dict(spec.defaults)
        base.update(expected)
        expected = base
    return expected == dict(options)


def run_metric_process(metric_id, command):
    """Run an optional external metric without losing crashes or findings."""

    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return [MetricResult(metric_id, metric_id, status="error",
                             status_type=StatusType.INTERNAL_ERROR, error=str(exc))]
    if completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip() or "no error output"
        return [MetricResult(metric_id, metric_id, status="error",
                             status_type=StatusType.INTERNAL_ERROR,
                             error=f"process exited {completed.returncode}: {message}")]
    try:
        payload = json.loads(completed.stdout)
        payload = payload if isinstance(payload, list) else [payload]
        return [MetricResult(**item) for item in payload]
    except (ValueError, TypeError) as exc:
        return [MetricResult(metric_id, metric_id, status="error",
                             status_type=StatusType.INTERNAL_ERROR,
                             error=f"invalid metric JSON: {exc}")]


def measure_arguments(name, manuscript, config):
    """Resolve one report's argument template against this run.

    Kept here, and used by the tests too, because the templates are a closed
    vocabulary: a second copy of this formatting is a second place that has to
    learn each new target word.
    """

    manuscript = Path(manuscript)
    # A report asking for the chapters directory gets it only if it is really
    # there. Falling back to the manuscript keeps a project that keeps no
    # chapters directory working, at the single-unit output those reports
    # degenerate to - which is what every report got before, so this is no
    # worse for that project and correct for the ones that do have chapters.
    try:
        chapters_dir = project_path("chapters_dir", "chapters", config)
    except (KeyError, TypeError, OSError):
        chapters_dir = None
    if not (chapters_dir and chapters_dir.is_dir() and any(chapters_dir.glob("*.md"))):
        chapters_dir = manuscript
    return [part.format(manuscript=str(manuscript),
                        chapters_dir=str(chapters_dir),
                        manuscript_dir=str(manuscript.parent))
            for part in BUNDLED_MEASURES[name]]


def run_bundled_measure(name, manuscript, config, timeout=120):
    """Run a bundled report as a structured metric, never via a shell.

    The report is launched with ``TEXTGRADER_CONFIG`` pointing at the same
    configuration this run loaded.  Without that, a ``--config other.json`` only
    reached the new metrics while every bundled report silently kept using the
    repository's own ``config.json``.
    """

    command = [sys.executable, "-m", f"textgrader.reports.{name}",
               *measure_arguments(name, manuscript, config)]
    environment = {**os.environ, "HALSTEAD_VIA_GRADE": "1",
                   CONFIG_ENV_VAR: str(config.get("_config_path", config_path())),
                   # ``-m`` resolves the package from the working directory, and
                   # the working directory is the project's, not ours.
                   "PYTHONPATH": os.pathsep.join(
                       [str(ROOT), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    try:
        completed = subprocess.run(command, cwd=config.get("_config_dir", ROOT),
                                   env=environment, capture_output=True, text=True,
                                   check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return MetricResult(f"measure.{name}", name.replace("_", " ").title(), status="error",
                            status_type=StatusType.INTERNAL_ERROR, error=str(exc),
                            family="project_report")
    output = completed.stdout.strip()
    error = completed.stderr.strip()
    crashed = "Traceback (most recent call last)" in error
    if completed.returncode not in (0, 1) or crashed:
        return MetricResult(f"measure.{name}", name.replace("_", " ").title(), status="error",
                            status_type=StatusType.INTERNAL_ERROR,
                            error=error or output or f"process exited {completed.returncode}",
                            family="project_report")
    return MetricResult(f"measure.{name}", name.replace("_", " ").title(),
                        status="review" if completed.returncode else "available",
                        status_type=(StatusType.DIAGNOSTIC if completed.returncode
                                     else StatusType.INFORMATIONAL),
                        details=[{"output": output}] if output else [],
                        warning=error or None, family="project_report")


def project_rules(analysis, config):
    """Configured house rules, compiled and bounded before anything is scanned."""

    results = []
    rules = config.get("project_rules", {}) or {}
    for rule in compile_rules(rules, config.get("regex", {})):
        if rule.error and rule.pattern is None:
            results.append(MetricResult(f"rule.{rule.rule_id}", rule.name, status="unavailable",
                                        status_type=StatusType.UNAVAILABLE,
                                        family="project_rule", error=rule.error))
            continue
        hits = rule.finditer(analysis.raw)
        if hits:
            results.append(MetricResult(
                f"rule.{rule.rule_id}", rule.name, len(hits), "hits", "violation",
                StatusType.PROJECT_RULE, sample_size=len(hits), details=hits[:50],
                evidence=hits[:20], family="project_rule", warning=rule.error))
        elif rule.error:
            results.append(MetricResult(f"rule.{rule.rule_id}", rule.name, 0, "hits",
                                        family="project_rule", warning=rule.error))
    if rules.get("em_dash") == "forbid":
        hits = [{"offset": index} for index, char in enumerate(analysis.raw) if char == "—"]
        if hits:
            results.append(MetricResult("rule.em_dash", "Em-dash policy", len(hits), "hits",
                                        "violation", StatusType.PROJECT_RULE,
                                        details=hits[:50], evidence=hits[:20],
                                        sample_size=len(hits), family="project_rule"))
    style = rules.get("quote_style")
    if style in ("straight", "curly"):
        banned = "“”" if style == "straight" else '"'
        hits = [{"offset": index, "match": char}
                for index, char in enumerate(analysis.raw) if char in banned]
        if hits:
            results.append(MetricResult("rule.quote_style", f"Quote style ({style})", len(hits),
                                        "hits", "violation", StatusType.PROJECT_RULE,
                                        details=hits[:50], evidence=hits[:20],
                                        sample_size=len(hits), family="project_rule"))
    return results


# --------------------------------------------------------------------- analysis

def load_profile(config, report):
    path = config.get("corpus_profile")
    if not path:
        return None
    candidate = Path(config.get("_config_dir", ROOT)) / path
    if not candidate.is_file():
        return None
    try:
        profile = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        report.results.append(MetricResult("corpus.profile", "Corpus profile",
                                           status="unavailable",
                                           status_type=StatusType.UNAVAILABLE,
                                           warning=str(exc)))
        return None
    header = profile.get("metadata") or {
        key: profile.get(key) for key in
        ("schema_version", "corpus_name", "build_timestamp", "parser_version",
         "metric_definition_version", "comparison_unit", "text_processing")
    }
    if not header.get("corpus_name"):
        header["corpus_name"] = candidate.stem
    header.setdefault("book_count", profile.get("book_count", len(profile.get("books", profile))))
    header["comparison_unit"] = profile_unit(profile)
    report.corpus_profile = header
    return profile


def check_preprocessing(profile, analysis, report):
    """Warn when the corpus was built from differently-prepared text."""

    if not profile:
        return
    recorded = profile.get("text_processing")
    if recorded is None:
        report.results.append(MetricResult(
            "corpus.text_processing", "Corpus preprocessing", status="unavailable",
            status_type=StatusType.UNAVAILABLE,
            warning="this profile predates recorded text_processing settings; rebuild it to "
                    "guarantee the corpus and the manuscript were prepared the same way"))
        return
    if dict(recorded) != analysis.processing.fingerprint():
        differing = sorted(key for key in set(recorded) | set(analysis.processing.fingerprint())
                           if recorded.get(key) != analysis.processing.fingerprint().get(key))
        report.results.append(MetricResult(
            "corpus.text_processing", "Corpus preprocessing", status="unavailable",
            status_type=StatusType.UNAVAILABLE,
            warning=f"corpus and manuscript were prepared differently ({', '.join(differing)}); "
                    f"comparisons may not describe the same kind of text"))


def analyze_text(text, config, source="<text>"):
    """Analyze a string.  Used where a caller already holds the text, such as
    one chapter sliced out of a manuscript, so nothing has to reach disk."""

    return _analyze(config, Report(source=source), text=text)


def analyze(path, config):
    path = Path(path)
    report = Report(source=str(path))
    if not path.is_file():
        report.results.append(MetricResult("input.manuscript", "Manuscript input",
                                           status="unavailable",
                                           status_type=StatusType.UNAVAILABLE,
                                           error=f"file not found: {path}"))
        return report
    return _analyze(config, report, path=path)


def configuration_results(config):
    """Report the configuration's own shape as results, not as an exception.

    An unrecognised key used to be read, stored and never looked at again. It
    produced no warning, no note in the JSON, and no non-zero exit, so a
    misspelt key was indistinguishable from a report that ran and found
    nothing - and configuration is this tool's whole extension mechanism.

    These surface the way every other limitation here surfaces: a visible
    result with a warning naming the key. A run still never fails on one.
    """

    results = []
    for issue in config.get("_config_issues", []):
        results.append(MetricResult(f"config.{issue['key']}", "Configuration",
                                    status="unavailable",
                                    status_type=StatusType.UNAVAILABLE,
                                    family="configuration", warning=issue["message"]))
    # ``metrics`` and ``project_measures`` are keyed by metric and report name,
    # which only this module knows, so project.py cannot check them.
    known = set(REGISTRY) | set(BUNDLED_MEASURES)
    for section in ("metrics", "project_measures"):
        values = config.get(section)
        if not isinstance(values, dict):
            continue
        for key in sorted(values):
            if str(key).startswith("_") or key in known:
                continue
            suggestion = near_miss(key, known)
            results.append(MetricResult(
                f"config.{section}.{key}", "Configuration", status="unavailable",
                status_type=StatusType.UNAVAILABLE, family="configuration",
                warning=f"unrecognised name {key!r} under {section!r}; no such metric "
                        "or report, so this entry does nothing"
                        + (f". Did you mean {suggestion!r}?" if suggestion else "")))
    return results


def _analyze(config, report, path=None, text=None):
    report.results.extend(configuration_results(config))
    settings = config.get("analysis", {}) or {}
    try:
        processing = TextProcessing.from_config(config.get("text_processing"))
        nlp_settings = NlpSettings.from_config(config.get("nlp"))
    except ValueError as exc:
        report.results.append(MetricResult("config.text_processing", "Configuration",
                                           status="error",
                                           status_type=StatusType.INTERNAL_ERROR, error=str(exc)))
        return report
    unit = settings.get("comparison_unit", "unknown")
    analysis = (DocumentAnalysis.from_path(path, processing=processing,
                                           nlp_settings=nlp_settings, comparison_unit=unit)
                if path is not None else
                DocumentAnalysis.from_text(text, processing=processing,
                                           nlp_settings=nlp_settings, source=report.source,
                                           comparison_unit=unit))
    report.document = analysis.describe()

    profile = load_profile(config, report)
    check_preprocessing(profile, analysis, report)
    comparator = Comparator(profile, analysis, settings)

    core = _core_results(analysis, config, comparator, report, profile)
    if core is not None:
        report.results.extend(core)
    report.results.extend(project_rules(analysis, config))
    report.results.extend(_optional_results(analysis, config, profile, comparator))
    if path is not None:
        report.results.extend(_external_results(path, config))
        report.results.extend(_bundled_results(path, config))
    return report


def _named_book(profile, name):
    """One corpus text by name, from either profile layout.

    The legacy profile is ``{book_name: {metric: value}}``. The current one
    puts the rows in a ``books`` LIST and identifies each by ``source_id``
    ("gutenberg:102"), ``source_filename`` or ``source_path``. Matching on any
    of those, plus the filename's stem, is what lets a benchmark be named the
    way a person would say it rather than the way the builder spelled it.
    """

    books = profile.get("books", profile)
    if isinstance(books, dict):
        found = books.get(name)
        return (found if isinstance(found, dict) else None,
                sorted(key for key in books if not str(key).startswith("_")))
    if not isinstance(books, list):
        return None, []
    names = []
    wanted = str(name).lower()
    for row in books:
        if not isinstance(row, dict):
            continue
        labels = [str(row[key]) for key in ("source_id", "source_filename", "source_path")
                  if row.get(key)]
        labels += [Path(label).stem for label in labels]
        if labels:
            names.append(labels[0])
        if any(label.lower() == wanted for label in labels):
            return row, sorted(names)
    return None, sorted(names)


def benchmark_comparison(profile, measured, name):
    """This document against ONE named corpus text, metric by metric.

    An aggregate says how far the whole has moved; it does not say what to
    open first. Ranking the losses by gap in corpus standard deviations does,
    and the standardization is what makes it possible: a 5-point gap in
    sentence-length CV and a 0.5-point gap in mean word length are not
    comparable in their own units, and are once both are in units of how much
    the corpus itself varies.

    Returns ``None`` when the named text is not in the profile, so a typo in
    the config is reported rather than silently producing no comparison.
    """

    if not profile or not name:
        return None
    row, available = _named_book(profile, name)
    if row is None:
        return {"name": name, "error": f"no text named {name!r} in the corpus profile",
                "available": available[:20], "gaps": [], "lost": 0, "of": 0}
    gaps, compared = [], 0
    for key, value in sorted(measured.items()):
        entry = METRIC_NAMES.get(key)
        if entry is None or entry[3] is NEUTRAL or value is None:
            continue
        target = row.get(key)
        if target is None:
            continue
        values = [item for item in distribution(profile, key) if item is not None]
        if len(values) < 2:
            continue
        spread = statistics.pstdev(values)
        compared += 1
        # Signed so that positive always means "behind the benchmark",
        # whichever way the metric points.
        behind = (target - value) if entry[3] is HIGHER else (value - target)
        if behind <= 0:
            continue
        gaps.append({"metric_id": f"prose.{key.lstrip('_')}", "name": entry[0],
                     "unit": entry[1], "value": round(value, 2),
                     "benchmark": round(target, 2),
                     "corpus_sd": round(spread, 4) if spread else None,
                     "gap_sd": round(behind / spread, 2) if spread else None})
    gaps.sort(key=lambda item: -(item["gap_sd"] or 0.0))
    return {"name": name, "error": None, "lost": len(gaps), "of": compared, "gaps": gaps}


def _core_results(analysis, config, comparator, report, profile=None):
    settings = config.get("analysis", {}) or {}
    lexile_source = settings.get("lexile_frequency_source", "none")
    try:
        # Floor 1 here on purpose: the measurement is descriptive and worth
        # having for a short text.  What tiny documents must NOT get is a pile
        # of confident corpus outliers, and that is the Comparator's job.
        got = core_measure(analysis, floor=1, lexile_source=lexile_source)
    except Exception as exc:
        report.results.append(MetricResult("prose.analysis", "Core prose analysis",
                                           status="error",
                                           status_type=StatusType.INTERNAL_ERROR,
                                           error=f"{type(exc).__name__}: {exc}"))
        return None
    if not got:
        report.results.append(MetricResult("prose.analysis", "Core prose analysis",
                                           status="unavailable",
                                           status_type=StatusType.UNAVAILABLE,
                                           warning="No measurable sentences"))
        return None
    out = []
    for key, value in got.items():
        if key == "_words_all" or value is None:
            continue
        name, unit, family, polarity = METRIC_NAMES.get(key, (key, None, "other", NEUTRAL))
        item = MetricResult(f"prose.{key.lstrip('_')}", name, value, unit,
                            family=family, sample_size=got.get("_sentences"),
                            comparison_unit=analysis.comparison_unit,
                            polarity=polarity)
        out.append(comparator.apply(item, key, value))
    benchmark = settings.get("benchmark")
    if benchmark:
        report.benchmark = benchmark_comparison(
            profile, {key: value for key, value in got.items() if key != "_words_all"},
            benchmark)
    lengths = analysis.sentence_lengths
    if lengths:
        shape = stats.summarize(lengths)
        out.append(MetricResult("prose.sentence_length_shape", "Sentence length distribution",
                                shape.get("median"), "words", family="sentence_rhythm",
                                sample_size=len(lengths), distribution=shape,
                                comparison_unit=analysis.comparison_unit))
    return out


def _optional_results(analysis, config, profile, comparator):
    metric_config = config.get("metrics", {}) or {}
    enabled = [name for name in REGISTRY if metric_enabled(metric_config, name)]
    out = []
    for name in enabled:
        spec = REGISTRY[name]
        options = metric_options(metric_config, name)
        started = time.monotonic()
        try:
            module = importlib.import_module(f"textgrader.metrics.{spec.module}")
            findings = module.measure(analysis, config=options, profile=profile)
        except Exception as exc:
            out.append(MetricResult(f"metric.{name}", name.replace("_", " ").title(),
                                    status="error", status_type=StatusType.INTERNAL_ERROR,
                                    family=spec.family,
                                    error=f"{type(exc).__name__}: {exc}"))
            continue
        elapsed = time.monotonic() - started
        comparable = options_match_profile(profile, name, options)
        for finding in findings or []:
            out.append(_finding_result(finding, spec, analysis, comparator, comparable,
                                       profile, elapsed))
    return out


def _finding_result(finding, spec, analysis, comparator, comparable, profile, elapsed):
    metric_id = finding["metric_id"]
    value = finding.get("value")
    warning = finding.get("warning")
    item = MetricResult(
        metric_id, finding.get("name", metric_id), value, finding.get("unit"),
        family=finding.get("family") or spec.family,
        sample_size=finding.get("sample_size"),
        distribution=finding.get("distribution"),
        details=finding.get("details", []), evidence=finding.get("evidence", []),
        channel=finding.get("channel", "full"),
        comparison_unit=analysis.comparison_unit, warning=warning)
    if elapsed > 1.0:
        item.warning = _join(item.warning,
                             f"this metric took {elapsed:.1f}s on this document ({spec.cost})")
    if value is None:
        item.status = "unavailable"
        item.status_type = StatusType.UNAVAILABLE
        item.action = Action.UNAVAILABLE
        return item
    if finding.get("sample_size_sensitive"):
        note = comparator.size_mismatch(finding.get("sample_size"))
        if note:
            item.warning = _join(item.warning, note)
            item.action = Action.INSUFFICIENT_DATA
            return item
    if finding.get("unit_sensitive"):
        # A raw count grows with the document, so comparing it across books of
        # different lengths ranks by length. Measured over thirty published
        # novels, style.repeated_ngrams correlates with word count at r = 0.96.
        # The value is still worth reporting; an outlier claim built on it is
        # not, and every one of these has a normalized sibling.
        item.warning = _join(item.warning,
                             "this is a raw count and scales with document length, so it is "
                             "reported without a corpus comparison; use the rate metric of "
                             "the same name for a length-independent reading")
        item.action = Action.INFORMATIONAL
        return item
    reference = distribution(profile, metric_id)
    if reference and not comparable:
        item.warning = _join(item.warning,
                             "the corpus distribution for this metric was built with different "
                             "or unrecorded options; comparison withheld")
        item.action = Action.INSUFFICIENT_DATA
        return item
    return comparator.apply(item, metric_id, value, finding.get("sample_size"),
                            finding.get("min_sample"))


def _external_results(path, config):
    commands = config.get("metric_commands", [])
    if not commands:
        return []
    if not config.get("allow_external_metric_commands", False):
        return [MetricResult(
            "metric.external_commands", "External metric commands", status="unavailable",
            status_type=StatusType.UNAVAILABLE,
            warning="metric_commands are disabled; set allow_external_metric_commands=true "
                    "only for trusted configuration")]
    out = []
    for item in commands:
        command = [part.format(manuscript=str(path)) for part in item["command"]]
        out.extend(run_metric_process(item["id"], command))
    return out


def _bundled_results(path, config):
    metric_config = config.get("metrics", {}) or {}
    timeout = (config.get("analysis", {}) or {}).get("bundled_timeout_seconds", 120)
    return [run_bundled_measure(name, path, config, timeout)
            for name in BUNDLED_MEASURES if metric_enabled(metric_config, name)]


# ---------------------------------------------------------------------- output

def render(report):
    print(f"TextGrader: {report.source}")
    document = report.document or {}
    if document:
        print(f"  {document.get('analyzed_words', 0):,} words analyzed of "
              f"{document.get('raw_words', 0):,} in the file; "
              f"{document.get('sentences', 0):,} sentences, "
              f"{document.get('paragraphs', 0):,} paragraphs; "
              f"segmenter={document.get('segmenter')}; "
              f"unit={document.get('comparison_unit')}")
    print("Metrics are evidence and diagnostics, not rewriting instructions.\n")
    for result in report.results:
        value = ("-" if result.value is None else f"{result.value:.2f}"
                 if isinstance(result.value, float) else str(result.value))
        value = value if len(value) <= 10 else value[:9] + "…"
        unit = f" {result.unit}" if result.unit else ""
        marker = {"review": "!", "rule_violation": "!", "error": "E"}.get(result.action.value, " ")
        print(f" {marker} {result.metric_id:<44} {value:>10}{unit:<22} "
              f"[{result.action.value}]")
        if result.warning:
            print(f"      warning: {result.warning}")
        if result.error:
            print(f"      ERROR: {result.error}")
    summary = report.summary()
    print(f"\n{summary['total']} structured results; "
          f"{summary['by_action']['review']} to review; "
          f"{summary['by_action']['rule_violation']} project-rule violations; "
          f"{summary['by_action']['insufficient_data']} without enough data; "
          f"{summary['by_status_type']['internal_error']} internal errors.")
    print_scorecard(summary["scorecard"])
    print_maturity(summary["maturity"])
    if summary["top_findings"]:
        print("\nFurthest from the corpus, most distant first:")
        for item in summary["top_findings"][:8]:
            severity = "-" if item["severity"] is None else f"{item['severity']:.1f}"
            print(f"  {item['metric_id']:<44} {item['direction']:<8} severity {severity}")


def print_maturity(maturity):
    """The aggregate, and where to start if a benchmark is configured."""

    benchmark = maturity["benchmark"]
    # A misnamed benchmark is reported even when there is no aggregate to
    # print. Without a corpus profile the percentile is None, and returning
    # early here would swallow a configured-but-wrong benchmark completely,
    # which is the failure mode the rest of this work exists to remove.
    if benchmark and benchmark.get("error"):
        print(f"\n  benchmark unavailable: {benchmark['error']}")
        if maturity["percentile"] is None:
            return
    if maturity["percentile"] is None:
        return
    # The printed label and the JSON key are the same word on purpose. A
    # measure that prints under one name and serializes under another is how
    # this repository's TTR column came to hold MSTTR.
    line = (f"\n  maturity percentile (median of {maturity['metric_count']} "
            f"oriented measures): {maturity['percentile']:.0f}")
    if benchmark and benchmark.get("error"):
        print(line)
        return
    if benchmark and benchmark["of"]:
        line += f"      behind {benchmark['name']} on {benchmark['lost']} of {benchmark['of']}"
    print(line)
    if not (benchmark and benchmark["gaps"]):
        return
    print(f"\n  largest gaps to {benchmark['name']}, in corpus standard deviations:")
    for gap in benchmark["gaps"][:8]:
        unit = (gap["unit"] or "")[:14]
        distance = "-" if gap["gap_sd"] is None else f"{gap['gap_sd']:.1f} sd"
        print(f"    {gap['name'][:40]:<42}{gap['value']:>9} vs {gap['benchmark']:>9}"
              f"  {unit:<15}{distance:>8}")


def print_scorecard(scorecard):
    """The census, with its denominator.

    ``not_taken`` is printed whether or not it is zero. It is the number that
    moves silently when instrumentation breaks, and a count that only appears
    when it is interesting is a count nobody learns to look for.
    """

    share = scorecard["passing_share"]
    print("\nSCORECARD")
    if scorecard["measured"]:
        print(f"  {scorecard['passing']} of {scorecard['measured']} measures "
              f"inside their reference ({share:.0f}%)")
    else:
        print("  no measure produced a comparable result")
    print(f"  {scorecard['not_taken']} not taken "
          "(no data, unavailable, or errored - neither a pass nor a failure)")
    if scorecard["configuration_issues"]:
        print(f"  {scorecard['configuration_issues']} configuration issue(s); "
              "see the config.* results above")
    families = {name: counts for name, counts in scorecard["by_family"].items()
                if counts["failing"]}
    if families:
        print("  outside their reference, by family:")
        for name, counts in sorted(families.items(), key=lambda kv: -kv[1]["failing"]):
            total = counts["passing"] + counts["failing"]
            print(f"    {name:<28}{counts['failing']:>3} of {total}")
    if scorecard["not_taken_detail"]:
        errors = [item for item in scorecard["not_taken_detail"] if item["action"] == "error"]
        if errors:
            print("  errored, so they left the count entirely:")
            for item in errors:
                print(f"    {item['metric_id']:<32}{(item['warning'] or '')[:60]}")


def list_metrics():
    rows = sorted(REGISTRY.values(), key=lambda spec: (spec.family, spec.name))
    print(f"{'metric':<34}{'family':<20}{'cost':<10}{'needs'}")
    for spec in rows:
        print(f"{spec.name:<34}{spec.family:<20}{spec.cost:<10}"
              f"{', '.join(spec.requires) or '-'}")
    print(f"\n{len(rows)} metrics. cost on a 300,000-word novel: "
          f"fast = under half a second, moderate = up to a few seconds, "
          f"parse = tens of seconds sharing one spaCy parse, "
          f"model = tens of seconds encoding the text with a sentence-embedding model.")


def apply_switches(config, enable, disable, families, enable_all):
    metrics = config.setdefault("metrics", {})
    names = set(enable or [])
    for family in families or []:
        names |= {name for name, spec in REGISTRY.items() if spec.family == family}
    if enable_all:
        names |= set(REGISTRY)
    unknown = sorted(names - set(REGISTRY) - set(BUNDLED_MEASURES))
    if unknown:
        raise SystemExit(f"error: unknown metric(s): {', '.join(unknown)}")
    for name in names:
        current = metrics.get(name)
        metrics[name] = {**(current if isinstance(current, dict) else {}), "enabled": True}
    for name in disable or []:
        current = metrics.get(name)
        metrics[name] = ({**current, "enabled": False} if isinstance(current, dict) else False)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manuscript", nargs="?", help="Markdown or text manuscript")
    parser.add_argument("--config", default=None,
                        help="project configuration; also reaches the bundled reports")
    parser.add_argument("--json", action="store_true",
                        help="emit stable JSON instead of terminal text")
    parser.add_argument("--json-out", help="write JSON to this file as well")
    parser.add_argument("--enable", action="append", metavar="METRIC",
                        help="turn one metric on for this run; repeatable")
    parser.add_argument("--enable-family", action="append", metavar="FAMILY",
                        help="turn on every metric in a family; repeatable")
    parser.add_argument("--enable-all", action="store_true",
                        help="turn on every registered metric (slow: includes the parse)")
    parser.add_argument("--disable", action="append", metavar="METRIC",
                        help="turn one metric off for this run; repeatable")
    parser.add_argument("--comparison-unit", choices=COMPARISON_UNITS,
                        help="what the input is, so scale-dependent comparisons are honest")
    parser.add_argument("--list-metrics", action="store_true")
    args = parser.parse_args(argv)

    if args.list_metrics:
        list_metrics()
        return 0
    config = load_config(args.config)
    config = apply_switches(config, args.enable, args.disable, args.enable_family,
                            args.enable_all)
    if args.comparison_unit:
        config.setdefault("analysis", {})["comparison_unit"] = args.comparison_unit
    manuscript = (Path(args.manuscript) if args.manuscript
                  else Path(config["_config_dir"]) / config.get("manuscript", "MANUSCRIPT.md"))
    report = analyze(manuscript, config)
    output = json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str)
    if args.json_out:
        Path(args.json_out).write_text(output + "\n", encoding="utf-8")
    if args.json:
        print(output)
    else:
        render(report)
    return 2 if report.summary()["has_internal_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
