#!/usr/bin/env python3
"""Analyze English-fiction prose and emit structured evidence.

The default run makes descriptive measurements only.  Corpus outliers are not
failures and project rules are applied only when explicitly enabled in the
configuration.  Use ``--json`` for the stable machine-readable representation.
"""

import argparse
import importlib.util
import importlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from project_config import CONFIG, CONFIG_PATH, MANUSCRIPT, PROJECT_RULES, load_config
from textgrader.results import MetricResult, Report, StatusType
from textgrader.metrics import MODULES, NLP_METRICS

ROOT = Path(__file__).resolve().parent
BUNDLED_MEASURES = {
    "absolutes": (), "banned_phrases": (), "check_edits": (),
    "dialogue_study": ("{manuscript_dir}",), "number_report": ("{manuscript}",),
    "prose_check": ("{manuscript}",), "quotable": (), "quote_length": (),
    "register": (), "style_report": ("{manuscript}",), "tics": (),
    "verify_citations": ("{manuscript}",), "voice_separation": (),
}
METRIC_NAMES = {
    "fk": ("Flesch-Kincaid grade", "grade"), "ari": ("Automated Readability Index", "grade"),
    "wps": ("Words per sentence", "words/sentence"), "slcv": ("Sentence-length variation", "%"),
    "wpp": ("Words per paragraph", "words/paragraph"), "spp": ("Sentences per paragraph", "sentences/paragraph"),
    "wlen": ("Mean word length", "characters"), "long7": ("Words of 7+ characters", "%"),
    "sttr": ("Standardized type-token ratio", "%"), "top100": ("Commonest-100 word share", "%"),
    "commas": ("Commas per sentence", "commas/sentence"),
    "subord": ("Subordinator-cue sentence share (lexical proxy)", "%"),
    "relcl": ("Relative-word sentence share (lexical proxy)", "%"),
    "simple": ("No-clause-cue sentence share (lexical proxy)", "%"),
    "u10": ("Sentences under 10 words", "%"), "b2035": ("Sentences 20–35 words", "%"),
    "shortruns": ("Sentences in short runs", "%"), "front": ("Front-loaded cue proxy", "%"),
    "and2": ('Sentences with two or more "and" tokens', "%"),
    "andrate": ('"and" share', "%"), "negative": ("Negative-cue sentence share", "%"),
    "_words": ("Word count", "words"), "_sentences": ("Sentence count", "sentences"),
    "_paragraphs": ("Paragraph count", "paragraphs"), "_transcript": ("Transcript word share", "%"),
}


def _load_prose_module():
    measures = ROOT / "measures"
    if str(measures) not in sys.path:
        sys.path.insert(0, str(measures))
    spec = importlib.util.spec_from_file_location("textgrader_prose_grade", measures / "prose_grade.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_metric_process(metric_id, command):
    """Run an optional external metric without losing crashes or findings.

    External metrics use a small JSON protocol: stdout is either one result or
    a list of result dictionaries.  A non-zero exit is always INTERNAL_ERROR;
    prose findings belong in successful structured output, never exit status.
    """
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


def run_bundled_measure(name, manuscript, config):
    """Run a bundled report as a structured metric, never via a shell."""
    if name == "quote_length" and not config.get("dialogue_targets"):
        return MetricResult(
            "measure.quote_length", "Quote Length", status="unavailable",
            status_type=StatusType.UNAVAILABLE,
            warning="quote_length requires configured dialogue_targets")
    arguments = [part.format(manuscript=str(manuscript), manuscript_dir=str(manuscript.parent))
                 for part in BUNDLED_MEASURES[name]]
    command = [sys.executable, str(ROOT / "measures" / f"{name}.py"), *arguments]
    try:
        environment = {**os.environ, "HALSTEAD_VIA_GRADE": "1"}
        completed = subprocess.run(command, cwd=config.get("_config_dir", ROOT), env=environment,
                                   capture_output=True, text=True,
                                   check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return MetricResult(f"measure.{name}", name.replace("_", " ").title(), status="error",
                            status_type=StatusType.INTERNAL_ERROR, error=str(exc))
    output = completed.stdout.strip()
    error = completed.stderr.strip()
    crashed = "Traceback (most recent call last)" in error
    if completed.returncode not in (0, 1) or crashed:
        return MetricResult(f"measure.{name}", name.replace("_", " ").title(), status="error",
                            status_type=StatusType.INTERNAL_ERROR,
                            error=error or output or f"process exited {completed.returncode}")
    details = [{"output": output}] if output else []
    return MetricResult(f"measure.{name}", name.replace("_", " ").title(),
                        status="review" if completed.returncode else "available",
                        status_type=(StatusType.DIAGNOSTIC if completed.returncode
                                     else StatusType.INFORMATIONAL),
                        details=details, warning=error or None)


def _distribution(profile, key):
    if not profile:
        return []
    # Current profiles store per-book metric dictionaries.  Also accept a
    # future pre-aggregated ``distributions`` mapping without changing grading.
    if "distributions" in profile:
        distribution = profile["distributions"].get(key, [])
        if isinstance(distribution, dict):
            distribution = distribution.get("values", [])
        return [v for v in distribution if v is not None]
    books = profile.get("books", profile)
    if isinstance(books, dict):
        return [row[key] for row in books.values() if isinstance(row, dict) and row.get(key) is not None]
    return []


def _corpus_result(key, value, profile):
    name, unit = METRIC_NAMES.get(key, (key, None))
    values = sorted(_distribution(profile, key))
    if not values:
        return MetricResult(f"prose.{key.lstrip('_')}", name, value, unit,
                            sample_size=None, warning="No corpus distribution available")
    median = statistics.median(values)
    deviations = [abs(item - median) for item in values]
    mad = statistics.median(deviations)
    percentile = 100 * (sum(item < value for item in values) + .5 * sum(item == value for item in values)) / len(values)
    # A zero MAD means every central observation is identical.  A differing
    # value is therefore maximally, rather than immeasurably, distant.
    robust_distance = (value - median) / (1.4826 * mad) if mad else (0.0 if value == median else None)
    outlier = (value != median) if mad == 0 else abs(robust_distance) > 3.5
    stats = {"profile_name": profile.get("corpus_name", profile.get("metadata", {}).get("corpus_name")),
             "count": len(values), "median": median, "mad": mad,
             "percentile": percentile, "robust_distance": robust_distance,
             "method": "two-sided median/MAD (|robust_distance| > 3.5)"}
    return MetricResult(f"prose.{key.lstrip('_')}", name, value, unit,
                        "outlier" if outlier else "inlier",
                        StatusType.CORPUS_OUTLIER if outlier else StatusType.CORPUS_INLIER,
                        stats, len(values))


def _rules(text, rules):
    import re
    results = []
    for index, rule in enumerate(rules.get("banned_phrases", [])):
        pattern = rule.get("pattern", "")
        hits = [{"match": match.group(0), "offset": match.start()} for match in re.finditer(pattern, text, re.I)]
        if hits:
            rid = rule.get("id", f"banned_phrase_{index + 1}")
            results.append(MetricResult(f"rule.{rid}", rule.get("name", rid), len(hits), "hits",
                                        "violation", StatusType.PROJECT_RULE, sample_size=len(hits), details=hits))
    if rules.get("em_dash") == "forbid":
        hits = [{"offset": m.start()} for m in re.finditer("—", text)]
        if hits:
            results.append(MetricResult("rule.em_dash", "Em-dash policy", len(hits), "hits", "violation",
                                        StatusType.PROJECT_RULE, details=hits, sample_size=len(hits)))
    return results


def analyze(path, config):
    path = Path(path)
    report = Report(source=str(path))
    if not path.is_file():
        report.results.append(MetricResult("input.manuscript", "Manuscript input", status="unavailable",
                                           status_type=StatusType.UNAVAILABLE, error=f"file not found: {path}"))
        return report
    text = path.read_text(encoding="utf-8")
    try:
        got = _load_prose_module().measure(text, floor=1)
    except Exception as exc:
        report.results.append(MetricResult("prose.analysis", "Core prose analysis", status="error",
                                           status_type=StatusType.INTERNAL_ERROR, error=f"{type(exc).__name__}: {exc}"))
        return report
    if not got:
        report.results.append(MetricResult("prose.analysis", "Core prose analysis", status="unavailable",
                                           status_type=StatusType.UNAVAILABLE,
                                           warning="No measurable sentences"))
        return report
    profile_path = config.get("corpus_profile")
    profile = None
    if profile_path:
        candidate = Path(config.get("_config_dir", ROOT)) / profile_path
        if candidate.is_file():
            try:
                profile = json.loads(candidate.read_text(encoding="utf-8"))
                report.corpus_profile = profile.get("metadata", {
                    key: profile.get(key) for key in ("schema_version", "corpus_name", "build_timestamp",
                                                      "parser_version", "metric_definition_version")
                })
                if not report.corpus_profile.get("corpus_name"):
                    report.corpus_profile["corpus_name"] = candidate.stem
                    report.corpus_profile["book_count"] = len(profile.get("books", profile))
            except (OSError, ValueError) as exc:
                report.results.append(MetricResult("corpus.profile", "Corpus profile", status="unavailable",
                                                   status_type=StatusType.UNAVAILABLE, warning=str(exc)))
    for key, value in got.items():
        if key == "_words_all" or value is None:
            continue
        report.results.append(_corpus_result(key, value, profile))
    report.results.extend(_rules(text, config.get("project_rules", {})))
    # Every configurable measurement uses the same switch map. Modules are
    # loaded independently so one optional dependency cannot hide the rest.
    metric_config = config.get("metrics", {})
    def metric_enabled(name):
        setting = metric_config.get(name, name in BUNDLED_MEASURES)
        return setting is True or (isinstance(setting, dict) and setting.get("enabled", False))

    def metric_options(name):
        setting = metric_config.get(name, {})
        return setting if isinstance(setting, dict) else {}

    def comparison_is_compatible(name):
        if not profile or "metric_settings" not in profile:
            return False
        expected = dict(profile["metric_settings"].get(name, {}))
        actual = dict(metric_options(name))
        expected.pop("enabled", None)
        actual.pop("enabled", None)
        return expected == actual

    enabled = [name for name in MODULES if metric_enabled(name)]
    nlp = None
    if any(name in NLP_METRICS for name in enabled):
        try:
            import spacy
            nlp = spacy.load(config.get("nlp", {}).get("model", "en_core_web_sm"))
        except (ImportError, OSError):
            nlp = None
    for name in enabled:
        try:
            module = importlib.import_module(f"textgrader.metrics.{MODULES[name]}")
            findings = module.measure(text, config=metric_options(name), profile=profile, nlp=nlp)
            for finding in findings:
                metric_id = finding["metric_id"]
                value = finding.get("value")
                # Scalar measures use exactly the same corpus machinery as the
                # longstanding prose metrics when a profile has that feature.
                if value is not None and _distribution(profile, metric_id) and comparison_is_compatible(name):
                    item = _corpus_result(metric_id, value, profile)
                    item.metric_id, item.name = metric_id, finding["name"]
                    item.unit = finding.get("unit")
                    item.details = finding.get("details", [])
                    item.warning = finding.get("warning")
                else:
                    unavailable = value is None and finding.get("warning")
                    mismatch = (value is not None and _distribution(profile, metric_id)
                                and not comparison_is_compatible(name))
                    item = MetricResult(metric_id, finding["name"], value, finding.get("unit"),
                                        status="unavailable" if unavailable else "available",
                                        status_type=StatusType.UNAVAILABLE if unavailable else StatusType.INFORMATIONAL,
                                        details=finding.get("details", []),
                                        warning=("Corpus distribution was built with different or unrecorded metric options"
                                                 if mismatch else finding.get("warning")))
                report.results.append(item)
        except Exception as exc:
            report.results.append(MetricResult(f"metric.{name}", name.replace("_", " ").title(),
                                               status="error", status_type=StatusType.INTERNAL_ERROR,
                                               error=f"{type(exc).__name__}: {exc}"))
    commands = config.get("metric_commands", [])
    if commands and not config.get("allow_external_metric_commands", False):
        report.results.append(MetricResult(
            "metric.external_commands", "External metric commands", status="unavailable",
            status_type=StatusType.UNAVAILABLE,
            warning="metric_commands are disabled; set allow_external_metric_commands=true only for trusted configuration"))
    else:
        for item in commands:
            command = [part.format(manuscript=str(path)) for part in item["command"]]
            report.results.extend(run_metric_process(item["id"], command))
    for name in BUNDLED_MEASURES:
        if metric_enabled(name):
            report.results.append(run_bundled_measure(name, path, config))
    return report


def render(report):
    print(f"TextGrader: {report.source}")
    print("Metrics are evidence and diagnostics, not rewriting instructions.\n")
    for result in report.results:
        value = "—" if result.value is None else f"{result.value:.2f}" if isinstance(result.value, float) else str(result.value)
        unit = f" {result.unit}" if result.unit else ""
        print(f"  {result.metric_id:<28} {value:>10}{unit:<20} [{result.status_type.value}]")
        if result.warning:
            print(f"    warning: {result.warning}")
        if result.error:
            print(f"    ERROR: {result.error}")
    summary = report.summary()
    print(f"\n{summary['total']} structured results; "
          f"{summary['by_status_type']['corpus_outlier']} corpus outliers; "
          f"{summary['by_status_type']['project_rule_violation']} project-rule violations; "
          f"{summary['by_status_type']['internal_error']} internal errors.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manuscript", nargs="?", help="Markdown or text manuscript")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--json", action="store_true", help="emit stable JSON instead of terminal text")
    parser.add_argument("--json-out", help="write JSON to this file in addition to terminal text")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    manuscript = Path(args.manuscript) if args.manuscript else (Path(config["_config_dir"]) / config.get("manuscript", "MANUSCRIPT.md"))
    report = analyze(manuscript, config)
    output = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).write_text(output + "\n", encoding="utf-8")
    if args.json:
        print(output)
    else:
        render(report)
    return 2 if report.summary()["has_internal_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
