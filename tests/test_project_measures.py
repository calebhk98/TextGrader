"""The bundled reports carry no manuscript of their own any more."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import grade

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "textgrader" / "reports"
EXAMPLE = ROOT / "examples" / "project_measures.example.json"

# Names and fragments from the one manuscript this repository was extracted
# from.  None of them may appear in the code any more; they live in the example
# configuration instead.
FORBIDDEN = ["chloe", "kavi", "nadia", "kayleigh", "marisol", "aldana",
             "28_nineteen", "tom_sawyer", "treasure_island", "wind_in_willows",
             "puts it back down", "both hands", "MANUSCRIPT_FULL"]


@pytest.mark.parametrize("path", sorted(REPORTS.glob("*.py")))
def test_no_measure_names_one_manuscript(path):
    body = path.read_text(encoding="utf-8").lower()
    found = [needle for needle in FORBIDDEN if needle.lower() in body]
    assert not found, f"{path.name} still hard-codes {found}"


def test_the_example_config_is_valid_json_and_marked_as_an_example():
    assert EXAMPLE.is_file(), "the extracted policy must be preserved as an example"
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert any("example" in str(key).lower() or "example" in str(value).lower()
               for key, value in data.items() if not isinstance(value, (dict, list)))


def test_example_patterns_all_compile():
    import re
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "pattern" and isinstance(value, str):
                    re.compile(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)


@pytest.mark.parametrize("name", sorted(grade.BUNDLED_MEASURES))
def test_every_bundled_report_runs_clean_with_no_policy(name, tmp_path, sample_text):
    """With no configured policy a report describes and succeeds.

    "No house rule configured" is not a violation, and a report that exited 1
    for it would turn every default run into a false diagnostic.
    """

    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "manuscript": "draft.md", "chapters_dir": "chapters",
        "characters_dir": "characters", "project_measures": {},
        "metrics": {}}), encoding="utf-8")
    (tmp_path / "chapters").mkdir()
    (tmp_path / "characters").mkdir()
    manuscript = tmp_path / "draft.md"
    manuscript.write_text(sample_text, encoding="utf-8")
    (tmp_path / "chapters" / "01_one.md").write_text(sample_text, encoding="utf-8")

    # Through grade.py's own resolver, not a second copy of it: the templates
    # name a closed set of targets and the test should not have to learn them.
    arguments = grade.measure_arguments(
        name, manuscript,
        {"_config_dir": str(tmp_path), "_config_path": str(config),
         "chapters_dir": "chapters"})
    completed = subprocess.run(
        [sys.executable, "-m", f"textgrader.reports.{name}", *arguments],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
        env={"PATH": "/usr/bin:/bin", "HALSTEAD_VIA_GRADE": "1",
             "TEXTGRADER_CONFIG": str(config),
             "PYTHONPATH": str(ROOT)})
    assert "Traceback" not in completed.stderr, completed.stderr[-2000:]
    assert completed.returncode == 0, (completed.returncode, completed.stdout[-1500:],
                                       completed.stderr[-1500:])


def test_a_bundled_report_reads_the_file_it_was_given(tmp_path):
    """Several reports used to read a global chapters directory regardless of
    which file grade.py was asked to analyze."""

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"chapters_dir": "chapters", "project_measures": {}}),
                      encoding="utf-8")
    (tmp_path / "chapters").mkdir()
    (tmp_path / "chapters" / "01_other.md").write_text(
        "Incontrovertible extraordinary circumstances notwithstanding. " * 40,
        encoding="utf-8")
    target = tmp_path / "given.md"
    target.write_text("The cat sat on the mat and then it left. " * 40, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "textgrader.reports.register", str(target)],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin", "HALSTEAD_VIA_GRADE": "1",
             "TEXTGRADER_CONFIG": str(config), "PYTHONPATH": str(ROOT)})
    assert completed.returncode == 0, completed.stderr[-1500:]
    assert "given" in completed.stdout
    assert "01_other" not in completed.stdout


def test_build_manuscript_checks_what_its_docstring_promises(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("bm", ROOT / "build_manuscript.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = (ROOT / "build_manuscript.py").read_text(encoding="utf-8")
    # The docstring promised gap and heading checks that check() never made.
    assert "gap" in source.lower()
    assert "heading" in source.lower()


def test_an_unknown_config_key_is_reported_not_swallowed(tmp_path):
    """A misspelt key used to be read, stored and never looked at again.

    Adding "totally_made_up_key": "banana" to a working config produced
    byte-identical output to leaving it out: no warning, no note in the JSON
    report, no non-zero exit. Configuration is this tool's whole extension
    mechanism, so a key that does nothing has to say so.
    """

    from textgrader.project import config_issues
    issues = config_issues({
        "manuscript": "draft.md",
        "reading_grade_bands": {"1-10": 5.5},
        "scorecard": {"enabled": True},
        "maturity_percentile": {"benchmark": "peter_pan"},
        "totally_made_up_key": "banana",
    })
    reported = {issue["key"] for issue in issues}
    assert reported == {"reading_grade_bands", "scorecard",
                        "maturity_percentile", "totally_made_up_key"}
    assert all(issue["kind"] == "unknown" for issue in issues)


def test_a_near_miss_key_suggests_the_real_one(tmp_path):
    from textgrader.project import config_issues
    issues = config_issues({"chapters_dr": "chapters",
                            "analysis": {"comparison_unt": "book"}})
    messages = " ".join(issue["message"] for issue in issues)
    assert "'chapters_dir'" in messages
    assert "'comparison_unit'" in messages


def test_a_comment_key_is_not_a_typo():
    # JSON has no comments, so "_comment" is how every example config in this
    # repository documents itself. Flagging those would train people to ignore
    # the warning, which costs more than the check is worth.
    from textgrader.project import config_issues
    assert config_issues({"_comment": "explanatory", "manuscript": "draft.md"}) == []


def test_the_shipped_configs_are_clean():
    # If the tool's own configuration cannot pass its own check, nobody will
    # believe the check.
    from textgrader.project import config_issues
    for name in ("config.json", "examples/project_measures.example.json"):
        assert config_issues(json.loads((ROOT / name).read_text(encoding="utf-8"))) == [], name


def test_an_advertised_but_unimplemented_rule_says_so():
    # config.json advertises project_rules.hard_line_breaks and chapter_length
    # and nothing reads either. Shipping them as null is fine; a user who sets
    # one is waiting for an effect that never arrives.
    from textgrader.project import config_issues
    issues = config_issues({"project_rules": {"chapter_length": {"min": 900},
                                              "hard_line_breaks": "forbid",
                                              "em_dash": "forbid"}})
    assert {issue["key"] for issue in issues} == {"project_rules.chapter_length",
                                                  "project_rules.hard_line_breaks"}
    assert all(issue["kind"] == "unimplemented" for issue in issues)


def test_an_unknown_metric_name_is_reported_against_the_live_registry(tmp_path, base_config):
    results = grade.configuration_results({**base_config,
                                           "metrics": {"registre": True, "register": True}})
    warnings = [item.warning for item in results]
    assert any("registre" in text and "'register'" in text for text in warnings)
    assert not any("'registre'" == text for text in warnings if text)


def test_a_run_still_exits_zero_with_a_bad_key(tmp_path, capsys, sample_text):
    # A run never fails. An unrecognised key is a visible result, not an exit
    # code and not an exception.
    manuscript = tmp_path / "draft.md"
    manuscript.write_text(sample_text, encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"manuscript": "draft.md", "corpus_profile": "",
                                  "metrics": {}, "totally_made_up_key": "banana"}),
                      encoding="utf-8")
    assert grade.main([str(manuscript), "--config", str(config), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    configuration = [item for item in payload["results"] if item["family"] == "configuration"]
    assert [item["metric_id"] for item in configuration] == ["config.totally_made_up_key"]
    assert configuration[0]["action"] == "unavailable"
