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

    arguments = [part.format(manuscript=str(manuscript)) for part in grade.BUNDLED_MEASURES[name]]
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
