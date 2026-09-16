"""Leave-one-out validation: the calibration check, checked."""

import pytest

import validate_corpus
from textgrader.corpus import build_profile


@pytest.fixture(scope="module")
def profile(tmp_path_factory, prose):
    directory = tmp_path_factory.mktemp("validation")
    for index in range(10):
        (directory / f"book{index:02d}.txt").write_text(prose(200 + index, 70),
                                                        encoding="utf-8")
    return directory, build_profile([directory], built_at="2026-01-01T00:00:00Z")


def test_holdout_removes_exactly_one_text(profile):
    _, full = profile
    target = full["books"][3]["source_id"]
    reduced = validate_corpus.holdout_profile(full, target)
    assert reduced["book_count"] == full["book_count"] - 1
    assert target not in {book["source_id"] for book in reduced["books"]}
    assert len(reduced["feature_profiles"]["function_words"]) == full["book_count"] - 1


def test_holdout_distribution_matches_a_real_rebuild(profile):
    """The shortcut has to be the same thing as rebuilding, or it proves nothing.

    Dropping a text's value from the pooled list must give the distribution a
    genuine rebuild from the remaining texts would give. It is exact because
    the distributions are pooled per-text values, and that identity is the only
    reason this check is affordable.
    """

    directory, full = profile
    target = full["books"][2]
    shortcut = validate_corpus.holdout_profile(full, target["source_id"])
    rebuilt = build_profile(
        [path for path in sorted(directory.glob("*.txt"))
         if path.name != target["source_filename"]],
        built_at="2026-01-01T00:00:00Z")
    for key in ("wps", "style.mattr", "prose.fk" if "prose.fk" in rebuilt["distributions"] else "fk"):
        if key not in rebuilt["distributions"]:
            continue
        assert shortcut["distributions"][key]["values"] == pytest.approx(
            rebuilt["distributions"][key]["values"]), key
        assert shortcut["distributions"][key]["median"] == pytest.approx(
            rebuilt["distributions"][key]["median"]), key


def test_validation_runs_and_reports_rates(tmp_path_factory, prose, base_config):
    directory = tmp_path_factory.mktemp("run")
    for index in range(9):
        (directory / f"book{index:02d}.txt").write_text(prose(300 + index, 60),
                                                        encoding="utf-8")
    config = {**base_config, "analysis": {"comparison_unit": "book"}}
    result = validate_corpus.validate([directory], config, quiet=True)
    assert result["held_out"] == 9
    assert all(row["compared"] > 0 for row in result["books"])
    assert not [error for row in result["books"] for error in row["errors"]]
    for row in result["metrics"]:
        assert 0 < row["rate"] <= 1
        assert row["of"] == 9


def test_validation_refuses_a_corpus_too_small(tmp_path, prose, base_config):
    (tmp_path / "only.txt").write_text(prose(1, 20), encoding="utf-8")
    with pytest.raises(SystemExit, match="at least 3"):
        validate_corpus.validate([tmp_path], base_config, quiet=True)
