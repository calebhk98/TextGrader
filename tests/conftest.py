"""Shared fixtures.

The suite is pytest-based.  ``python -m unittest discover`` used to be the
documented command, and it silently skipped every module-level test function
in ``test_build_corpus.py`` - a green run that had not run the tests.
"""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _prose(seed: int, paragraphs: int = 60, dialogue: bool = True) -> str:
    """Deterministic pseudo-prose with sentences, paragraphs and dialogue."""

    rng = random.Random(seed)
    vocab = ("the quiet room held a long silence while she considered what had happened "
             "and whether anyone would notice however perhaps not because nobody asked "
             "her directly about any of it").split()
    names = ["Ruth", "Sam", "Nadia"]
    blocks = []
    for _ in range(paragraphs):
        if dialogue and rng.random() < 0.3:
            turns = []
            for _ in range(rng.randint(1, 3)):
                line = " ".join(rng.choice(vocab) for _ in range(rng.randint(2, 16)))
                turns.append(f'"{line.capitalize()}," {rng.choice(names)} said.')
            blocks.append(" ".join(turns))
        else:
            sentences = [" ".join(rng.choice(vocab) for _ in range(rng.randint(3, 28)))
                         .capitalize() + rng.choice([".", ".", ".", "?", "!"])
                         for _ in range(rng.randint(1, 5))]
            blocks.append(" ".join(sentences))
    return "\n\n".join(blocks)


@pytest.fixture(scope="session")
def prose():
    return _prose


@pytest.fixture(scope="session")
def sample_text():
    return _prose(11)


@pytest.fixture
def manuscript(tmp_path, sample_text):
    path = tmp_path / "draft.md"
    path.write_text(sample_text, encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("corpus")
    for index in range(12):
        (directory / f"book{index:02d}.txt").write_text(_prose(100 + index, 70),
                                                        encoding="utf-8")
    return directory


@pytest.fixture
def base_config(tmp_path):
    """A configuration with every bundled report off and no corpus."""

    import grade
    return {
        "_config_dir": str(tmp_path),
        "_config_path": str(tmp_path / "config.json"),
        "analysis": {"comparison_unit": "book"},
        "metrics": {name: False for name in grade.BUNDLED_MEASURES},
    }
