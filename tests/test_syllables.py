"""The core syllable counter, which feeds the core Flesch-Kincaid value.

Checked against CMUdict over the reference corpus's vocabulary (see the
docstring of ``core_metrics.syllables``).  These pin the cases the previous
vowel-group count got wrong, one or two per rule, plus words it already got
right and must keep.
"""

import pytest

from textgrader.core_metrics import syllables


@pytest.mark.parametrize("word, expected", [
    # Hiatus: two vowels spoken as two syllables.
    ("fire", 2), ("hour", 2), ("poem", 2), ("science", 2), ("cruel", 2),
    ("idea", 3), ("quiet", 2), ("going", 2), ("being", 2), ("playing", 2),
    ("curious", 3), ("society", 4), ("area", 3), ("inquired", 3),
    # Silent -ed and -es after a consonant.
    ("looked", 1), ("seemed", 1), ("times", 1), ("gives", 1),
    # Silent final e, including before -le when a vowel precedes it.
    ("while", 1), ("whole", 1), ("smile", 1), ("value", 2),
    # Silent e closing the first part of a word.
    ("something", 2), ("therefore", 2), ("carefully", 3), ("movement", 2),
    # Abbreviations and contractions spoken in full.
    ("mr", 2), ("mrs", 2), ("isn't", 2), ("couldn't", 2),
])
def test_cases_the_old_count_got_wrong(word, expected):
    assert syllables(word) == expected


@pytest.mark.parametrize("word, expected", [
    ("table", 2), ("little", 2), ("wanted", 2), ("horses", 2), ("places", 2),
    ("settled", 2), ("hundred", 2), ("agreed", 2), ("people", 2), ("nation", 2),
    ("sea", 1), ("tea", 1), ("free", 1), ("four", 1), ("your", 1), ("power", 2),
    ("quality", 3), ("square", 1), ("patient", 2), ("tales", 1), ("tables", 2),
    ("filled", 1), ("a", 1),
])
def test_cases_it_must_keep_right(word, expected):
    assert syllables(word) == expected


def test_every_word_counts_at_least_one_syllable():
    for word in ("", "'", "hmm", "brr", "x", "'s"):
        assert syllables(word) >= 1
