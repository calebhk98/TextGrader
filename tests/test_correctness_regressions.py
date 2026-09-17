"""Focused regression tests for correctness bugs in legacy metric entrypoints."""
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from textgrader import core_metrics as prose_grade
from textgrader.chapters import chapter_number
from textgrader.reports.voice_separation import profile, show


class CorrectnessRegressions(unittest.TestCase):
    def test_sttr_includes_exactly_one_complete_window(self):
        result = prose_grade.measure("word " * 1000 + ".", floor=1)
        self.assertAlmostEqual(result["sttr"], 0.1)

    def test_sttr_includes_both_complete_windows(self):
        text = "alpha " * 1000 + ". " + "beta " * 1000 + "."
        result = prose_grade.measure(text, floor=1)
        self.assertAlmostEqual(result["sttr"], 0.1)

    def test_arbitrary_length_chapter_number(self):
        self.assertEqual(chapter_number("100_epilogue.md"), 100)
        self.assertEqual(chapter_number("7-opening.md"), 7)
        self.assertIsNone(chapter_number("appendix.md"))

    def test_twenty_word_sentence_is_in_20_to_35_bucket(self):
        text = " ".join(["word"] * 20) + "."
        result = prose_grade.measure(text, floor=1)
        self.assertEqual(result["b2035"], 100.0)

    def test_voice_spread_with_all_zero_shares(self):
        # Eight four-word lines per speaker make the short-line shares all zero.
        data = {"a": ["one two three four"] * 8,
                "b": ["five six seven eight"] * 8}
        with redirect_stdout(io.StringIO()) as output:
            show("voices", data, 8, "test")
        self.assertIn("spread 0%", output.getvalue())

    def test_voice_lexical_diversity_is_segment_standardized(self):
        short = profile(["same"] * 50)["msttr"]
        long = profile(["same"] * 100)["msttr"]
        self.assertEqual(short, long)

    def test_msttr_drops_the_short_trailing_window(self):
        # A tail scores near 100% type-token ratio and was averaged with the
        # same weight as a full window, so it pulled the mean up by 1/segments:
        # hardest on the speakers with least text, which is the plain-TTR
        # artefact MSTTR exists to remove.
        lines = ["alpha beta gamma delta epsilon"] * 20   # 100 words, two full windows
        clean = profile(lines)
        self.assertEqual(clean["msttr_segments"], 2)
        tailed = profile(lines + ["zeta"])                # 101 words, one-word tail
        self.assertEqual(tailed["msttr_segments"], 2)
        self.assertEqual(clean["msttr"], tailed["msttr"])

    def test_msttr_is_withheld_below_one_full_window(self):
        # Fewer than fifty words leaves no equal-size comparison to make, so a
        # number computed from one short sample is withheld rather than printed
        # as though it meant what the others mean.
        thin = profile(["alpha beta gamma"] * 5)          # 15 words
        self.assertIsNone(thin["msttr"])
        self.assertEqual(thin["msttr_segments"], 0)
        self.assertIsNotNone(thin["ttr"])

    def test_plain_ttr_is_reported_and_is_not_msttr(self):
        # MSTTR took plain TTR's key, so the column header still read TTR%
        # while the values came from a different scale. Both are reported now,
        # each under its own name, and they disagree by construction.
        lines = ["alpha beta gamma delta epsilon"] * 20
        row = profile(lines)
        # Five types over 100 words is 5%; over each 50-word window it is 10%.
        # That gap IS the sample-size dependence, and it is only visible
        # because the two numbers now have separate names.
        self.assertAlmostEqual(row["ttr"], 5.0)
        self.assertAlmostEqual(row["msttr"], 10.0)

    def test_the_table_names_both_columns(self):
        data = {"a": ["alpha beta gamma delta epsilon"] * 20}
        with redirect_stdout(io.StringIO()) as output:
            show("voices", data, 8, "test")
        printed = output.getvalue()
        self.assertIn("TTR%", printed)
        self.assertIn("MSTTR%", printed)

    def test_no_dialogue_is_reported_without_crashing(self):
        with redirect_stdout(io.StringIO()) as output:
            show("voices", {}, 8, "test")
        self.assertIn("nothing with at least 8 lines", output.getvalue())


class BundledFrequencyTable(unittest.TestCase):
    """The bundled Lexile source, which no default run exercises.

    ``word_frequency`` used to reference ``json`` in a module that never
    imported it.  The resulting NameError was not caught by the loader's
    ``except (OSError, ValueError)``, so it escaped ``measure`` and was
    swallowed one level up by the whole-analysis handler in ``grade.py``:
    turning Lexile on cost every core metric, not just Lexile, and the run
    still exited zero.  Anything non-default needs a test that runs it.
    """

    def setUp(self):
        prose_grade._FREQ_CACHE.clear()

    tearDown = setUp

    def test_every_frequency_source_loads_without_raising(self):
        for source in prose_grade.FREQUENCY_SOURCES:
            with self.subTest(source=source):
                self.assertIsInstance(prose_grade.word_frequency(source), dict)

    def test_bundled_table_is_the_rates_not_the_file_around_them(self):
        # The file is {"tokens": ..., "freq_per_million": {word: rate}}.
        # Reading the outer object gave a two-key dict, every lookup missed,
        # every word counted as rare and Lexile came back ~400 points high.
        table = prose_grade.word_frequency("bundled")
        self.assertGreater(len(table), 1000)
        self.assertGreater(table.get("the", 0), 0)

    def test_bundled_lexile_is_plausible_and_costs_no_other_metric(self):
        text = " ".join(["the quiet room held a long silence while she waited."] * 60)
        without = prose_grade.measure(text, floor=1, lexile_source="none")
        with_lexile = prose_grade.measure(text, floor=1, lexile_source="bundled")
        self.assertIsNone(without["lexile"])
        self.assertEqual(set(without), set(with_lexile))
        for key in without:
            if key != "lexile":
                self.assertEqual(without[key], with_lexile[key], key)
        self.assertTrue(200 < with_lexile["lexile"] < 1800, with_lexile["lexile"])

    def test_an_unreadable_table_costs_only_lexile(self):
        # A loader failure must degrade to a missing Lexile, never to a
        # missing core analysis.
        text = " ".join(["the quiet room held a long silence while she waited."] * 60)
        original = prose_grade.WORDFREQ
        broken = Path(tempfile.mkdtemp()) / "word_frequency.json"
        broken.write_text("{not json at all", encoding="utf-8")
        prose_grade.WORDFREQ = broken
        try:
            result = prose_grade.measure(text, floor=1, lexile_source="bundled")
        finally:
            prose_grade.WORDFREQ = original
        self.assertIsNone(result["lexile"])
        self.assertIsNotNone(result["fk"])


if __name__ == "__main__":
    unittest.main()
