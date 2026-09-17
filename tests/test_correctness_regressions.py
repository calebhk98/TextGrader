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
        short = profile(["same"] * 50)["ttr"]
        long = profile(["same"] * 100)["ttr"]
        self.assertEqual(short, long)

    def test_no_dialogue_is_reported_without_crashing(self):
        with redirect_stdout(io.StringIO()) as output:
            show("voices", {}, 8, "test")
        self.assertIn("nothing with at least 8 lines", output.getvalue())


if __name__ == "__main__":
    unittest.main()
