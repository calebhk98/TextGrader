import tempfile
import unittest
from pathlib import Path

import grade
from textgrader.corpus import build_profile


class OptionalMetricTests(unittest.TestCase):
    def test_all_are_off_unless_enabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "story.txt"
            source.write_text("He walked home. He walked home.", encoding="utf-8")
            report = grade.analyze(source, {"_config_dir": temporary, "metrics": {}})
            self.assertFalse(any(item.metric_id.startswith(("style.", "nlp."))
                                 for item in report.results))

    def test_enabled_metrics_are_structured_and_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "story.txt"
            source.write_text("He walked to the old house. He walked to the old house.", encoding="utf-8")
            config = {"_config_dir": temporary, "metrics": {
                "repeated_ngrams": {"enabled": True}, "mattr": {"enabled": True}}}
            ids = {item.metric_id for item in grade.analyze(source, config).results}
            self.assertIn("style.repeated_ngrams", ids)
            self.assertIn("style.mattr", ids)

    def test_profile_contains_advanced_comparison_distributions(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "book.txt"
            source.write_text("One short line. Another longer line follows it.", encoding="utf-8")
            profile = build_profile([source], built_at="2026-01-01T00:00:00Z")
            self.assertIn("style.mattr", profile["distributions"])
            self.assertEqual(len(profile["feature_profiles"]["function_words"]), 1)

    def test_nlp_metric_without_model_is_a_visible_warning(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "story.txt"
            source.write_text("The door was opened.", encoding="utf-8")
            report = grade.analyze(source, {"_config_dir": temporary,
                "metrics": {"passive_voice": {"enabled": True}},
                "nlp": {"model": "certainly_missing_model"}})
            item = next(item for item in report.results if item.metric_id == "nlp.passive_voice")
            self.assertIsNotNone(item.warning)


if __name__ == "__main__":
    unittest.main()
