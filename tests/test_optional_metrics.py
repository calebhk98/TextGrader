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

    def test_profile_uses_and_records_metric_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "book.txt"
            source.write_text("one two three four five six.", encoding="utf-8")
            metrics = {"mattr": {"enabled": True, "window": 3}}
            profile = build_profile([source], metrics=metrics)
            self.assertEqual(profile["metric_settings"], metrics)
            self.assertAlmostEqual(profile["distributions"]["style.mattr"]["values"][0], 100.0)

    def test_mismatched_metric_options_are_not_compared(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "book.txt"
            source.write_text("one two three one two three.", encoding="utf-8")
            profile = build_profile([source], metrics={"mattr": {"window": 100}})
            (root / "profile.json").write_text(__import__("json").dumps(profile), encoding="utf-8")
            report = grade.analyze(source, {"_config_dir": temporary,
                "corpus_profile": "profile.json",
                "metrics": {"mattr": {"enabled": True, "window": 3}}})
            item = next(item for item in report.results if item.metric_id == "style.mattr")
            self.assertEqual(item.status_type.value, "informational")
            self.assertIn("different", item.warning)

    def test_external_commands_require_explicit_trust(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "story.txt"
            source.write_text("One sentence.", encoding="utf-8")
            report = grade.analyze(source, {"_config_dir": temporary,
                "metric_commands": [{"id": "unsafe", "command": ["false"]}]})
            item = next(item for item in report.results if item.metric_id == "metric.external_commands")
            self.assertEqual(item.status_type.value, "unavailable")

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
