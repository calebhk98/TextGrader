import json
import sys
import tempfile
import unittest
from pathlib import Path

import grade
from textgrader.results import StatusType


class StructuredGradeTests(unittest.TestCase):
    def test_one_metrics_map_controls_bundled_and_module_measures(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
        self.assertTrue(all(config["metrics"][name] is True for name in grade.BUNDLED_MEASURES))
        self.assertFalse(config["metrics"]["mattr"]["enabled"])

    def test_missing_profile_does_not_use_an_implicit_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            manuscript = Path(temporary) / "story.md"
            manuscript.write_text("One sentence. Another sentence follows.", encoding="utf-8")
            report = grade.analyze(manuscript, {"_config_dir": temporary,
                "corpus_profile": "corpus_profile.json",
                "metrics": {name: False for name in grade.BUNDLED_MEASURES}})
            fk = next(item for item in report.results if item.metric_id == "prose.fk")
            self.assertIsNone(fk.sample_size)
            self.assertIsNone(report.corpus_profile)

    def test_bundled_measure_can_run_as_structured_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            manuscript = Path(temporary) / "story.md"
            manuscript.write_text("There were 12 birds and 3 trees.", encoding="utf-8")
            report = grade.analyze(manuscript, {"_config_dir": temporary,
                "metrics": {name: name == "number_report" for name in grade.BUNDLED_MEASURES}})
            item = next(item for item in report.results if item.metric_id == "measure.number_report")
            self.assertEqual(item.status_type, StatusType.INFORMATIONAL)
            self.assertTrue(item.details and "output" in item.details[0])

    def test_bundled_measures_share_metric_toggles_and_default_on(self):
        with tempfile.TemporaryDirectory() as temporary:
            manuscript = Path(temporary) / "story.md"
            manuscript.write_text("There were 12 birds and 3 trees.", encoding="utf-8")
            enabled = grade.analyze(manuscript, {"_config_dir": temporary})
            self.assertTrue(any(item.metric_id.startswith("measure.") for item in enabled.results))
            disabled = grade.analyze(manuscript, {"_config_dir": temporary,
                "metrics": {name: False for name in grade.BUNDLED_MEASURES}})
            self.assertFalse(any(item.metric_id.startswith("measure.") for item in disabled.results))

    def test_value_differing_from_zero_mad_corpus_is_an_outlier(self):
        profile = {"corpus_name": "flat", "distributions": {
            "wps": {"values": [10.0, 10.0, 10.0]}}}
        result = grade._corpus_result("wps", 11.0, profile)
        self.assertEqual(result.status_type, StatusType.CORPUS_OUTLIER)
        self.assertIsNone(result.corpus["robust_distance"])

    def test_crashed_child_is_counted_as_internal_error(self):
        result = grade.run_metric_process("fixture.crash", [sys.executable, "-c", "raise RuntimeError('boom')"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].metric_id, "fixture.crash")
        self.assertEqual(result[0].status_type, StatusType.INTERNAL_ERROR)
        self.assertIn("exited", result[0].error)

    def test_invalid_child_json_is_an_internal_error(self):
        result = grade.run_metric_process("fixture.invalid", [sys.executable, "-c", "print('PASS words can change')"])
        self.assertEqual(result[0].status_type, StatusType.INTERNAL_ERROR)

    def test_json_ids_and_accounting_are_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            manuscript = Path(temporary) / "story.md"
            manuscript.write_text("# Heading\n\nOne small sentence. Another longer sentence follows here.", encoding="utf-8")
            report = grade.analyze(manuscript, {"_config_dir": temporary, "project_rules": {}})
            payload = report.to_dict()
            self.assertEqual(payload["summary"]["total"], len(payload["results"]))
            self.assertTrue(all(item["metric_id"] for item in payload["results"]))
            self.assertEqual(json.loads(json.dumps(payload, sort_keys=True)), payload)

    def test_project_rules_disabled_then_explicitly_enabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            manuscript = Path(temporary) / "story.md"
            manuscript.write_text("I noticed the door. It opened.", encoding="utf-8")
            empty = grade.analyze(manuscript, {"_config_dir": temporary, "project_rules": {}})
            self.assertFalse(any(r.status_type == StatusType.PROJECT_RULE for r in empty.results))
            configured = grade.analyze(manuscript, {"_config_dir": temporary, "project_rules": {
                "banned_phrases": [{"id": "noticed", "name": "Noticed", "pattern": r"\bI noticed\b"}]
            }})
            self.assertEqual([r.metric_id for r in configured.results if r.status_type == StatusType.PROJECT_RULE],
                             ["rule.noticed"])


if __name__ == "__main__":
    unittest.main()
