import json
import sys
import tempfile
import unittest
from pathlib import Path

import grade
from textgrader.results import StatusType


class StructuredGradeTests(unittest.TestCase):
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
