import json
import tempfile
import unittest
from pathlib import Path

from textgrader.corpus import build_profile, load_profile, write_profile


class CorpusProfileTests(unittest.TestCase):
    def test_core_prose_distributions_are_built_by_default(self):
        expected = {"fk", "ari", "slcv", "spp", "long7", "sttr", "top100",
                    "commas", "subord", "relcl", "simple", "u10", "b2035",
                    "shortruns", "front", "and2", "andrate", "negative"}
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "book.txt"
            source.write_text(("Although it rained, we stayed. And then we left! " * 125), encoding="utf-8")
            profile = build_profile([source], built_at="2026-01-01T00:00:00Z")
        self.assertTrue(expected <= profile["distributions"].keys())
        self.assertTrue(all(profile["distributions"][key]["count"] == 1 for key in expected))

    def test_core_prose_distributions_can_be_disabled_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "book.txt"
            source.write_text("A sentence.", encoding="utf-8")
            profile = build_profile([source], include_core_metrics=False)
        self.assertNotIn("fk", profile["distributions"])

    def test_duplicate_filenames_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first", root / "second"
            first.mkdir(); second.mkdir()
            (first / "book.txt").write_text("First book.", encoding="utf-8")
            (second / "book.txt").write_text("A different second book.", encoding="utf-8")
            profile = build_profile([first, second], built_at="2026-01-01T00:00:00Z")
            self.assertEqual(profile["book_count"], 2)
            self.assertEqual(len({book["source_id"] for book in profile["books"]}), 2)
            self.assertEqual([book["source_filename"] for book in profile["books"]], ["book.txt", "book.txt"])

    def test_gutenberg_boilerplate_is_not_profiled(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "book.txt"
            source.write_text("Header license\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
                              "Actual story words.\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\n"
                              "Footer license", encoding="utf-8")
            profile = build_profile([source], built_at="2026-01-01T00:00:00Z")
            self.assertEqual(profile["word_frequency"], {"actual": 1, "story": 1, "words": 1})

    def test_rebuild_is_byte_reproducible_and_loadable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "story.txt"
            source.write_text("One sentence. Another sentence!", encoding="utf-8")
            kwargs = {"corpus_name": "fixture", "built_at": "2026-01-01T00:00:00Z"}
            first = build_profile([source], **kwargs)
            second = build_profile([source], **kwargs)
            one, two = root / "one.json", root / "two.json"
            write_profile(first, one); write_profile(second, two)
            self.assertEqual(one.read_bytes(), two.read_bytes())
            self.assertEqual(load_profile(one), first)
            self.assertTrue(first["books"][0]["source_hash"].startswith("sha256:"))

    def test_empty_corpus_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "no .txt"):
                build_profile([temporary])


if __name__ == "__main__":
    unittest.main()


class ProfileVectorAlignment(unittest.TestCase):
    """``feature_profiles[name][i]`` must describe ``books[i]``.

    A metric caches a per-book vector by exposing ``profile_vector``.  Appending
    only when one comes back shifts every later row when a single book fails,
    so a consumer indexing rows against books gets a confident wrong answer --
    the nearest reference book would be the wrong book -- instead of a missing
    one.
    """

    MODULE = Path(__file__).resolve().parents[1] / "textgrader" / "metrics" / "_align_probe.py"
    SOURCE = '''from .common import finding
FAMILY = "lexical"
COST = "fast"
REQUIRES = ()


def measure(analysis, config=None, profile=None):
    return [finding("style.align_probe", "Probe", float(analysis.word_count), "words",
                    family=FAMILY)]


def profile_vector(analysis, config=None):
    if "FAILING" in analysis.text:
        raise RuntimeError("this book's vector cannot be built")
    return {"d0": float(analysis.word_count)}
'''

    def setUp(self):
        from textgrader.metrics import REGISTRY, MetricSpec
        self.MODULE.write_text(self.SOURCE, encoding="utf-8")
        REGISTRY["_align_probe"] = MetricSpec("_align_probe", "_align_probe", "lexical", "fast")

    def tearDown(self):
        from textgrader.metrics import REGISTRY
        REGISTRY.pop("_align_probe", None)
        self.MODULE.unlink(missing_ok=True)

    def _profile(self, tags):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index, tag in enumerate(tags):
                (directory / f"{index}_book.txt").write_text(
                    f"{tag} words here. More text follows now.\n\nA second paragraph.\n" * 25,
                    encoding="utf-8")
            return build_profile([directory], built_at="2026-01-01T00:00:00Z",
                                 metrics={"_align_probe": {"enabled": True}})

    def test_a_failing_book_leaves_a_placeholder_rather_than_shifting_rows(self):
        profile = self._profile(["FIRST", "FAILING", "LAST"])
        rows = profile["feature_profiles"]["_align_probe"]
        self.assertEqual(len(rows), len(profile["books"]))
        self.assertEqual(rows[1], {})
        self.assertTrue(rows[0] and rows[2])

    def test_a_failing_last_book_is_padded_too(self):
        profile = self._profile(["FIRST", "SECOND", "FAILING"])
        rows = profile["feature_profiles"]["_align_probe"]
        self.assertEqual(len(rows), len(profile["books"]))
        self.assertEqual(rows[-1], {})

    def test_the_failure_is_recorded_rather_than_swallowed(self):
        profile = self._profile(["FIRST", "FAILING"])
        errors = json.dumps(profile.get("metric_errors") or profile.get("errors") or {})
        self.assertIn("_align_probe", errors)
