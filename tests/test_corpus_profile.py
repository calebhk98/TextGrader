import json
import tempfile
import unittest
from pathlib import Path

from corpus_profile import build_profile, load_profile, write_profile


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
