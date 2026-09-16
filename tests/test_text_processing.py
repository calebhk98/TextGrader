import unittest

from textgrader.text import (
    TranscriptConfig,
    normalize_quotes,
    paragraphs,
    remove_markdown_headings,
    sentences,
    split_dialogue,
    strip_gutenberg,
    strip_transcript,
    transcript_lines,
    words,
)


class TextProcessingTests(unittest.TestCase):
    def test_unicode_words_and_apostrophes(self):
        self.assertEqual(words("na\u00efve l'\u00e9t\u00e9 O\u2019Brien 42 snake_case"),
                         ["na\u00efve", "l'\u00e9t\u00e9", "O\u2019Brien", "snake", "case"])

    def test_paragraph_and_sentence_segmentation(self):
        text = 'One soft\nline. "Two?"\n\nThree! Tail'
        self.assertEqual(paragraphs(text), ['One soft line. "Two?"', "Three! Tail"])
        self.assertEqual(sentences(text), ["One soft line.", '"Two?"', "Three!", "Tail"])

    def test_markdown_headings(self):
        self.assertEqual(remove_markdown_headings("# Title\nText\n### Part\nSubhead\n---\nMore"),
                         "Text\nMore")

    def test_quote_normalization(self):
        self.assertEqual(normalize_quotes("\u201cIt\u2019s so,\u201d"), '"It\'s so,"')

    def test_gutenberg_header_and_footer(self):
        source = "license boilerplate\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nStory.\n*** END OF THIS PROJECT GUTENBERG EBOOK X ***\nlicense"
        self.assertEqual(strip_gutenberg(source), "Story.")

    def test_multiline_curly_and_straight_dialogue(self):
        text = 'Before \u201cfirst\nsecond\u201d between "third" after.'
        result = split_dialogue(text)
        self.assertEqual(result.quotations, ("first\nsecond", "third"))
        self.assertNotIn("first", result.narrated)
        self.assertIn("Before", result.narrated)

    def test_long_dialogue_is_not_discarded(self):
        content = "word " * 1000
        self.assertEqual(len(split_dialogue(f'"{content}"').quotations[0]), len(content))

    def test_single_quote_dialogue_reports_limitation(self):
        result = split_dialogue("\u2018This is speech.\u2019 Narration.")
        self.assertTrue(result.warnings)
        self.assertFalse(result.quotations)

    def test_transcript_usernames_have_no_length_limit(self):
        text = "a: short\nvery_long_username_123: long\nordinary prose"
        self.assertEqual(len(transcript_lines(text)), 2)
        stripped, share = strip_transcript(text)
        self.assertEqual(stripped, "ordinary prose")
        self.assertGreater(share, 0)

    def test_transcript_pattern_is_configurable(self):
        config = TranscriptConfig(pattern=r"^\[(?P<username>[^]]+)\] (?P<message>.*)$")
        self.assertEqual(len(transcript_lines("[Jane Doe] Hello", config)), 1)

    def test_empty_input_is_safe(self):
        self.assertEqual(strip_transcript(""), ("", 0.0))
        self.assertEqual(split_dialogue("").spoken, "")


if __name__ == "__main__":
    unittest.main()
