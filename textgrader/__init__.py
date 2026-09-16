"""Shared building blocks for TextGrader."""

from .text import (
    DialogueSplit,
    TranscriptConfig,
    normalize_quotes,
    paragraphs,
    parse_quotations,
    remove_markdown_headings,
    sentences,
    split_dialogue,
    strip_gutenberg,
    strip_transcript,
    words,
)
from .results import MetricResult, Report, StatusType

__all__ = [
    "DialogueSplit",
    "TranscriptConfig",
    "normalize_quotes",
    "paragraphs",
    "parse_quotations",
    "remove_markdown_headings",
    "sentences",
    "split_dialogue",
    "strip_gutenberg",
    "strip_transcript",
    "words",
    "MetricResult",
    "Report",
    "StatusType",
]
