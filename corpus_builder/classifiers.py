"""Lightweight, auditable document classifiers."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
THIRD_PERSON = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "they", "them", "their", "theirs", "themselves"}
PAST_MARKERS = {"was", "were", "had", "did", "said", "went", "came", "saw", "thought", "knew", "looked", "seemed", "began", "found", "felt", "took", "made", "told", "asked", "answered", "stood", "sat", "gave", "left", "heard"}
PRESENT_MARKERS = {"is", "are", "has", "does", "says", "goes", "comes", "sees", "thinks", "knows", "looks", "seems", "begins", "finds", "feels", "takes", "makes", "tells", "asks", "answers", "stands", "sits", "gives", "leaves", "hears"}


def narration_without_quotes(text: str) -> str:
    out: list[str] = []
    in_quote = False
    closing: str | None = None
    for char in text:
        if char == "“":
            in_quote, closing = True, "”"
            out.append(" ")
        elif char == "”" and closing == "”":
            in_quote, closing = False, None
            out.append(" ")
        elif char == '"':
            in_quote = not in_quote
            closing = '"' if in_quote else None
            out.append(" ")
        elif not in_quote or char == "\n":
            out.append(char)
        else:
            out.append(" ")
    return "".join(out)


def _words(text: str) -> list[str]:
    return [word.casefold().replace("’", "'") for word in WORD_RE.findall(narration_without_quotes(text))]


def classify_pov(text: str) -> dict[str, Any]:
    counts = Counter(_words(text))
    first, third = sum(counts[w] for w in FIRST_PERSON), sum(counts[w] for w in THIRD_PERSON)
    total = first + third
    if total < 40:
        return {"label": "unknown", "confidence": 0.0, "first": first, "third": third, "sample": total}
    share = first / total
    confidence = abs(share - 0.5) * 2 * min(1.0, total / 400)
    label = "first" if share >= 0.62 else "third" if share <= 0.38 else "mixed"
    return {"label": label, "confidence": round(confidence, 3), "first_share": round(share, 4), "first": first, "third": third, "sample": total}


def classify_tense(text: str) -> dict[str, Any]:
    words = _words(text)
    counts = Counter(words)
    past, present = sum(counts[w] for w in PAST_MARKERS), sum(counts[w] for w in PRESENT_MARKERS)
    past += min(sum(1 for w in words if len(w) > 4 and w.endswith("ed")), max(50, past * 2))
    total = past + present
    if total < 60:
        return {"label": "unknown", "confidence": 0.0, "past": past, "present": present, "sample": total}
    share = past / total
    confidence = abs(share - 0.5) * 2 * min(1.0, total / 600)
    label = "past" if share >= 0.62 else "present" if share <= 0.38 else "mixed"
    return {"label": label, "confidence": round(confidence, 3), "past_share": round(share, 4), "past": past, "present": present, "sample": total}


def classification_allowed(result: dict[str, Any], wanted: str, confidence: float) -> bool:
    return wanted == "any" or (result["label"] == wanted and float(result["confidence"]) >= confidence)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))
