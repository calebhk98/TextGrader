"""The per-sentence noun-chunk and entity index matches spaCy exactly.

``Span.noun_chunks`` walks the whole document for every sentence, which made
the coherence and graph suites' entity extraction quadratic in book length
(300 seconds on Treasure Island).  The index must stay a drop-in replacement.
"""

import pytest

from textgrader import optional
from textgrader.spans import sentence_ents, sentence_noun_chunks

TEXT = ("Captain Smollett met Jim Hawkins in Bristol on Monday. The old sailor "
        "carried a heavy sea chest up the hill. Long John Silver laughed at the "
        "young boy. Nobody in the Admiral Benbow inn slept that night.")


@pytest.fixture(scope="module")
def doc():
    spacy, reason = optional.require("spacy")
    if spacy is None:
        pytest.skip(reason)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as exc:  # model not downloaded
        pytest.skip(str(exc))
    return nlp(TEXT)


def test_sentence_noun_chunks_match_spacy(doc):
    for sent in doc.sents:
        assert [(c.start, c.end) for c in sentence_noun_chunks(sent)] == \
               [(c.start, c.end) for c in sent.noun_chunks]
    assert any(sentence_noun_chunks(sent) for sent in doc.sents)


def test_sentence_ents_match_spacy(doc):
    for sent in doc.sents:
        assert [(e.start, e.end, e.label_) for e in sentence_ents(sent)] == \
               [(e.start, e.end, e.label_) for e in sent.ents]


def test_the_index_is_built_once_per_document(doc):
    first = list(doc.sents)[0]
    sentence_noun_chunks(first)
    index = doc.user_data["textgrader.noun_chunks_by_sentence"]
    sentence_noun_chunks(list(doc.sents)[-1])
    assert doc.user_data["textgrader.noun_chunks_by_sentence"] is index
