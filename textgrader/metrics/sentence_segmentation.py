"""Optional sentence measurements using the rule-based pySBD segmenter."""

from .common import result, tokens


def measure(text, config=None, **_):
    config = config or {}
    try:
        import pysbd
    except ImportError:
        return [result("style.sentences_pysbd", "pySBD sentence count", None, "sentences",
                       warning="Install the optional 'pysbd' package to enable improved sentence segmentation")]

    language = config.get("language", "en")
    segmenter = pysbd.Segmenter(language=language, clean=bool(config.get("clean", False)))
    sentences = [sentence for sentence in segmenter.segment(text) if tokens(sentence)]
    lengths = [len(tokens(sentence)) for sentence in sentences]
    mean = sum(lengths) / len(lengths) if lengths else None
    return [
        result("style.sentences_pysbd", "pySBD sentence count", len(sentences), "sentences"),
        result("style.wps_pysbd", "Words per pySBD sentence", mean, "words/sentence"),
    ]
