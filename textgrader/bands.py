"""Reading-level targets that move across a book.

A single whole-book target cannot express a book whose reading level is
*supposed* to change - a coming-of-age novel whose protagonist ages, a
textbook, a tutorial series, a graded reader, documentation running from
quickstart to reference. Worse, a flat target actively hides the change: a
manuscript measured here reads Flesch-Kincaid 6.8 against a target of 7.0,
which looks slightly under and fine, while its last fourteen chapters average
8.85 against a floor of 7.2 with three of them between 10.8 and 11.2. The flat
number averages a real structural problem into invisibility.

Two details are load-bearing, both learned the hard way in the tool this came
from.

The BAND AVERAGE is what gets judged, not each chapter. An earlier version
judged the band average and then also failed the band whenever its own
footnote listed a chapter below the floor, which made two bands permanently
unfixable: the average cleared, a chapter did not, and nothing a writer could
do satisfied both. The chapters below the floor are listed because they say
where to look, and they do not themselves fail the band.

A band may carry a CEILING as well as a floor. The original had floors only,
which is why nothing in it flagged three chapters running at Flesch-Kincaid 11
in a book aimed at ninth graders. A target that can only be undershot is half
a target.
"""

from __future__ import annotations

import statistics


class BandError(ValueError):
    """A band specification that cannot be read."""


def parse_range(text):
    """``"11-15"`` or ``"7"`` -> the inclusive set of chapter numbers."""

    if isinstance(text, int):
        return {text}
    parts = str(text).split("-")
    try:
        if len(parts) == 1:
            return {int(parts[0])}
        if len(parts) == 2:
            first, last = int(parts[0]), int(parts[1])
            if last < first:
                raise BandError(f"band {text!r} runs backwards")
            return set(range(first, last + 1))
    except ValueError as exc:
        raise BandError(f"band {text!r} is not a chapter number or range") from exc
    raise BandError(f"band {text!r} is not a chapter number or range")


def compile_bands(settings):
    """Validate one ``chapter_bands`` block, returning ``(metric, bands, problems)``.

    Problems are returned rather than raised, because a misconfigured band
    should be a visible result like everything else here, not a crash that
    costs the rest of the run.
    """

    problems = []
    if not isinstance(settings, dict):
        return None, [], ["chapter_bands must be an object"]
    metric = settings.get("metric")
    if not metric:
        problems.append("chapter_bands.metric is required (for example \"fk\")")
    raw = settings.get("bands")
    if not isinstance(raw, list) or not raw:
        problems.append("chapter_bands.bands must be a non-empty list")
        return metric, [], problems
    bands, claimed = [], {}
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            problems.append(f"band {index + 1} must be an object")
            continue
        try:
            numbers = parse_range(entry.get("chapters", ""))
        except BandError as exc:
            problems.append(str(exc))
            continue
        if entry.get("floor") is None and entry.get("ceiling") is None:
            problems.append(f"band {entry.get('chapters')!r} sets neither floor nor ceiling")
            continue
        floor, ceiling = entry.get("floor"), entry.get("ceiling")
        if floor is not None and ceiling is not None and ceiling < floor:
            problems.append(f"band {entry.get('chapters')!r} has a ceiling below its floor")
            continue
        # Overlapping bands would put a chapter in two places and judge it
        # twice, which reads as two findings for one problem.
        for number in sorted(numbers & set(claimed)):
            problems.append(f"chapter {number} is in both {claimed[number]!r} "
                            f"and {entry.get('chapters')!r}")
        for number in numbers:
            claimed.setdefault(number, entry.get("chapters"))
        bands.append({"chapters": entry.get("chapters"), "numbers": numbers,
                      "floor": floor, "ceiling": ceiling,
                      "exempt": entry.get("exempt", {}) or {}})
    return metric, bands, problems


def judge(bands, values, exempt=None):
    """Judge each band's average against its floor and ceiling.

    ``values`` maps a chapter number to that chapter's measured value.
    ``exempt`` maps a chapter number to a documented reason; an exempt chapter
    is left out of the average and reported, so an exemption is a decision on
    the record rather than a number quietly going missing.
    """

    exempt = exempt or {}
    out = []
    for band in bands:
        band_exempt = {**exempt, **{int(key): reason
                                    for key, reason in band["exempt"].items()}}
        members = sorted(number for number in band["numbers"] if number in values)
        counted = [number for number in members if number not in band_exempt]
        excused = [number for number in members if number in band_exempt]
        if not counted:
            out.append({**_describe(band), "status": "insufficient_data",
                        "average": None, "chapters": members, "excused": excused,
                        "outside": [], "reason": "no chapter in this band was measured"
                                    if not members else
                                    "every chapter in this band is exempt"})
            continue
        average = statistics.fmean(values[number] for number in counted)
        under = band["floor"] is not None and average < band["floor"]
        over = band["ceiling"] is not None and average > band["ceiling"]
        # The chapters below the floor are a pointer to where to look. They do
        # NOT fail the band on their own: judging the average and then also
        # failing on any single chapter made bands that no revision could
        # satisfy.
        if band["floor"] is not None:
            outside = [number for number in counted if values[number] < band["floor"]]
        else:
            outside = []
        if band["ceiling"] is not None:
            outside += [number for number in counted
                        if values[number] > band["ceiling"] and number not in outside]
        out.append({**_describe(band),
                    "status": "under" if under else "over" if over else "within",
                    "average": round(average, 2), "chapters": counted,
                    "excused": excused, "outside": sorted(outside),
                    "reason": None})
    return out


def _describe(band):
    return {"band": band["chapters"], "floor": band["floor"], "ceiling": band["ceiling"]}
