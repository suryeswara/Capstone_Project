"""BioScope-style certainty levels (1 speculative, 2 indicative, 3 definitive).

Carried over unchanged from the previous implementation so that the
certainty-inflation guard behaves the same way.
"""

import re

HEDGE_CUES = {
    "weak": [
        "may", "might", "could", "suggests", "associated with", "correlated with",
        "possible", "potential", "preliminary", "appears to", "trend towards",
        "observed in vitro", "hypothesized", "tentative",
    ],
    "moderate": [
        "indicates", "shows", "demonstrates", "supports", "likely",
        "reduces risk of", "increases risk of", "consistent with",
        "evidence suggests", "data supports", "improves",
    ],
    "strong": [
        "causes", "cures", "prevents", "proves", "proven to", "definitively",
        "eradicates", "guarantees", "eliminates", "direct cause",
        "has been established", "conclusively",
    ],
}


def _has_cue(text_lower: str, cue: str) -> bool:
    return re.search(r"\b" + re.escape(cue) + r"\b", text_lower) is not None


def get_certainty_level(text: str) -> int:
    """Highest certainty level expressed in ``text`` (default 2 when no cue)."""
    t = (text or "").lower()
    for cue in HEDGE_CUES["strong"]:
        if _has_cue(t, cue):
            return 3
    for cue in HEDGE_CUES["moderate"]:
        if _has_cue(t, cue):
            return 2
    for cue in HEDGE_CUES["weak"]:
        if _has_cue(t, cue):
            return 1
    return 2
