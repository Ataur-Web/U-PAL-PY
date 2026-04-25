"""emotion detection. 5 states + neutral, in both English and Welsh.

we use a simple keyword approach instead of a trained model because:
  - the training data would be tiny (student chat is niche)
  - we want it predictable so the tone-matching prompt is defensible in the viva
  - the downstream use is just "soften the reply if the user sounds stressed",
    so we don't need high precision

labels returned:
  "distressed" | "stressed" | "frustrated" | "confused" | "excited" | "neutral"

ref: Ekman, P. (1992) An argument for basic emotions, Cognition & Emotion 6
ref: keyword lists adapted from u-pal-rag/lib/nlp.js
"""
from __future__ import annotations

import re
from typing import Any


# order matters, we check "distressed" first so safeguarding keywords take
# priority over mere stress keywords.
_KEYWORDS: list[tuple[str, re.Pattern]] = [
    ("distressed", re.compile(
        r"\b(can't\s+cope|want\s+to\s+give\s+up|suicid|self[-\s]?harm|"
        r"breakdown|panic\s+attack|crisis|hopeless|methu\s+ymdopi|argyfwng)\b",
        re.I,
    )),
    ("stressed", re.compile(
        r"\b(stressed|stressing|overwhelmed|anxious|anxiety|burn[-\s]?out|"
        r"exhausted|can't\s+sleep|dissertation|deadline|exam\s+stress|"
        r"gorlethu|straen|pryder)\b",
        re.I,
    )),
    ("frustrated", re.compile(
        r"\b(frustrated|annoyed|fed\s+up|useless|terrible|awful|stupid|"
        r"broken|not\s+working|trash|rubbish|hate\s+this|rhwystredig|wedi\s+cael\s+llond\s+bol)\b",
        re.I,
    )),
    ("confused", re.compile(
        r"\b(confused|don't\s+understand|what\s+does\s+that\s+mean|"
        r"lost|makes\s+no\s+sense|can\s+you\s+explain|dw[iy]?\s+ddim\s+yn\s+deall|dryslyd)\b",
        re.I,
    )),
    ("excited", re.compile(
        r"\b(excited|can't\s+wait|amazing|awesome|brilliant|love\s+it|"
        r"so\s+happy|cyffrous|gwych|anhygoel)\b",
        re.I,
    )),
]


def detect(
    message: str,
    history: list[dict[str, Any]] | None = None,
    lang: str = "en",              # noqa: ARG001  (reserved for future CY-specific tuning)
) -> str:
    if not message:
        return "neutral"

    # check the current message first, it's the strongest signal
    for label, rx in _KEYWORDS:
        if rx.search(message):
            return label

    # short follow-ups like "ok" or "yeah" inherit the mood of the last
    # user turn. we only look at the last 3 turns to keep noise down.
    if history:
        for turn in reversed(history[-3:]):
            if not isinstance(turn, dict) or turn.get("role") != "user":
                continue
            prev = turn.get("text") or ""
            for label, rx in _KEYWORDS:
                if rx.search(prev):
                    return label

    return "neutral"
