"""Conservative Unicode-safe text handling for Adamawa Fulfulde."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_WORD_RE = re.compile(r"[^\W\d_]+(?:[\u02bc'\u2019-][^\W\d_]+)*", re.UNICODE)
_ORTHOGRAPHIC_APOSTROPHES = {"'", "\u02bc", "\u2019"}


@dataclass(frozen=True)
class TextAnalysis:
    """Normalized text and non-destructive review flags."""

    normalized: str
    flags: tuple[str, ...]
    unknown_script_characters: tuple[str, ...]


def normalize_text(text: str) -> str:
    """Apply NFC and collapse whitespace without changing orthography."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(unicodedata.normalize("NFC", text).split())


def words(text: str) -> tuple[str, ...]:
    """Return Unicode letter tokens while retaining internal apostrophes."""

    return tuple(_WORD_RE.findall(text))


def analyze_text(text: str) -> TextAnalysis:
    """Normalize text and flag items needing human review.

    Flags never cause transliteration or text replacement.
    """

    normalized = normalize_text(text)
    flags: list[str] = []

    if any(character.isdigit() for character in normalized):
        flags.append("contains_number")

    if any(len(token) >= 2 and token.isalpha() and token.isupper() for token in words(normalized)):
        flags.append("possible_abbreviation")

    unknown_scripts = sorted(
        {
            character
            for character in normalized
            if character.isalpha()
            and character not in _ORTHOGRAPHIC_APOSTROPHES
            and "LATIN" not in unicodedata.name(character, "")
        }
    )
    if unknown_scripts:
        flags.append("unknown_script")

    return TextAnalysis(
        normalized=normalized,
        flags=tuple(flags),
        unknown_script_characters=tuple(unknown_scripts),
    )
