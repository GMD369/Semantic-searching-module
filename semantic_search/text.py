"""Turning text into index terms. Shared by every model so they see the same words."""
from __future__ import annotations

import re

WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "been",
    "being", "but", "by", "can", "could", "did", "do", "does", "each", "for", "from", "get",
    "gets", "getting", "had", "has", "have", "he", "her", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "may", "more", "most", "my", "no", "nor", "not", "of", "on", "only", "or",
    "our", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "to", "too", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "why", "will", "with", "would", "you",
}


def stem(word: str) -> str:
    """Light suffix stripper: leaks/leakage -> leak, overheated/overheating -> overheat.

    Deliberately weaker than Porter. The point of this week is that meaning is
    learned from co-occurrence (LSI), not hand-written rules.
    """
    w = word
    for suffix, replacement, min_stem in (
        ("ations", "", 4), ("ation", "", 4), ("ings", "", 3), ("ing", "", 3),
        ("ages", "", 4), ("age", "", 4), ("ies", "y", 3),
        ("ed", "", 3), ("es", "", 4), ("s", "", 3),
    ):
        if w.endswith(suffix) and len(w) - len(suffix) >= min_stem and not w.endswith("ss"):
            w = w[: len(w) - len(suffix)] + replacement
            break
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz":   # running -> runn -> run
        w = w[:-1]
    if len(w) > 4 and w.endswith("e"):                          # cycle -> cycl
        w = w[:-1]
    return w


def raw_words(text: str) -> list[str]:
    """Lower-case words with stop words and pure numbers removed (not stemmed)."""
    return [w for w in WORD.findall(text.lower()) if w not in STOPWORDS and not w.isdigit()]


def terms(text: str) -> list[str]:
    """The index terms of a text: lower-case, stop words removed, stemmed."""
    return [stem(w) for w in raw_words(text)]


def word_spans(text: str):
    """Yield (start, end, stemmed term) for every non-stop word, for highlighting."""
    for match in WORD.finditer(text.lower()):
        w = match.group()
        if w not in STOPWORDS and not w.isdigit():
            yield match.start(), match.end(), stem(w)
