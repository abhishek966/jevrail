from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str


_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_API_KEY = re.compile(
    r"(?:sk-[A-Za-z0-9]{20,}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|Bearer\s+[A-Za-z0-9\-._~+/]{20,}=*)"
)
_PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")


def _luhn_ok(raw: str) -> bool:
    digits = [int(char) for char in raw if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def find_sensitive(text: str, labels: frozenset[str] | set[str] | None = None) -> list[Span]:
    """Find structured PII and secrets. ``labels`` limits which kinds are returned."""
    wanted = set(labels) if labels is not None else None
    spans: list[Span] = []

    def keep(label: str) -> bool:
        return wanted is None or label in wanted

    if keep("email"):
        spans.extend(Span(match.start(), match.end(), "email") for match in _EMAIL.finditer(text))
    if keep("ssn"):
        spans.extend(Span(match.start(), match.end(), "ssn") for match in _SSN.finditer(text))
    if keep("credit_card"):
        for match in _CARD.finditer(text):
            if _luhn_ok(match.group()):
                spans.append(Span(match.start(), match.end(), "credit_card"))
    if keep("api_key"):
        spans.extend(Span(match.start(), match.end(), "api_key") for match in _API_KEY.finditer(text))
    if keep("private_key"):
        spans.extend(
            Span(match.start(), match.end(), "private_key") for match in _PRIVATE_KEY.finditer(text)
        )
    spans.sort(key=lambda span: (span.start, span.end))
    return spans


def redact(text: str, spans: list[Span], token: str = "[REDACTED]") -> str:
    if not spans:
        return text
    ordered = sorted(spans, key=lambda span: span.start)
    merged: list[tuple[int, int]] = []
    for span in ordered:
        if not merged or span.start > merged[-1][1]:
            merged.append((span.start, span.end))
        else:
            start, end = merged[-1]
            merged[-1] = (start, max(end, span.end))
    pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        pieces.append(text[cursor:start])
        pieces.append(token)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)
