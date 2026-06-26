"""Cache-bust linter.

The culprit behind low cache adoption is usually prompt *layout*: dynamic
content injected too early, or blocks of state that are supposed to be stable
getting reordered or rewritten between requests, breaking the prefix reuse that
facilitates caching.

A block you *believe* is static is worthless for caching if it secretly carries
volatile content -- a timestamp, a request ID, a UUID, today's date. One such
token near the front of a "stable" prefix changes the prefix hash on every call
and silently drops your cache-hit rate to zero. Providers don't warn you; the
tokens just quietly bill at full price.

This module is the automated tooling that gap calls for. It does two things:

1. ``lint_segment`` / ``lint_prompt`` -- static analysis: scan a block marked
   static for embedded volatile patterns and report each with a severity and
   character offset (earlier = worse, because it invalidates a longer prefix).
2. ``CacheBustDetector`` -- dynamic analysis: feed it the rendered static
   blocks from successive requests; it flags any "static" block whose content
   actually changed between calls (the real-world cache-busting signal).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .planner import Segment

# Patterns that almost always indicate per-request volatile content sitting in
# a block the author thinks is stable. Each is (name, severity, compiled regex).
_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    ("uuid", "high", re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")),
    ("iso_datetime", "high", re.compile(
        r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?\b")),
    ("iso_date", "medium", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("us_date", "medium", re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    ("clock_time", "medium", re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\s?(AM|PM|am|pm)?\b")),
    ("epoch_millis", "high", re.compile(r"\b1[0-9]{12}\b")),
    ("epoch_seconds", "high", re.compile(r"\b1[0-9]{9}\b")),
    ("long_hex_id", "high", re.compile(r"\b[0-9a-f]{24,}\b")),
    ("labeled_id", "high", re.compile(
        r"\b(?:request|trace|session|correlation|conversation|message|run|job|"
        r"user|order|ticket)[ _-]?id\s*[:=]\s*\S+", re.I)),
    ("bearer_token", "high", re.compile(r"\b(?:Bearer\s+|sk-)[A-Za-z0-9._-]{12,}")),
    ("counter_phrase", "low", re.compile(
        r"\b(?:attempt|retry|iteration|turn|step)\s*#?\s*\d+\b", re.I)),
    ("relative_now", "medium", re.compile(
        r"\b(?:today is|current (?:date|time)|right now|as of)\b", re.I)),
]


@dataclass
class Finding:
    segment: str
    pattern: str
    severity: str
    offset: int          # char offset within the segment (lower = worse)
    snippet: str

    def __str__(self) -> str:
        return (f"[{self.severity.upper():6}] {self.segment}@{self.offset}: "
                f"{self.pattern} -> {self.snippet!r}")


@dataclass
class LintReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def worst_offset(self) -> int | None:
        """Earliest offending offset across segments (the prefix it kills)."""
        return min((f.offset for f in self.findings), default=None)

    def by_severity(self, level: str) -> list[Finding]:
        return [f for f in self.findings if f.severity == level]

    def render(self) -> str:
        if self.ok:
            return "No cache-busting content found in static blocks."
        lines = [f"{len(self.findings)} cache-bust risk(s) found:"]
        order = {"high": 0, "medium": 1, "low": 2}
        for f in sorted(self.findings, key=lambda x: (order[x.severity], x.offset)):
            lines.append("  " + str(f))
        if self.worst_offset == 0 or (self.worst_offset or 99) < 40:
            lines.append("  ! A risk sits at the very start of a static block; "
                         "it invalidates the ENTIRE cacheable prefix.")
        return "\n".join(lines)


def lint_text(text: str, segment_name: str = "static") -> list[Finding]:
    findings: list[Finding] = []
    for name, severity, rx in _PATTERNS:
        for m in rx.finditer(text):
            findings.append(Finding(
                segment=segment_name, pattern=name, severity=severity,
                offset=m.start(),
                snippet=m.group(0)[:48]))
    return findings


def lint_segment(seg: Segment) -> list[Finding]:
    """Only static segments matter -- volatile blocks are *expected* to change."""
    if not seg.static:
        return []
    return lint_text(seg.text, seg.name)


def lint_prompt(segments: list[Segment]) -> LintReport:
    report = LintReport()
    for seg in segments:
        report.findings.extend(lint_segment(seg))
    return report


class CacheBustDetector:
    """Dynamic check: did a 'static' block actually change between requests?

    Record the rendered static blocks each call. If a block's content hash
    differs from the previous call, its cache prefix was busted in production --
    the classic symptom of "stable" blocks being rewritten between requests.
    """

    def __init__(self) -> None:
        self._last: dict[str, str] = {}
        self.busts: list[tuple[int, str]] = []  # (call_index, segment_name)
        self._calls = 0

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def record(self, segments: list[Segment]) -> list[str]:
        """Record one request's static blocks; return names that changed."""
        self._calls += 1
        changed: list[str] = []
        for seg in segments:
            if not seg.static:
                continue
            h = self._hash(seg.text)
            prev = self._last.get(seg.name)
            if prev is not None and prev != h:
                changed.append(seg.name)
                self.busts.append((self._calls, seg.name))
            self._last[seg.name] = h
        return changed

    @property
    def bust_rate(self) -> float:
        """Fraction of recorded calls (after the first) that busted a cache."""
        denom = max(0, self._calls - 1)
        if denom == 0:
            return 0.0
        busted_calls = len({c for c, _ in self.busts})
        return busted_calls / denom
