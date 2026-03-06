from __future__ import annotations

import re
from dataclasses import dataclass

from massscriber.types import SegmentData, TranscriptionSettings


@dataclass(slots=True)
class GlossaryRule:
    source: str
    target: str


def parse_glossary_rules(raw_text: str) -> list[GlossaryRule]:
    rules: list[GlossaryRule] = []
    for line in raw_text.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue

        separator = "=>"
        if "=>" not in cleaned and "->" in cleaned:
            separator = "->"
        if separator not in cleaned:
            continue

        source, target = [part.strip() for part in cleaned.split(separator, 1)]
        if not source or not target:
            continue
        rules.append(GlossaryRule(source=source, target=target))
    return rules


def _build_rule_pattern(rule: GlossaryRule, whole_word: bool, *, case_sensitive: bool) -> re.Pattern[str]:
    escaped = re.escape(rule.source)
    if whole_word:
        escaped = rf"(?<!\w){escaped}(?!\w)"
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(escaped, flags)


def apply_glossary_to_text(text: str, settings: TranscriptionSettings) -> str:
    rules = parse_glossary_rules(settings.glossary_text)
    if not rules or not text:
        return text

    transformed = text
    for rule in rules:
        pattern = _build_rule_pattern(
            rule,
            settings.glossary_whole_word,
            case_sensitive=settings.glossary_case_sensitive,
        )
        transformed = pattern.sub(rule.target, transformed)
    return transformed


def apply_glossary_to_segments(segments: list[SegmentData], settings: TranscriptionSettings) -> list[SegmentData]:
    rules = parse_glossary_rules(settings.glossary_text)
    if not rules:
        return segments

    for segment in segments:
        segment.text = apply_glossary_to_text(segment.text, settings)
        for word in segment.words:
            prefix_length = len(word.word) - len(word.word.lstrip())
            prefix = word.word[:prefix_length]
            token = word.word[prefix_length:]
            updated = apply_glossary_to_text(token, settings)
            word.word = prefix + updated

    return segments


def build_glossary_summary(settings: TranscriptionSettings) -> str | None:
    rules = parse_glossary_rules(settings.glossary_text)
    if not rules:
        return None
    mode = "case-sensitive" if settings.glossary_case_sensitive else "case-insensitive"
    scope = "whole-word" if settings.glossary_whole_word else "phrase"
    return f"[INFO] Glossary aktif: {len(rules)} kural ({mode}, {scope})"
