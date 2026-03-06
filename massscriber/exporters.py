from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from massscriber.types import SegmentData, TranscriptionResult, TranscriptionSettings


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "transcript"


def format_timestamp(seconds: float, *, decimal_marker: str) -> str:
    total_milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return (
        f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"
        f"{decimal_marker}{milliseconds:03d}"
    )


def to_plain_text(segments: list[SegmentData]) -> str:
    text = "".join(segment.text for segment in segments).strip()
    return re.sub(r"[ \t]+", " ", text)


def normalize_caption_text(value: str) -> str:
    collapsed = re.sub(r"\s+", " ", value).strip()
    collapsed = re.sub(r"\s+([,.;:!?])", r"\1", collapsed)
    collapsed = re.sub(r"([(\[{])\s+", r"\1", collapsed)
    return collapsed


def wrap_subtitle_text(text: str, max_chars: int, max_lines: int = 2) -> str:
    words = text.split()
    if len(words) <= 1 or max_lines <= 1 or len(text) <= max_chars:
        return text

    soft_limit = max(12, max_chars // max_lines)
    lines: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join(current + [word]).strip()
        if current and len(candidate) > soft_limit and len(lines) < max_lines - 1:
            lines.append(" ".join(current).strip())
            current = [word]
            continue
        current.append(word)

    if current:
        lines.append(" ".join(current).strip())

    if len(lines) > max_lines:
        merged = lines[: max_lines - 1]
        merged.append(" ".join(lines[max_lines - 1 :]).strip())
        lines = merged

    return "\n".join(line for line in lines if line)


def _format_subtitle_text(text: str, speaker: str | None, max_chars: int) -> str:
    base = normalize_caption_text(text)
    if speaker:
        base = f"[{speaker}] {base}"
    return wrap_subtitle_text(base, max_chars=max_chars)


def build_subtitle_segments(
    result: TranscriptionResult,
    settings: TranscriptionSettings | None = None,
) -> list[tuple[float, float, str]]:
    config = settings or TranscriptionSettings()
    max_chars = max(12, int(config.subtitle_max_chars))
    max_duration = max(1.0, float(config.subtitle_max_duration))
    pause_threshold = (
        max(0.0, float(config.subtitle_pause_threshold))
        if config.subtitle_split_on_pause
        else None
    )

    word_units: list[tuple[float, float, str, str | None]] = []
    for segment in result.segments:
        if segment.words:
            for word in segment.words:
                token = str(word.word)
                if token.strip():
                    word_units.append((word.start, word.end, token, segment.speaker))

    if word_units:
        return _build_subtitle_segments_from_words(
            word_units,
            max_chars=max_chars,
            max_duration=max_duration,
            pause_threshold=pause_threshold,
        )

    return _build_subtitle_segments_from_segments(
        result.segments,
        max_chars=max_chars,
        max_duration=max_duration,
        pause_threshold=pause_threshold,
    )


def _build_subtitle_segments_from_words(
    units: list[tuple[float, float, str, str | None]],
    *,
    max_chars: int,
    max_duration: float,
    pause_threshold: float | None,
) -> list[tuple[float, float, str]]:
    subtitles: list[tuple[float, float, str]] = []
    current_tokens: list[str] = []
    current_start: float | None = None
    current_end: float | None = None
    current_speaker: str | None = None

    def flush() -> None:
        nonlocal current_tokens, current_start, current_end, current_speaker
        if not current_tokens or current_start is None or current_end is None:
            return
        subtitles.append(
            (
                current_start,
                current_end,
                _format_subtitle_text("".join(current_tokens), current_speaker, max_chars),
            )
        )
        current_tokens = []
        current_start = None
        current_end = None
        current_speaker = None

    for start, end, token, speaker in units:
        candidate_tokens = current_tokens + [token]
        candidate_text = normalize_caption_text("".join(candidate_tokens))
        candidate_duration = (end - current_start) if current_start is not None else (end - start)
        pause_break = (
            current_end is not None
            and pause_threshold is not None
            and (start - current_end) >= pause_threshold
        )
        speaker_break = current_speaker is not None and speaker is not None and speaker != current_speaker

        if current_tokens and (
            pause_break
            or speaker_break
            or len(candidate_text) > max_chars
            or candidate_duration > max_duration
        ):
            flush()

        if current_start is None:
            current_start = start
        current_end = end
        current_speaker = speaker or current_speaker
        current_tokens.append(token)

    flush()
    return subtitles


def _build_subtitle_segments_from_segments(
    segments: list[SegmentData],
    *,
    max_chars: int,
    max_duration: float,
    pause_threshold: float | None,
) -> list[tuple[float, float, str]]:
    subtitles: list[tuple[float, float, str]] = []
    current_segments: list[SegmentData] = []

    def flush() -> None:
        nonlocal current_segments
        if not current_segments:
            return
        subtitles.append(
            (
                current_segments[0].start,
                current_segments[-1].end,
                _format_subtitle_text(
                    " ".join(segment.text for segment in current_segments),
                    current_segments[0].speaker,
                    max_chars,
                ),
            )
        )
        current_segments = []

    for segment in segments:
        candidate_segments = current_segments + [segment]
        candidate_text = normalize_caption_text(" ".join(item.text for item in candidate_segments))
        candidate_duration = candidate_segments[-1].end - candidate_segments[0].start
        pause_break = (
            current_segments
            and pause_threshold is not None
            and (segment.start - current_segments[-1].end) >= pause_threshold
        )
        speaker_break = (
            current_segments
            and current_segments[-1].speaker
            and segment.speaker
            and current_segments[-1].speaker != segment.speaker
        )
        if current_segments and (
            pause_break
            or speaker_break
            or len(candidate_text) > max_chars
            or candidate_duration > max_duration
        ):
            flush()
        current_segments.append(segment)

    flush()
    return subtitles


def build_txt(result: TranscriptionResult) -> str:
    return result.text.strip() + "\n"


def build_srt(result: TranscriptionResult, settings: TranscriptionSettings | None = None) -> str:
    lines: list[str] = []
    for idx, (start, end, text) in enumerate(build_subtitle_segments(result, settings), start=1):
        lines.extend(
            [
                str(idx),
                (
                    f"{format_timestamp(start, decimal_marker=',')} --> "
                    f"{format_timestamp(end, decimal_marker=',')}"
                ),
                text,
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_vtt(result: TranscriptionResult, settings: TranscriptionSettings | None = None) -> str:
    lines = ["WEBVTT", ""]
    for start, end, text in build_subtitle_segments(result, settings):
        lines.extend(
            [
                (
                    f"{format_timestamp(start, decimal_marker='.')} --> "
                    f"{format_timestamp(end, decimal_marker='.')}"
                ),
                text,
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_json(result: TranscriptionResult) -> str:
    payload = {
        "audio_path": str(result.audio_path),
        "model": result.model,
        "task": result.task,
        "language": result.language,
        "language_probability": result.language_probability,
        "duration_seconds": result.duration,
        "device": result.device,
        "compute_type": result.compute_type,
        "text": result.text,
        "segments": [asdict(segment) for segment in result.segments],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def export_result(
    result: TranscriptionResult,
    output_dir: Path,
    settings: TranscriptionSettings | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    writers = {
        "txt": build_txt,
        "srt": build_srt,
        "vtt": build_vtt,
        "json": build_json,
    }

    written: dict[str, Path] = {}
    for file_format, builder in writers.items():
        if file_format not in result.output_files:
            continue
        target = result.output_files[file_format]
        if file_format in {"srt", "vtt"}:
            content = builder(result, settings)
        else:
            content = builder(result)
        target.write_text(content, encoding="utf-8")
        written[file_format] = target

    return written
