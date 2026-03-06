from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from massscriber.types import SegmentData, TranscriptionResult


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


def build_txt(result: TranscriptionResult) -> str:
    return result.text.strip() + "\n"


def build_srt(result: TranscriptionResult) -> str:
    lines: list[str] = []
    for idx, segment in enumerate(result.segments, start=1):
        lines.extend(
            [
                str(idx),
                (
                    f"{format_timestamp(segment.start, decimal_marker=',')} --> "
                    f"{format_timestamp(segment.end, decimal_marker=',')}"
                ),
                segment.text.strip(),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_vtt(result: TranscriptionResult) -> str:
    lines = ["WEBVTT", ""]
    for segment in result.segments:
        lines.extend(
            [
                (
                    f"{format_timestamp(segment.start, decimal_marker='.')} --> "
                    f"{format_timestamp(segment.end, decimal_marker='.')}"
                ),
                segment.text.strip(),
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


def export_result(result: TranscriptionResult, output_dir: Path) -> dict[str, Path]:
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
        target.write_text(builder(result), encoding="utf-8")
        written[file_format] = target

    return written
