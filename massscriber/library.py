from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REVIEW_STATE_FILE_NAME = ".massscriber-review-state.json"


@dataclass(slots=True)
class TranscriptRecord:
    transcript_id: str
    transcript_path: Path
    text: str
    audio_path: str
    language: str
    model: str
    status: str = "pending"
    reviewed_at: str = ""
    note: str = ""


def load_review_state(output_dir: str | Path) -> dict[str, dict[str, str]]:
    state_file = Path(output_dir).expanduser().resolve() / REVIEW_STATE_FILE_NAME
    if not state_file.exists():
        return {}
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_review_state(output_dir: str | Path, state: dict[str, dict[str, str]]) -> None:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_file = output_root / REVIEW_STATE_FILE_NAME
    state_file.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def index_transcripts(output_dir: str | Path) -> list[TranscriptRecord]:
    output_root = Path(output_dir).expanduser().resolve()
    review_state = load_review_state(output_root)
    records_by_id: dict[str, TranscriptRecord] = {}

    for json_file in sorted(output_root.glob("*.json")):
        if json_file.name == REVIEW_STATE_FILE_NAME:
            continue
        record = _record_from_json(json_file, review_state)
        if record:
            records_by_id[record.transcript_id] = record

    for txt_file in sorted(output_root.glob("*.txt")):
        transcript_id = txt_file.stem
        if transcript_id in records_by_id:
            continue
        review = review_state.get(transcript_id, {})
        records_by_id[transcript_id] = TranscriptRecord(
            transcript_id=transcript_id,
            transcript_path=txt_file.resolve(),
            text=txt_file.read_text(encoding="utf-8").strip(),
            audio_path="",
            language="unknown",
            model="unknown",
            status=str(review.get("status", "pending")),
            reviewed_at=str(review.get("reviewed_at", "")),
            note=str(review.get("note", "")),
        )

    return sorted(records_by_id.values(), key=lambda record: record.transcript_id)


def search_transcripts(
    output_dir: str | Path,
    *,
    query: str = "",
    status_filter: str = "all",
) -> tuple[list[TranscriptRecord], str]:
    records = index_transcripts(output_dir)
    normalized_query = query.strip().lower()

    filtered: list[TranscriptRecord] = []
    for record in records:
        if status_filter != "all" and record.status != status_filter:
            continue
        haystack = " ".join(
            [
                record.transcript_id,
                record.text,
                record.audio_path,
                record.language,
                record.model,
                record.note,
            ]
        ).lower()
        if normalized_query and normalized_query not in haystack:
            continue
        filtered.append(record)

    summary = f"[INFO] {len(filtered)} transcript bulundu."
    if normalized_query:
        summary += f" Sorgu: '{query.strip()}'"
    return filtered, summary


def records_to_rows(records: list[TranscriptRecord]) -> list[list[str]]:
    rows: list[list[str]] = []
    for record in records:
        snippet = record.text.replace("\n", " ").strip()
        if len(snippet) > 120:
            snippet = snippet[:117].rstrip() + "..."
        rows.append(
            [
                record.transcript_id,
                record.language or "unknown",
                record.model or "unknown",
                record.status,
                record.reviewed_at,
                snippet,
            ]
        )
    return rows


def build_preview(records: list[TranscriptRecord]) -> str:
    if not records:
        return ""
    record = records[0]
    lines = [
        f"ID: {record.transcript_id}",
        f"Status: {record.status}",
        f"Language: {record.language or 'unknown'}",
        f"Model: {record.model or 'unknown'}",
    ]
    if record.audio_path:
        lines.append(f"Source: {record.audio_path}")
    if record.note:
        lines.append(f"Note: {record.note}")
    lines.extend(["", record.text.strip()])
    return "\n".join(lines)


def update_review_status(
    output_dir: str | Path,
    transcript_ids: list[str],
    *,
    status: str,
    note: str = "",
) -> int:
    state = load_review_state(output_dir)
    count = 0
    for transcript_id in transcript_ids:
        cleaned_id = transcript_id.strip()
        if not cleaned_id:
            continue
        state[cleaned_id] = {
            "status": status,
            "note": note.strip(),
            "reviewed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        count += 1
    save_review_state(output_dir, state)
    return count


def extract_transcript_ids(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _record_from_json(
    json_file: Path,
    review_state: dict[str, dict[str, str]],
) -> TranscriptRecord | None:
    try:
        payload = json.loads(json_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    transcript_id = json_file.stem
    review = review_state.get(transcript_id, {})
    return TranscriptRecord(
        transcript_id=transcript_id,
        transcript_path=json_file.resolve(),
        text=str(payload.get("text", "")).strip(),
        audio_path=str(payload.get("audio_path", "")),
        language=str(payload.get("language", "unknown") or "unknown"),
        model=str(payload.get("model", "unknown") or "unknown"),
        status=str(review.get("status", "pending")),
        reviewed_at=str(review.get("reviewed_at", "")),
        note=str(review.get("note", "")),
    )
