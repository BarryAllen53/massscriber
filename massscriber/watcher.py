from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from massscriber.transcriber import TranscriptionEngine
from massscriber.types import TranscriptionSettings

SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".mp4", ".mkv")
STATE_FILE_NAME = ".massscriber-watch-state.json"

logger = logging.getLogger(__name__)


def is_supported_media_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def build_file_snapshot(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def load_watch_state(state_file: Path) -> dict[str, dict[str, object]]:
    if not state_file.exists():
        return {}
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_watch_state(state_file: Path, state: dict[str, dict[str, object]]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def iter_media_files(folder: Path, recursive: bool) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(path.resolve() for path in iterator if is_supported_media_file(path))


def file_is_stable(path: Path, stable_seconds: float) -> bool:
    age = time.time() - path.stat().st_mtime
    return age >= stable_seconds


def should_process_file(path: Path, state: dict[str, dict[str, object]]) -> bool:
    snapshot = build_file_snapshot(path)
    existing = state.get(str(path))
    if not existing:
        return True
    return (
        existing.get("size") != snapshot["size"]
        or existing.get("mtime_ns") != snapshot["mtime_ns"]
    )


def move_to_archive(source: Path, archive_dir: Path, watch_root: Path) -> Path:
    try:
        relative_path = source.relative_to(watch_root)
    except ValueError:
        relative_path = Path(source.name)

    target = archive_dir / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        target = target.with_stem(f"{target.stem}_{timestamp}")
    shutil.move(str(source), str(target))
    return target


def watch_folder(
    folder: str | Path,
    settings: TranscriptionSettings,
    output_dir: str | Path,
    *,
    recursive: bool = True,
    poll_interval: float = 5.0,
    stable_seconds: float = 10.0,
    once: bool = False,
    archive_dir: str | Path | None = None,
) -> Iterator[str]:
    watch_root = Path(folder).expanduser().resolve()
    if not watch_root.exists():
        raise FileNotFoundError(f"Watch folder not found: {watch_root}")
    if not watch_root.is_dir():
        raise NotADirectoryError(f"Expected a folder to watch: {watch_root}")

    output_root = Path(output_dir).expanduser().resolve()
    state_file = output_root / STATE_FILE_NAME
    state = load_watch_state(state_file)
    engine = TranscriptionEngine()
    archive_root = Path(archive_dir).expanduser().resolve() if archive_dir else None

    while True:
        processed_this_round = 0
        pending_this_round = 0

        for media_file in iter_media_files(watch_root, recursive):
            if not should_process_file(media_file, state):
                continue

            if not file_is_stable(media_file, stable_seconds):
                pending_this_round += 1
                yield (
                    f"[BEKLIYOR] {media_file.name}: dosya hala yaziliyor olabilir, {stable_seconds:.1f} sn beklenecek"
                )
                continue

            yield f"[BASLADI] {media_file.name}: otomatik transkripsiyon"
            result = None
            for _, message, maybe_result in engine.stream_file(media_file, settings, output_root):
                if message.startswith("[UYARI]") or message.startswith("[INFO]"):
                    yield message
                if maybe_result is not None:
                    result = maybe_result
            if result is None:
                raise RuntimeError(f"Watch transcription finished without a result: {media_file.name}")
            snapshot = build_file_snapshot(media_file)
            state[str(media_file)] = {
                **snapshot,
                "processed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "outputs": {file_format: str(path) for file_format, path in result.output_files.items()},
            }
            save_watch_state(state_file, state)
            processed_this_round += 1
            yield (
                f"[OK] {media_file.name}: dil={result.language or 'unknown'}, cikti={len(result.output_files)}"
            )

            if archive_root:
                archived_path = move_to_archive(media_file, archive_root, watch_root)
                yield f"[ARSIV] {media_file.name}: {archived_path}"

        if once:
            if processed_this_round == 0 and pending_this_round == 0:
                yield "[INFO] Izleme turu tamamlandi, yeni dosya bulunmadi."
            break

        if processed_this_round == 0 and pending_this_round == 0:
            yield "[BEKLEME] Yeni dosya bekleniyor..."
        time.sleep(max(1.0, poll_interval))
