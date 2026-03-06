from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path

from massscriber.types import SegmentData, TranscriptionSettings


@dataclass(slots=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def diarize_audio(
    audio_path: str | Path,
    settings: TranscriptionSettings,
    *,
    prefer_device: str = "cpu",
) -> tuple[list[SpeakerTurn], list[str]]:
    if not settings.enable_diarization:
        return [], []

    token = settings.diarization_token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        return [], [
            "[UYARI] Speaker diarization istendi ama HF token bulunamadi. "
            "HUGGINGFACE_HUB_TOKEN ayarla ya da UI/CLI alanina token gir."
        ]

    try:
        pyannote_audio = importlib.import_module("pyannote.audio")
        torch = importlib.import_module("torch")
    except Exception:
        return [], [
            "[UYARI] Speaker diarization icin opsiyonel bagimliliklar eksik. "
            "'pip install -e \".[diarization]\"' ile kurabilirsin."
        ]

    try:
        pipeline = pyannote_audio.Pipeline.from_pretrained(
            settings.diarization_model,
            use_auth_token=token,
        )
        if hasattr(pipeline, "to"):
            target = torch.device("cuda" if prefer_device == "cuda" and torch.cuda.is_available() else "cpu")
            pipeline.to(target)
        diarization = pipeline(str(Path(audio_path).expanduser().resolve()))
    except Exception as exc:
        return [], [f"[UYARI] Speaker diarization baslatilamadi: {exc}"]

    turns: list[SpeakerTurn] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            SpeakerTurn(
                start=float(turn.start),
                end=float(turn.end),
                speaker=str(speaker),
            )
        )

    if not turns:
        return [], ["[UYARI] Speaker diarization calisti ama speaker turn bulunamadi."]

    return normalize_speaker_labels(turns), [
        f"[INFO] Speaker diarization tamamlandi: {len(turns)} turn bulundu."
    ]


def normalize_speaker_labels(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    label_map: dict[str, str] = {}
    normalized: list[SpeakerTurn] = []
    for turn in turns:
        mapped = label_map.setdefault(turn.speaker, f"Speaker {len(label_map) + 1}")
        normalized.append(SpeakerTurn(start=turn.start, end=turn.end, speaker=mapped))
    return normalized


def assign_speakers_to_segments(
    segments: list[SegmentData],
    turns: list[SpeakerTurn],
) -> list[SegmentData]:
    if not turns:
        return segments

    for segment in segments:
        best_speaker: str | None = None
        best_overlap = 0.0
        for turn in turns:
            overlap = min(segment.end, turn.end) - max(segment.start, turn.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker
        if best_speaker:
            segment.speaker = best_speaker

    return segments
