from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

OutputFormat = Literal["txt", "srt", "vtt", "json"]
TaskType = Literal["transcribe", "translate"]


@dataclass(slots=True)
class WordTiming:
    start: float
    end: float
    word: str
    probability: float | None = None


@dataclass(slots=True)
class SegmentData:
    index: int
    start: float
    end: float
    text: str
    speaker: str | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None
    words: list[WordTiming] = field(default_factory=list)


@dataclass(slots=True)
class TranscriptionSettings:
    provider: str = "local"
    model: str = "large-v3"
    provider_model: str | None = None
    provider_api_key: str | None = None
    provider_base_url: str | None = None
    provider_remote_url: str | None = None
    provider_timeout_seconds: float = 900.0
    provider_poll_interval: float = 3.0
    provider_smart_format: bool = True
    provider_speaker_labels: bool = False
    provider_keywords: str = ""
    provider_keep_raw_response: bool = False
    provider_fallback_to_local: bool = True
    language: str | None = None
    task: TaskType = "transcribe"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    batch_size: int = 8
    temperature: float = 0.0
    vad_filter: bool = True
    vad_min_silence_ms: int = 500
    word_timestamps: bool = True
    condition_on_previous_text: bool = False
    initial_prompt: str = ""
    cpu_threads: int | None = None
    subtitle_max_chars: int = 42
    subtitle_max_duration: float = 6.0
    subtitle_split_on_pause: bool = True
    subtitle_pause_threshold: float = 0.6
    enable_diarization: bool = False
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    diarization_token: str | None = None
    glossary_text: str = ""
    glossary_case_sensitive: bool = False
    glossary_whole_word: bool = True
    output_formats: tuple[OutputFormat, ...] = ("txt", "srt", "json")


@dataclass(slots=True)
class TranscriptionResult:
    audio_path: Path
    base_name: str
    provider: str
    model: str
    language: str | None
    language_probability: float | None
    duration: float | None
    task: TaskType
    device: str
    compute_type: str
    text: str
    segments: list[SegmentData]
    output_files: dict[str, Path]
    provider_job_id: str | None = None
    provider_request_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
