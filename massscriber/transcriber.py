from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

import ctranslate2
from faster_whisper import BatchedInferencePipeline, WhisperModel

from massscriber.exporters import export_result, sanitize_name, to_plain_text
from massscriber.types import SegmentData, TranscriptionResult, TranscriptionSettings, WordTiming

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = (
    "large-v3",
    "turbo",
    "medium",
    "small",
    "base",
    "tiny",
)


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


def resolve_compute_type(device: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "float16" if device == "cuda" else "int8"


def build_base_name(audio_path: Path) -> str:
    digest = hashlib.sha1(str(audio_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{sanitize_name(audio_path.stem)}_{digest}"


class WhisperRuntime:
    _model_cache: dict[tuple[str, str, str, int | None], WhisperModel] = {}
    _pipeline_cache: dict[tuple[str, str, str, int | None], BatchedInferencePipeline] = {}
    _lock = threading.Lock()

    @classmethod
    def get_backend(
        cls, settings: TranscriptionSettings
    ) -> tuple[WhisperModel, BatchedInferencePipeline | None, str, str]:
        device = detect_device(settings.device)
        compute_type = resolve_compute_type(device, settings.compute_type)
        cpu_threads = settings.cpu_threads if settings.cpu_threads and settings.cpu_threads > 0 else None
        key = (settings.model, device, compute_type, cpu_threads)

        with cls._lock:
            if key not in cls._model_cache:
                kwargs = {
                    "device": device,
                    "compute_type": compute_type,
                }
                if cpu_threads:
                    kwargs["cpu_threads"] = cpu_threads
                cls._model_cache[key] = WhisperModel(settings.model, **kwargs)

            model = cls._model_cache[key]

            pipeline = None
            if settings.batch_size > 1:
                if key not in cls._pipeline_cache:
                    cls._pipeline_cache[key] = BatchedInferencePipeline(model=model)
                pipeline = cls._pipeline_cache[key]

        return model, pipeline, device, compute_type


class TranscriptionEngine:
    def transcribe_file(
        self,
        audio_path: str | Path,
        settings: TranscriptionSettings,
        output_dir: str | Path,
    ) -> TranscriptionResult:
        source = Path(audio_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Audio file not found: {source}")

        if settings.task == "translate" and settings.model == "turbo":
            raise ValueError(
                "The turbo model is optimized for transcription. "
                "Use large-v3 or medium for translation."
            )

        model, pipeline, device, compute_type = WhisperRuntime.get_backend(settings)
        transcribe_fn = pipeline.transcribe if pipeline else model.transcribe
        kwargs = {
            "beam_size": settings.beam_size,
            "language": settings.language,
            "task": settings.task,
            "temperature": settings.temperature,
            "condition_on_previous_text": settings.condition_on_previous_text,
            "word_timestamps": settings.word_timestamps,
            "vad_filter": settings.vad_filter,
        }

        if settings.initial_prompt.strip():
            kwargs["initial_prompt"] = settings.initial_prompt.strip()
        if settings.vad_filter:
            kwargs["vad_parameters"] = {
                "min_silence_duration_ms": settings.vad_min_silence_ms
            }
        if pipeline:
            kwargs["batch_size"] = settings.batch_size

        logger.info("Transcribing %s with model=%s device=%s", source.name, settings.model, device)
        raw_segments, info = transcribe_fn(str(source), **kwargs)
        segments = [self._coerce_segment(index, segment) for index, segment in enumerate(raw_segments)]
        text = to_plain_text(segments)
        base_name = build_base_name(source)

        output_root = Path(output_dir).expanduser().resolve()
        output_files = {
            file_format: output_root / f"{base_name}.{file_format}"
            for file_format in settings.output_formats
        }

        result = TranscriptionResult(
            audio_path=source,
            base_name=base_name,
            model=settings.model,
            language=getattr(info, "language", settings.language),
            language_probability=getattr(info, "language_probability", None),
            duration=getattr(info, "duration", None),
            task=settings.task,
            device=device,
            compute_type=compute_type,
            text=text,
            segments=segments,
            output_files=output_files,
        )
        export_result(result, output_root)
        return result

    @staticmethod
    def _coerce_segment(index: int, segment: object) -> SegmentData:
        words = []
        raw_words = getattr(segment, "words", None) or []
        for word in raw_words:
            words.append(
                WordTiming(
                    start=float(word.start),
                    end=float(word.end),
                    word=str(word.word),
                    probability=getattr(word, "probability", None),
                )
            )

        return SegmentData(
            index=index,
            start=float(segment.start),
            end=float(segment.end),
            text=str(segment.text),
            avg_logprob=getattr(segment, "avg_logprob", None),
            compression_ratio=getattr(segment, "compression_ratio", None),
            no_speech_prob=getattr(segment, "no_speech_prob", None),
            words=words,
        )
