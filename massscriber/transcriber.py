from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Iterator
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

CUDA_RUNTIME_HINTS = (
    "cublas64_12.dll",
    "cudnn64_9.dll",
    "cudart64_12.dll",
    "curand64_10.dll",
    "cufft64_11.dll",
)


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


def resolve_compute_type(device: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "float16" if device == "cuda" else "int8"


def resolve_cpu_fallback_compute_type(requested: str) -> str:
    if requested in {"int8", "float32"}:
        return requested
    return "int8"


def is_missing_cuda_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(hint.lower() in message for hint in CUDA_RUNTIME_HINTS)


def format_cuda_fallback_warning(source_name: str, exc: Exception) -> str:
    matched_hint = next(
        (hint for hint in CUDA_RUNTIME_HINTS if hint.lower() in str(exc).lower()),
        "gerekli CUDA kutuphanesi",
    )
    return (
        f"{source_name}: GPU kutuphaneleri yuklenemedi ({matched_hint}). "
        "CPU moduna geciliyor. GPU kullanmak icin Windows'ta CUDA 12 ve cuDNN 9 runtime'larini kur."
    )


def build_base_name(audio_path: Path) -> str:
    digest = hashlib.sha1(str(audio_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{sanitize_name(audio_path.stem)}_{digest}"


class WhisperRuntime:
    _model_cache: dict[tuple[str, str, str, int | None], WhisperModel] = {}
    _pipeline_cache: dict[tuple[str, str, str, int | None], BatchedInferencePipeline] = {}
    _lock = threading.Lock()

    @staticmethod
    def build_cache_key(
        settings: TranscriptionSettings,
        device: str,
        compute_type: str,
    ) -> tuple[str, str, str, int | None]:
        cpu_threads = settings.cpu_threads if settings.cpu_threads and settings.cpu_threads > 0 else None
        return (settings.model, device, compute_type, cpu_threads)

    @classmethod
    def get_backend(
        cls,
        settings: TranscriptionSettings,
        *,
        forced_device: str | None = None,
        forced_compute_type: str | None = None,
    ) -> tuple[WhisperModel, BatchedInferencePipeline | None, str, str]:
        device = forced_device or detect_device(settings.device)
        compute_type = forced_compute_type or resolve_compute_type(device, settings.compute_type)
        cpu_threads = settings.cpu_threads if settings.cpu_threads and settings.cpu_threads > 0 else None
        key = cls.build_cache_key(settings, device, compute_type)

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

    @classmethod
    def clear_backend(
        cls,
        settings: TranscriptionSettings,
        *,
        device: str,
        compute_type: str,
    ) -> None:
        key = cls.build_cache_key(settings, device, compute_type)
        with cls._lock:
            cls._pipeline_cache.pop(key, None)
            cls._model_cache.pop(key, None)


class TranscriptionEngine:
    @staticmethod
    def _build_transcribe_kwargs(
        settings: TranscriptionSettings,
        *,
        use_pipeline: bool,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {
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
        if use_pipeline:
            kwargs["batch_size"] = settings.batch_size

        return kwargs

    def _transcribe_once(
        self,
        source: Path,
        settings: TranscriptionSettings,
        *,
        forced_device: str | None = None,
        forced_compute_type: str | None = None,
    ) -> tuple[object, object, str, str]:
        model, pipeline, device, compute_type = WhisperRuntime.get_backend(
            settings,
            forced_device=forced_device,
            forced_compute_type=forced_compute_type,
        )
        transcribe_fn = pipeline.transcribe if pipeline else model.transcribe
        kwargs = self._build_transcribe_kwargs(settings, use_pipeline=bool(pipeline))
        logger.info("Transcribing %s with model=%s device=%s", source.name, settings.model, device)
        raw_segments, info = transcribe_fn(str(source), **kwargs)
        return raw_segments, info, device, compute_type

    def stream_file(
        self,
        audio_path: str | Path,
        settings: TranscriptionSettings,
        output_dir: str | Path,
    ) -> Iterator[tuple[float, str, TranscriptionResult | None]]:
        source = Path(audio_path).expanduser().resolve()
        yield 0.02, f"{source.name}: dosya hazirlaniyor", None

        if not source.exists():
            raise FileNotFoundError(f"Audio file not found: {source}")

        if settings.task == "translate" and settings.model == "turbo":
            raise ValueError(
                "The turbo model is optimized for transcription. "
                "Use large-v3 or medium for translation."
            )

        yield 0.08, f"{source.name}: model hazirlaniyor ({settings.model})", None
        try:
            yield 0.15, f"{source.name}: transkripsiyon baslatiliyor", None
            raw_segments, info, device, compute_type = self._transcribe_once(source, settings)
        except Exception as exc:
            if settings.device not in {"auto", "cuda"} or not is_missing_cuda_runtime_error(exc):
                raise

            failed_device = "cuda"
            failed_compute_type = resolve_compute_type(failed_device, settings.compute_type)
            WhisperRuntime.clear_backend(
                settings,
                device=failed_device,
                compute_type=failed_compute_type,
            )

            warning_message = format_cuda_fallback_warning(source.name, exc)
            logger.warning("%s", warning_message)
            yield 0.12, f"[UYARI] {warning_message}", None

            fallback_compute_type = resolve_cpu_fallback_compute_type(settings.compute_type)
            yield 0.14, f"{source.name}: CPU moduna geciliyor ({fallback_compute_type})", None
            yield 0.18, f"{source.name}: CPU transkripsiyonu baslatiliyor", None
            raw_segments, info, device, compute_type = self._transcribe_once(
                source,
                settings,
                forced_device="cpu",
                forced_compute_type=fallback_compute_type,
            )

        total_duration = float(getattr(info, "duration", 0) or 0)
        segments: list[SegmentData] = []
        last_progress = 0.15
        for index, segment in enumerate(raw_segments):
            coerced = self._coerce_segment(index, segment)
            segments.append(coerced)

            if total_duration > 0:
                ratio = min(max(coerced.end / total_duration, 0.0), 0.995)
                current_progress = 0.15 + (0.75 * ratio)
                if current_progress - last_progress >= 0.02 or ratio >= 0.995:
                    last_progress = current_progress
                    percent = int(ratio * 100)
                    yield current_progress, f"{source.name}: transkribe ediliyor ({percent}%)", None
            else:
                current_progress = min(0.2 + ((index + 1) * 0.02), 0.88)
                if current_progress - last_progress >= 0.02:
                    last_progress = current_progress
                    yield current_progress, f"{source.name}: transkribe ediliyor (segment {index + 1})", None

        yield 0.92, f"{source.name}: metin toparlaniyor", None
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
        yield 0.97, f"{source.name}: ciktilar yaziliyor", None
        export_result(result, output_root)
        yield 1.0, f"{source.name}: tamamlandi", result

    def transcribe_file(
        self,
        audio_path: str | Path,
        settings: TranscriptionSettings,
        output_dir: str | Path,
    ) -> TranscriptionResult:
        result: TranscriptionResult | None = None
        for _, _, maybe_result in self.stream_file(audio_path, settings, output_dir):
            if maybe_result is not None:
                result = maybe_result

        if result is None:
            raise RuntimeError("Transcription finished without producing a result.")

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
