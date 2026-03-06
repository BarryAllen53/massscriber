from __future__ import annotations

import hashlib
import logging
import os
import site
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

from massscriber.cloud import RemoteTranscriptionEngine
from massscriber.diarization import assign_speakers_to_segments, diarize_audio
from massscriber.exporters import export_result, sanitize_name, to_plain_text
from massscriber.postprocess import apply_glossary_to_segments, apply_glossary_to_text, build_glossary_summary
from massscriber.providers import provider_uses_remote_api
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

WINDOWS_NVIDIA_BIN_SUBDIRS = (
    Path("nvidia") / "cublas" / "bin",
    Path("nvidia") / "cudnn" / "bin",
    Path("nvidia") / "cuda_runtime" / "bin",
    Path("nvidia") / "cufft" / "bin",
    Path("nvidia") / "curand" / "bin",
    Path("nvidia") / "nvjitlink" / "bin",
)

WINDOWS_DLL_HANDLES: list[object] = []


def configure_windows_cuda_runtime_paths() -> list[Path]:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []

    site_roots: list[Path] = []
    for candidate in [Path(sys.prefix) / "Lib" / "site-packages", Path(sys.base_prefix) / "Lib" / "site-packages"]:
        if candidate.exists() and candidate not in site_roots:
            site_roots.append(candidate)

    try:
        for raw_path in site.getsitepackages():
            candidate = Path(raw_path)
            if candidate.exists() and candidate not in site_roots:
                site_roots.append(candidate)
    except Exception:
        pass

    try:
        user_site = Path(site.getusersitepackages())
        if user_site.exists() and user_site not in site_roots:
            site_roots.append(user_site)
    except Exception:
        pass

    added_dirs: list[Path] = []
    seen_dirs: set[Path] = set()
    current_path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for site_root in site_roots:
        for relative_dir in WINDOWS_NVIDIA_BIN_SUBDIRS:
            candidate = (site_root / relative_dir).resolve()
            if not candidate.is_dir() or candidate in seen_dirs:
                continue
            WINDOWS_DLL_HANDLES.append(os.add_dll_directory(str(candidate)))
            added_dirs.append(candidate)
            seen_dirs.add(candidate)
            candidate_text = str(candidate)
            if candidate_text not in current_path_entries:
                current_path_entries.insert(0, candidate_text)

    if added_dirs:
        os.environ["PATH"] = os.pathsep.join(current_path_entries)

    return added_dirs


CONFIGURED_WINDOWS_CUDA_DIRS = configure_windows_cuda_runtime_paths()

import ctranslate2
from faster_whisper import BatchedInferencePipeline, WhisperModel


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
    def __init__(self) -> None:
        self._remote_engine = RemoteTranscriptionEngine()

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
        if provider_uses_remote_api(settings.provider):
            yield from self._remote_engine.stream_file(audio_path, settings, output_dir)
            return

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
        glossary_summary = build_glossary_summary(settings)
        if glossary_summary:
            yield 0.1, glossary_summary, None
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

        if settings.enable_diarization:
            yield 0.89, f"{source.name}: speaker diarization calisiyor", None
            speaker_turns, diarization_messages = diarize_audio(
                source,
                settings,
                prefer_device=device,
            )
            for message in diarization_messages:
                if message.startswith("[UYARI]"):
                    logger.warning("%s", message)
                else:
                    logger.info("%s", message)
                yield 0.9, message, None
            if speaker_turns:
                assign_speakers_to_segments(segments, speaker_turns)
                yield 0.91, f"{source.name}: speaker etiketleri uygulandi", None

        yield 0.92, f"{source.name}: metin toparlaniyor", None
        apply_glossary_to_segments(segments, settings)
        text = to_plain_text(segments)
        text = apply_glossary_to_text(text, settings)
        base_name = build_base_name(source)

        output_root = Path(output_dir).expanduser().resolve()
        output_files = {
            file_format: output_root / f"{base_name}.{file_format}"
            for file_format in settings.output_formats
        }

        result = TranscriptionResult(
            audio_path=source,
            base_name=base_name,
            provider="local",
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
            metadata={},
        )
        yield 0.97, f"{source.name}: ciktilar yaziliyor", None
        export_result(result, output_root, settings)
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
