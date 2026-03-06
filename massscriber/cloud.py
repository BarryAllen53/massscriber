from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import time
from collections.abc import Iterator
from pathlib import Path

import httpx

from massscriber.exporters import export_result, sanitize_name, to_plain_text
from massscriber.postprocess import apply_glossary_to_segments, apply_glossary_to_text, build_glossary_summary
from massscriber.providers import (
    PROVIDER_LABELS,
    get_provider_api_key,
    get_provider_base_url,
    provider_file_limit_warning,
    provider_supports_speaker_labels,
    provider_supports_translation,
    provider_uses_remote_api,
    redact_secret,
    resolve_provider_model,
)
from massscriber.types import SegmentData, TranscriptionResult, TranscriptionSettings, WordTiming

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    pass


def build_base_name(audio_path: Path) -> str:
    digest = hashlib.sha1(str(audio_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{sanitize_name(audio_path.stem)}_{digest}"


class RemoteTranscriptionEngine:
    def stream_file(
        self,
        audio_path: str | Path,
        settings: TranscriptionSettings,
        output_dir: str | Path,
    ) -> Iterator[tuple[float, str, TranscriptionResult | None]]:
        provider = settings.provider
        if not provider_uses_remote_api(provider):
            raise ProviderError("Remote engine local provider ile cagrildi.")

        source = Path(audio_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Audio file not found: {source}")

        yield 0.02, f"{source.name}: {PROVIDER_LABELS.get(provider, provider)} icin hazirlaniyor", None
        warning = provider_file_limit_warning(provider, source.stat().st_size)
        if warning:
            raise ProviderError(warning)

        api_key = get_provider_api_key(provider, settings.provider_api_key)
        if not api_key:
            raise ProviderError(
                f"{PROVIDER_LABELS.get(provider, provider)} icin API key gerekli. "
                f"UI alanini doldur ya da ortam degiskeni kullan."
            )

        if settings.task == "translate" and not provider_supports_translation(provider):
            raise ProviderError(f"{PROVIDER_LABELS.get(provider, provider)} su an translate modunu desteklemiyor.")

        provider_model = resolve_provider_model(provider, settings.provider_model, settings.model)
        glossary_summary = build_glossary_summary(settings)
        if glossary_summary:
            yield 0.05, glossary_summary, None
        yield 0.08, f"{source.name}: provider={provider}, model={provider_model}", None
        yield 0.12, f"{source.name}: API anahtari bulundu ({redact_secret(api_key)})", None

        if provider == "openai":
            payload, request_id = self._call_openai_family(
                provider="openai",
                source=source,
                settings=settings,
                api_key=api_key,
                model=provider_model,
            )
            result = self._build_openai_family_result(
                source=source,
                settings=settings,
                output_dir=output_dir,
                provider=provider,
                model=provider_model,
                payload=payload,
                request_id=request_id,
            )
        elif provider == "groq":
            payload, request_id = self._call_openai_family(
                provider="groq",
                source=source,
                settings=settings,
                api_key=api_key,
                model=provider_model,
            )
            result = self._build_openai_family_result(
                source=source,
                settings=settings,
                output_dir=output_dir,
                provider=provider,
                model=provider_model,
                payload=payload,
                request_id=request_id,
            )
        elif provider == "deepgram":
            payload, request_id = self._call_deepgram(
                source=source,
                settings=settings,
                api_key=api_key,
                model=provider_model,
            )
            result = self._build_deepgram_result(
                source=source,
                settings=settings,
                output_dir=output_dir,
                model=provider_model,
                payload=payload,
                request_id=request_id,
            )
        elif provider == "assemblyai":
            payload, request_id = self._call_assemblyai(
                source=source,
                settings=settings,
                api_key=api_key,
                model=provider_model,
            )
            result = self._build_assemblyai_result(
                source=source,
                settings=settings,
                output_dir=output_dir,
                model=provider_model,
                payload=payload,
                request_id=request_id,
            )
        elif provider == "elevenlabs":
            payload, request_id = self._call_elevenlabs(
                source=source,
                settings=settings,
                api_key=api_key,
                model=provider_model,
            )
            result = self._build_elevenlabs_result(
                source=source,
                settings=settings,
                output_dir=output_dir,
                model=provider_model,
                payload=payload,
                request_id=request_id,
            )
        else:
            raise ProviderError(f"Desteklenmeyen provider: {provider}")

        yield 0.92, f"{source.name}: provider yaniti isleniyor", None
        yield 0.97, f"{source.name}: ciktilar yaziliyor", None
        export_result(result, Path(output_dir).expanduser().resolve(), settings)
        yield 1.0, f"{source.name}: tamamlandi", result

    def _call_openai_family(
        self,
        *,
        provider: str,
        source: Path,
        settings: TranscriptionSettings,
        api_key: str,
        model: str,
    ) -> tuple[dict[str, object], str | None]:
        endpoint = "/audio/translations" if settings.task == "translate" else "/audio/transcriptions"
        response_format = "verbose_json" if model.startswith("whisper") else "json"
        data: list[tuple[str, str]] = [
            ("model", model),
            ("response_format", response_format),
        ]
        if settings.language and settings.task != "translate":
            data.append(("language", settings.language))
        prompt = settings.initial_prompt.strip()
        if settings.provider_keywords.strip():
            keyword_block = ", ".join(_parse_keywords(settings.provider_keywords))
            prompt = f"{prompt}\nKeywords: {keyword_block}".strip()
        if prompt:
            data.append(("prompt", prompt))
        if response_format == "verbose_json" and settings.word_timestamps:
            data.append(("timestamp_granularities[]", "word"))
            data.append(("timestamp_granularities[]", "segment"))

        response = self._request(
            provider=provider,
            method="POST",
            url=f"{get_provider_base_url(provider, settings.provider_base_url)}{endpoint}",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": _build_file_tuple(source)},
            data=data,
            timeout_seconds=settings.provider_timeout_seconds,
        )
        return _ensure_json(response), response.headers.get("x-request-id")

    def _call_deepgram(
        self,
        *,
        source: Path,
        settings: TranscriptionSettings,
        api_key: str,
        model: str,
    ) -> tuple[dict[str, object], str | None]:
        params: dict[str, str] = {
            "model": model,
            "punctuate": "true",
            "utterances": "true",
            "paragraphs": "true",
            "smart_format": "true" if settings.provider_smart_format else "false",
        }
        if settings.language:
            params["language"] = settings.language
        else:
            params["detect_language"] = "true"
        if settings.word_timestamps:
            params["words"] = "true"
        if settings.provider_speaker_labels and provider_supports_speaker_labels("deepgram"):
            params["diarize"] = "true"

        response = self._request(
            provider="deepgram",
            method="POST",
            url=f"{get_provider_base_url('deepgram', settings.provider_base_url)}/listen",
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": _guess_content_type(source),
            },
            params=params,
            content=source.read_bytes(),
            timeout_seconds=settings.provider_timeout_seconds,
        )
        return _ensure_json(response), response.headers.get("dg-request-id")

    def _call_assemblyai(
        self,
        *,
        source: Path,
        settings: TranscriptionSettings,
        api_key: str,
        model: str,
    ) -> tuple[dict[str, object], str | None]:
        base_url = get_provider_base_url("assemblyai", settings.provider_base_url)
        headers = {"authorization": api_key}
        upload_response = self._request(
            provider="assemblyai",
            method="POST",
            url=f"{base_url}/upload",
            headers=headers,
            content=source.read_bytes(),
            timeout_seconds=settings.provider_timeout_seconds,
        )
        upload_payload = _ensure_json(upload_response)
        audio_url = str(upload_payload.get("upload_url") or "").strip()
        if not audio_url:
            raise ProviderError("AssemblyAI upload_url donmedi.")

        transcript_payload: dict[str, object] = {
            "audio_url": audio_url,
            "speech_model": model,
            "speaker_labels": bool(settings.provider_speaker_labels),
            "format_text": bool(settings.provider_smart_format),
        }
        if settings.language:
            transcript_payload["language_code"] = settings.language
        keywords = _parse_keywords(settings.provider_keywords)
        if keywords:
            transcript_payload["word_boost"] = keywords
            transcript_payload["boost_param"] = "high"

        create_response = self._request(
            provider="assemblyai",
            method="POST",
            url=f"{base_url}/transcript",
            headers=headers,
            json=transcript_payload,
            timeout_seconds=settings.provider_timeout_seconds,
        )
        created = _ensure_json(create_response)
        transcript_id = str(created.get("id") or "").strip()
        if not transcript_id:
            raise ProviderError("AssemblyAI transcript id donmedi.")

        started_at = time.time()
        while True:
            poll_response = self._request(
                provider="assemblyai",
                method="GET",
                url=f"{base_url}/transcript/{transcript_id}",
                headers=headers,
                timeout_seconds=settings.provider_timeout_seconds,
            )
            payload = _ensure_json(poll_response)
            status = str(payload.get("status") or "").lower()
            if status == "completed":
                return payload, transcript_id
            if status == "error":
                raise ProviderError(f"AssemblyAI hata dondu: {payload.get('error') or 'bilinmeyen hata'}")
            if time.time() - started_at > settings.provider_timeout_seconds:
                raise ProviderError("AssemblyAI polling timeout asimina ugradi.")
            time.sleep(max(0.5, float(settings.provider_poll_interval)))

    def _call_elevenlabs(
        self,
        *,
        source: Path,
        settings: TranscriptionSettings,
        api_key: str,
        model: str,
    ) -> tuple[dict[str, object], str | None]:
        data: dict[str, str] = {
            "model_id": model,
            "diarize": "true" if settings.provider_speaker_labels else "false",
            "tag_audio_events": "true" if settings.provider_smart_format else "false",
        }
        if settings.language:
            data["language_code"] = settings.language

        response = self._request(
            provider="elevenlabs",
            method="POST",
            url=f"{get_provider_base_url('elevenlabs', settings.provider_base_url)}/speech-to-text",
            headers={"xi-api-key": api_key},
            files={"file": _build_file_tuple(source)},
            data=data,
            timeout_seconds=settings.provider_timeout_seconds,
        )
        return _ensure_json(response), response.headers.get("x-request-id")

    def _request(
        self,
        *,
        provider: str,
        method: str,
        url: str,
        timeout_seconds: float,
        **kwargs: object,
    ) -> httpx.Response:
        last_error: Exception | None = None
        timeout = httpx.Timeout(timeout_seconds)
        for attempt in range(1, 4):
            try:
                with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                    response = client.request(method, url, **kwargs)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                    time.sleep(attempt)
                    continue
                response.raise_for_status()
                return response
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(attempt)
                    continue
        raise ProviderError(f"{PROVIDER_LABELS.get(provider, provider)} istegi basarisiz: {last_error}")

    def _build_openai_family_result(
        self,
        *,
        source: Path,
        settings: TranscriptionSettings,
        output_dir: str | Path,
        provider: str,
        model: str,
        payload: dict[str, object],
        request_id: str | None,
    ) -> TranscriptionResult:
        segments, text = _parse_openai_family_payload(payload)
        return _finalize_result(
            source=source,
            settings=settings,
            output_dir=output_dir,
            provider=provider,
            model=model,
            text=text,
            segments=segments,
            language=str(payload.get("language") or settings.language or "unknown"),
            duration=_coerce_optional_float(payload.get("duration")),
            request_id=request_id,
            metadata={"raw_response": payload} if settings.provider_keep_raw_response else {},
        )

    def _build_deepgram_result(
        self,
        *,
        source: Path,
        settings: TranscriptionSettings,
        output_dir: str | Path,
        model: str,
        payload: dict[str, object],
        request_id: str | None,
    ) -> TranscriptionResult:
        segments, text, language, duration = _parse_deepgram_payload(payload)
        return _finalize_result(
            source=source,
            settings=settings,
            output_dir=output_dir,
            provider="deepgram",
            model=model,
            text=text,
            segments=segments,
            language=language or settings.language or "unknown",
            duration=duration,
            request_id=request_id,
            metadata={"raw_response": payload} if settings.provider_keep_raw_response else {},
        )

    def _build_assemblyai_result(
        self,
        *,
        source: Path,
        settings: TranscriptionSettings,
        output_dir: str | Path,
        model: str,
        payload: dict[str, object],
        request_id: str | None,
    ) -> TranscriptionResult:
        segments, text = _parse_assemblyai_payload(payload)
        return _finalize_result(
            source=source,
            settings=settings,
            output_dir=output_dir,
            provider="assemblyai",
            model=model,
            text=text,
            segments=segments,
            language=str(payload.get("language_code") or settings.language or "unknown"),
            duration=_coerce_optional_float(payload.get("audio_duration")),
            request_id=request_id,
            job_id=str(payload.get("id") or "") or None,
            metadata={"raw_response": payload} if settings.provider_keep_raw_response else {},
        )

    def _build_elevenlabs_result(
        self,
        *,
        source: Path,
        settings: TranscriptionSettings,
        output_dir: str | Path,
        model: str,
        payload: dict[str, object],
        request_id: str | None,
    ) -> TranscriptionResult:
        segments, text = _parse_elevenlabs_payload(payload)
        return _finalize_result(
            source=source,
            settings=settings,
            output_dir=output_dir,
            provider="elevenlabs",
            model=model,
            text=text,
            segments=segments,
            language=str(payload.get("language_code") or settings.language or "unknown"),
            duration=_coerce_optional_float(payload.get("audio_duration") or payload.get("duration_seconds")),
            request_id=request_id,
            metadata={"raw_response": payload} if settings.provider_keep_raw_response else {},
        )


def _finalize_result(
    *,
    source: Path,
    settings: TranscriptionSettings,
    output_dir: str | Path,
    provider: str,
    model: str,
    text: str,
    segments: list[SegmentData],
    language: str,
    duration: float | None,
    request_id: str | None,
    job_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> TranscriptionResult:
    apply_glossary_to_segments(segments, settings)
    normalized_text = text.strip() or to_plain_text(segments)
    normalized_text = apply_glossary_to_text(normalized_text, settings)
    base_name = build_base_name(source)
    output_root = Path(output_dir).expanduser().resolve()
    output_files = {
        file_format: output_root / f"{base_name}.{file_format}"
        for file_format in settings.output_formats
    }
    return TranscriptionResult(
        audio_path=source,
        base_name=base_name,
        provider=provider,
        model=model,
        language=language,
        language_probability=None,
        duration=duration,
        task=settings.task,
        device=f"remote:{provider}",
        compute_type="api",
        text=normalized_text,
        segments=segments,
        output_files=output_files,
        provider_job_id=job_id,
        provider_request_id=request_id,
        metadata=metadata or {},
    )


def _build_file_tuple(source: Path) -> tuple[str, bytes, str]:
    return source.name, source.read_bytes(), _guess_content_type(source)


def _guess_content_type(source: Path) -> str:
    return mimetypes.guess_type(source.name)[0] or "application/octet-stream"


def _ensure_json(response: httpx.Response) -> dict[str, object]:
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise ProviderError(f"Provider JSON donmedi: {exc}") from exc
    if isinstance(payload, dict):
        return payload
    raise ProviderError("Provider JSON yaniti beklenen dict formatinda degil.")


def _parse_openai_family_payload(payload: dict[str, object]) -> tuple[list[SegmentData], str]:
    text = str(payload.get("text") or "").strip()
    segments_data = payload.get("segments")
    if isinstance(segments_data, list):
        segments = [_coerce_segment(index, item) for index, item in enumerate(segments_data)]
        return segments, text or to_plain_text(segments)
    if text:
        return [_single_segment(text)], text
    raise ProviderError("Provider metin donmedi.")


def _parse_deepgram_payload(
    payload: dict[str, object],
) -> tuple[list[SegmentData], str, str | None, float | None]:
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ProviderError("Deepgram yaniti eksik veya bozuk.")
    utterances_data = results.get("utterances")
    if isinstance(utterances_data, list) and utterances_data:
        segments = [_coerce_deepgram_utterance(index, item) for index, item in enumerate(utterances_data)]
    else:
        channels = results.get("channels")
        if not isinstance(channels, list) or not channels:
            raise ProviderError("Deepgram channel bilgisi donmedi.")
        alternatives = channels[0].get("alternatives") if isinstance(channels[0], dict) else None
        if not isinstance(alternatives, list) or not alternatives:
            raise ProviderError("Deepgram alternatives bilgisi donmedi.")
        alternative = alternatives[0]
        words = alternative.get("words") if isinstance(alternative, dict) else None
        segments = _segments_from_words(words)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    duration = _coerce_optional_float(metadata.get("duration")) if isinstance(metadata, dict) else None
    summary_language = None
    channels = results.get("channels")
    if isinstance(channels, list) and channels and isinstance(channels[0], dict):
        alternatives = channels[0].get("alternatives")
        if isinstance(alternatives, list) and alternatives and isinstance(alternatives[0], dict):
            summary_language = alternatives[0].get("detected_language")
            transcript = str(alternatives[0].get("transcript") or "").strip()
        else:
            transcript = ""
    else:
        transcript = ""
    text = transcript or to_plain_text(segments)
    return segments, text, str(summary_language) if summary_language else None, duration


def _parse_assemblyai_payload(payload: dict[str, object]) -> tuple[list[SegmentData], str]:
    utterances = payload.get("utterances")
    if isinstance(utterances, list) and utterances:
        segments = [_coerce_assembly_utterance(index, item) for index, item in enumerate(utterances)]
        return segments, str(payload.get("text") or "").strip() or to_plain_text(segments)
    words = payload.get("words")
    segments = _segments_from_words(words, milliseconds=True)
    text = str(payload.get("text") or "").strip()
    if segments:
        return segments, text or to_plain_text(segments)
    if text:
        return [_single_segment(text)], text
    raise ProviderError("AssemblyAI metin donmedi.")


def _parse_elevenlabs_payload(payload: dict[str, object]) -> tuple[list[SegmentData], str]:
    words = payload.get("words") or payload.get("word_timestamps")
    segments = _segments_from_words(words)
    text = str(payload.get("text") or payload.get("transcript") or "").strip()
    if segments:
        return segments, text or to_plain_text(segments)
    if text:
        return [_single_segment(text)], text
    raise ProviderError("ElevenLabs metin donmedi.")


def _coerce_segment(index: int, raw_segment: object) -> SegmentData:
    if not isinstance(raw_segment, dict):
        return _single_segment(str(raw_segment), index=index)
    words = _coerce_words(raw_segment.get("words"))
    return SegmentData(
        index=index,
        start=_coerce_optional_float(raw_segment.get("start")) or 0.0,
        end=_coerce_optional_float(raw_segment.get("end")) or 0.0,
        text=str(raw_segment.get("text") or "").strip(),
        words=words,
        avg_logprob=_coerce_optional_float(raw_segment.get("avg_logprob")),
        compression_ratio=_coerce_optional_float(raw_segment.get("compression_ratio")),
        no_speech_prob=_coerce_optional_float(raw_segment.get("no_speech_prob")),
    )


def _coerce_deepgram_utterance(index: int, raw_utterance: object) -> SegmentData:
    if not isinstance(raw_utterance, dict):
        return _single_segment(str(raw_utterance), index=index)
    speaker = raw_utterance.get("speaker")
    return SegmentData(
        index=index,
        start=_coerce_optional_float(raw_utterance.get("start")) or 0.0,
        end=_coerce_optional_float(raw_utterance.get("end")) or 0.0,
        text=str(raw_utterance.get("transcript") or raw_utterance.get("text") or "").strip(),
        speaker=f"speaker_{speaker}" if speaker is not None else None,
        words=_coerce_words(raw_utterance.get("words")),
    )


def _coerce_assembly_utterance(index: int, raw_utterance: object) -> SegmentData:
    if not isinstance(raw_utterance, dict):
        return _single_segment(str(raw_utterance), index=index)
    speaker = raw_utterance.get("speaker")
    words = _coerce_words(raw_utterance.get("words"), milliseconds=True)
    return SegmentData(
        index=index,
        start=_coerce_optional_float(raw_utterance.get("start"), milliseconds=True) or 0.0,
        end=_coerce_optional_float(raw_utterance.get("end"), milliseconds=True) or 0.0,
        text=str(raw_utterance.get("text") or "").strip(),
        speaker=f"speaker_{speaker}" if speaker is not None else None,
        words=words,
    )


def _coerce_words(raw_words: object, *, milliseconds: bool = False) -> list[WordTiming]:
    if not isinstance(raw_words, list):
        return []
    words: list[WordTiming] = []
    for raw_word in raw_words:
        if not isinstance(raw_word, dict):
            continue
        text = str(
            raw_word.get("word")
            or raw_word.get("text")
            or raw_word.get("punctuated_word")
            or ""
        )
        if not text.strip():
            continue
        words.append(
            WordTiming(
                start=_coerce_optional_float(raw_word.get("start"), milliseconds=milliseconds) or 0.0,
                end=_coerce_optional_float(raw_word.get("end"), milliseconds=milliseconds) or 0.0,
                word=text,
                probability=_coerce_optional_float(raw_word.get("confidence") or raw_word.get("probability")),
            )
        )
    return words


def _segments_from_words(raw_words: object, *, milliseconds: bool = False) -> list[SegmentData]:
    words = _coerce_words(raw_words, milliseconds=milliseconds)
    if not words:
        return []
    segments: list[SegmentData] = []
    current: list[WordTiming] = []
    for word in words:
        if current and (word.start - current[-1].end) >= 1.25:
            segments.append(_segment_from_words(len(segments), current))
            current = []
        current.append(word)
    if current:
        segments.append(_segment_from_words(len(segments), current))
    return segments


def _segment_from_words(index: int, words: list[WordTiming]) -> SegmentData:
    text = "".join(word.word for word in words).strip()
    if not text:
        text = " ".join(word.word.strip() for word in words if word.word.strip())
    return SegmentData(
        index=index,
        start=words[0].start,
        end=words[-1].end,
        text=text,
        words=words,
    )


def _single_segment(text: str, *, index: int = 0) -> SegmentData:
    return SegmentData(index=index, start=0.0, end=0.0, text=text.strip())


def _coerce_optional_float(value: object, *, milliseconds: bool = False) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric / 1000.0 if milliseconds else numeric


def _parse_keywords(value: str) -> list[str]:
    keywords: list[str] = []
    for line in value.splitlines():
        cleaned = line.strip().strip(",")
        if cleaned:
            keywords.append(cleaned)
    return keywords
