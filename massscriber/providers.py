from __future__ import annotations

import os

PROVIDERS = (
    "local",
    "openai",
    "groq",
    "deepgram",
    "assemblyai",
    "elevenlabs",
)

PROVIDER_LABELS = {
    "local": "Local Faster-Whisper",
    "openai": "OpenAI Speech-to-Text",
    "groq": "Groq Speech-to-Text",
    "deepgram": "Deepgram Nova",
    "assemblyai": "AssemblyAI",
    "elevenlabs": "ElevenLabs Scribe",
}

PROVIDER_MODELS = {
    "local": ["large-v3", "turbo", "medium", "small", "base", "tiny"],
    "openai": ["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"],
    "groq": ["whisper-large-v3", "whisper-large-v3-turbo", "distil-whisper-large-v3-en"],
    "deepgram": ["nova-3", "nova-2", "enhanced"],
    "assemblyai": ["universal", "best"],
    "elevenlabs": ["scribe_v1", "scribe_v1_experimental"],
}

DEFAULT_PROVIDER_MODELS = {
    "local": "large-v3",
    "openai": "whisper-1",
    "groq": "whisper-large-v3-turbo",
    "deepgram": "nova-3",
    "assemblyai": "universal",
    "elevenlabs": "scribe_v1",
}

PROVIDER_API_KEY_ENVS = {
    "local": (),
    "openai": ("OPENAI_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "deepgram": ("DEEPGRAM_API_KEY",),
    "assemblyai": ("ASSEMBLYAI_API_KEY",),
    "elevenlabs": ("ELEVENLABS_API_KEY",),
}

PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "deepgram": "https://api.deepgram.com/v1",
    "assemblyai": "https://api.assemblyai.com/v2",
    "elevenlabs": "https://api.elevenlabs.io/v1",
}

REMOTE_URL_SUPPORTED = {
    "local": False,
    "openai": False,
    "groq": False,
    "deepgram": True,
    "assemblyai": True,
    "elevenlabs": False,
}

TRANSLATION_SUPPORTED = {
    "local": True,
    "openai": True,
    "groq": False,
    "deepgram": False,
    "assemblyai": False,
    "elevenlabs": False,
}

WORD_TIMESTAMPS_SUPPORTED = {
    "local": True,
    "openai": True,
    "groq": True,
    "deepgram": True,
    "assemblyai": True,
    "elevenlabs": True,
}

SPEAKER_LABEL_SUPPORTED = {
    "local": False,
    "openai": False,
    "groq": False,
    "deepgram": True,
    "assemblyai": True,
    "elevenlabs": True,
}

PROVIDER_FILE_LIMIT_MB = {
    "local": 0,
    "openai": 25,
    "groq": 25,
    "deepgram": 2000,
    "assemblyai": 2000,
    "elevenlabs": 1000,
}


def normalize_provider_name(value: str | None) -> str:
    normalized = (value or "local").strip().lower()
    return normalized if normalized in PROVIDERS else "local"


def provider_uses_remote_api(provider: str | None) -> bool:
    return normalize_provider_name(provider) != "local"


def get_provider_default_model(provider: str | None) -> str:
    normalized = normalize_provider_name(provider)
    return DEFAULT_PROVIDER_MODELS[normalized]


def get_provider_models(provider: str | None) -> list[str]:
    normalized = normalize_provider_name(provider)
    return list(PROVIDER_MODELS[normalized])


def resolve_provider_model(provider: str | None, requested_model: str | None, fallback_model: str) -> str:
    requested = (requested_model or "").strip()
    if requested:
        return requested
    if fallback_model.strip():
        return fallback_model.strip()
    return get_provider_default_model(provider)


def get_provider_api_key(provider: str | None, explicit_key: str | None = None) -> str | None:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    normalized = normalize_provider_name(provider)
    for env_name in PROVIDER_API_KEY_ENVS[normalized]:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return None


def get_provider_env_keys(provider: str | None) -> tuple[str, ...]:
    normalized = normalize_provider_name(provider)
    return PROVIDER_API_KEY_ENVS[normalized]


def get_provider_base_url(provider: str | None, override: str | None = None) -> str:
    if override and override.strip():
        return override.strip().rstrip("/")
    normalized = normalize_provider_name(provider)
    return PROVIDER_BASE_URLS[normalized]


def provider_supports_remote_url(provider: str | None) -> bool:
    return REMOTE_URL_SUPPORTED[normalize_provider_name(provider)]


def provider_supports_translation(provider: str | None) -> bool:
    return TRANSLATION_SUPPORTED[normalize_provider_name(provider)]


def provider_supports_word_timestamps(provider: str | None) -> bool:
    return WORD_TIMESTAMPS_SUPPORTED[normalize_provider_name(provider)]


def provider_supports_speaker_labels(provider: str | None) -> bool:
    return SPEAKER_LABEL_SUPPORTED[normalize_provider_name(provider)]


def get_provider_file_limit_mb(provider: str | None) -> int:
    return PROVIDER_FILE_LIMIT_MB[normalize_provider_name(provider)]


def provider_file_limit_warning(provider: str | None, file_size_bytes: int) -> str | None:
    limit_mb = get_provider_file_limit_mb(provider)
    if limit_mb <= 0:
        return None
    limit_bytes = limit_mb * 1024 * 1024
    if file_size_bytes <= limit_bytes:
        return None
    normalized = normalize_provider_name(provider)
    return (
        f"{PROVIDER_LABELS[normalized]} icin dosya boyutu siniri yaklasik {limit_mb} MB. "
        "Bu dosya icin local engine ya da URL destekleyen bir provider kullan."
    )


def redact_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"
