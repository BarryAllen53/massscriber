from __future__ import annotations

import json
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
APP_STATE_DIR = APP_ROOT / ".massscriber"
PROFILE_FILE = APP_STATE_DIR / "profiles.json"


def load_profiles() -> dict[str, dict[str, object]]:
    if not PROFILE_FILE.exists():
        return {}
    try:
        payload = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_profile(name: str, payload: dict[str, object]) -> dict[str, dict[str, object]]:
    cleaned_name = name.strip()
    if not cleaned_name:
        raise ValueError("Profil adi bos olamaz.")

    profiles = load_profiles()
    profiles[cleaned_name] = payload
    APP_STATE_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_FILE.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return profiles


def delete_profile(name: str) -> dict[str, dict[str, object]]:
    profiles = load_profiles()
    profiles.pop(name, None)
    APP_STATE_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_FILE.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return profiles


def get_profile(name: str) -> dict[str, object] | None:
    return load_profiles().get(name)


def list_profile_names() -> list[str]:
    return sorted(load_profiles())
