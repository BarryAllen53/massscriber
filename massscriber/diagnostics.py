from __future__ import annotations

import platform
import shutil
from pathlib import Path

from massscriber.transcriber import CONFIGURED_WINDOWS_CUDA_DIRS


def detect_system_status(output_dir: str | Path | None = None) -> dict[str, object]:
    output_root = Path(output_dir).expanduser().resolve() if output_dir else None
    output_writable = None
    if output_root is not None:
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            probe = output_root / ".write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            output_writable = True
        except OSError:
            output_writable = False

    try:
        import ctranslate2  # noqa: PLC0415

        cuda_devices = ctranslate2.get_cuda_device_count()
    except Exception:
        cuda_devices = 0

    nvidia_smi = shutil.which("nvidia-smi")
    pyinstaller_available = shutil.which("pyinstaller") is not None
    diarization_available = _module_available("pyannote.audio")

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_devices": cuda_devices,
        "nvidia_smi": nvidia_smi or "",
        "cuda_runtime_dirs": [str(path) for path in CONFIGURED_WINDOWS_CUDA_DIRS],
        "pyinstaller_available": pyinstaller_available,
        "diarization_available": diarization_available,
        "output_dir": str(output_root) if output_root else "",
        "output_writable": output_writable,
    }


def render_system_status(status: dict[str, object]) -> str:
    cuda_ready = "hazir" if int(status["cuda_devices"]) > 0 else "hazir degil"
    output_status = "bilinmiyor"
    if status["output_writable"] is True:
        output_status = "yazilabilir"
    elif status["output_writable"] is False:
        output_status = "yazilamiyor"

    lines = [
        "## Sistem Durumu",
        f"- Python: `{status['python']}`",
        f"- Platform: `{status['platform']}`",
        f"- CUDA: `{cuda_ready}` ({status['cuda_devices']} cihaz)",
        f"- nvidia-smi: `{status['nvidia_smi'] or 'bulunamadi'}`",
        f"- NVIDIA DLL klasorleri: `{len(status['cuda_runtime_dirs'])}` adet",
        f"- Desktop build araci: `{'hazir' if status['pyinstaller_available'] else 'yok'}`",
        f"- Diarization bagimliligi: `{'hazir' if status['diarization_available'] else 'yok'}`",
    ]
    if status["output_dir"]:
        lines.append(f"- Cikti klasoru: `{status['output_dir']}` ({output_status})")
    return "\n".join(lines)


def _module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False
