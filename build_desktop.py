from __future__ import annotations

from pathlib import Path

import PyInstaller.__main__


def main() -> None:
    project_root = Path(__file__).resolve().parent
    app_path = project_root / "app.py"

    PyInstaller.__main__.run(
        [
            str(app_path),
            "--noconfirm",
            "--clean",
            "--onedir",
            "--name",
            "Massscriber",
            "--collect-all",
            "gradio",
            "--collect-all",
            "gradio_client",
            "--collect-all",
            "faster_whisper",
            "--collect-all",
            "av",
            "--hidden-import",
            "ctranslate2",
            "--copy-metadata",
            "massscriber",
        ]
    )


if __name__ == "__main__":
    main()
