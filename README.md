# Massscriber

[![CI](https://github.com/BarryAllen53/massscriber/actions/workflows/ci.yml/badge.svg)](https://github.com/BarryAllen53/massscriber/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Massscriber is a local-first audio transcription app built on top of `faster-whisper`.
It is designed for people who want:

- no paid API dependency
- no upload requirement for private recordings
- unlimited local processing
- strong multilingual transcription quality
- both a simple UI and a scriptable CLI

## Highlights

- Local transcription with `faster-whisper`
- Multilingual speech-to-text with automatic language detection
- Batch processing for multiple audio or video files
- Gradio UI for drag-and-drop use
- Local disk mode for direct file paths or folder scans without browser upload
- Live stage-by-stage progress for long-running transcriptions
- CLI mode for automation and power users
- Export formats: `txt`, `srt`, `vtt`, `json`
- Quality-first model option: `large-v3`
- Speed-first model option: `turbo`
- Installable in supported browsers as a PWA

## Supported Inputs

The UI currently accepts:

- `.mp3`
- `.wav`
- `.m4a`
- `.flac`
- `.ogg`
- `.aac`
- `.wma`
- `.mp4`
- `.mkv`

`faster-whisper` handles media decoding through its own stack, so a manual `ffmpeg` install is usually not required.

## Model Guidance

| Model | Best for | Notes |
| --- | --- | --- |
| `large-v3` | Highest accuracy | Best default for serious transcription work |
| `turbo` | Fastest practical transcription | Great speed/quality balance |
| `medium` | Mid-range systems | Useful fallback for lower VRAM devices |
| `small`, `base`, `tiny` | Lightweight testing | Faster, but lower accuracy |

Important note: no speech recognition model is perfectly error-free. For a fully free and local workflow, `large-v3` is one of the strongest practical choices available today.

## Quick Start

### Windows helper scripts

```bat
install.bat
start_ui.bat
```

On Windows, `install.bat` now checks for `nvidia-smi` and, when an NVIDIA GPU is present, also installs the NVIDIA CUDA runtime Python packages that provide DLLs such as `cublas64_12.dll` and `cudnn64_9.dll`.

### Manual setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run the App

### Launch the UI

```powershell
.venv\Scripts\activate
python app.py
```

The default UI address is `http://127.0.0.1:7860`.
In supported browsers, the app can also be installed as a PWA for quicker relaunching.

If browser upload becomes unreliable for very large media, use the local disk mode in the UI:

- paste one or more absolute file paths
- or point the app at a folder and let it scan supported media files

### Use the CLI

```powershell
.venv\Scripts\activate
python app.py transcribe "C:\audio\meeting.mp3" --model large-v3 --formats txt srt json
```

You can also use the installed console entry point:

```powershell
massscriber transcribe "C:\audio\meeting.mp3" --model turbo --formats txt srt
```

## Recommended Settings

### Best quality

- Model: `large-v3`
- Beam size: `5`
- VAD: enabled
- Word timestamps: enabled

### Best speed

- Model: `turbo`
- Device: `cuda` when available
- Compute type: `float16`
- Batch size: `8` or `16`

## Outputs

By default, transcripts are written to the `outputs` directory:

- `txt`
- `srt`
- `json`

You can also enable `vtt` from the UI or CLI.

## GPU Notes

If you use an NVIDIA GPU, `faster-whisper` may require CUDA runtime libraries on your machine.
CPU mode works too; it is simply slower.
If CUDA libraries such as `cublas64_12.dll` are missing, Massscriber now falls back to CPU automatically and logs a clear warning instead of stopping the job.
On Windows, Massscriber also auto-registers NVIDIA runtime DLL folders from installed Python packages before loading `ctranslate2`, which helps GPU mode come up cleanly after `install.bat`.

According to the `faster-whisper` recommendations:

- `int8` is a good default for CPU execution
- `float16` or `int8_float16` are good GPU options

If you hit CUDA DLL issues on Windows, follow the Windows notes in the official `faster-whisper` documentation.

## First Run Behavior

The selected model is downloaded automatically on first use.
For `large-v3`, the first run can take a while because the model is large.

## Development

### Run tests

```powershell
python -m unittest discover -s tests -v
```

### Local verification

```powershell
python -m py_compile app.py massscriber\__init__.py massscriber\types.py massscriber\exporters.py massscriber\transcriber.py massscriber\ui.py
```

## Versioning and Releases

- Project version is defined in `massscriber.__version__`
- Packaging reads the version dynamically from the package
- Human-readable release history lives in [CHANGELOG.md](CHANGELOG.md)
- Release steps are documented in [RELEASING.md](RELEASING.md)
- Pushing a tag like `v0.1.0` triggers the GitHub release workflow

## Roadmap

- Speaker diarization
- Better subtitle segmentation options
- Packaged desktop builds
- Folder watch / auto-transcribe workflows

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## References

- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [openai/whisper](https://github.com/openai/whisper)
