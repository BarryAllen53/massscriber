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
- Configurable subtitle segmentation for cleaner `srt` and `vtt` exports
- Experimental speaker diarization with optional `pyannote.audio` support
- Folder watch CLI workflow for auto-transcribing new media files
- UI watch panel with live logs and history refresh
- Glossary-aware transcript cleanup for names, brands, and recurring corrections
- Saved workflow profiles for watch, glossary, and transcription presets
- Transcript library search with batch review status tracking
- Built-in system health panel plus `doctor` CLI command
- Live stage-by-stage progress for long-running transcriptions
- CLI mode for automation and power users
- Export formats: `txt`, `srt`, `vtt`, `json`
- Quality-first model option: `large-v3`
- Speed-first model option: `turbo`
- Installable in supported browsers as a PWA
- Windows desktop bundle build script and GitHub Actions artifact workflow

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

The UI also now includes:

- a live system status panel for CUDA/runtime health
- a watch panel for repeated folder scans
- glossary rules for post-transcription cleanup
- saved workflow profiles for repeatable presets
- a transcript library panel for search and batch review

### Use the CLI

```powershell
.venv\Scripts\activate
python app.py transcribe "C:\audio\meeting.mp3" --model large-v3 --formats txt srt json
```

You can also use the installed console entry point:

```powershell
massscriber transcribe "C:\audio\meeting.mp3" --model turbo --formats txt srt
```

### Watch a folder for new files

```powershell
massscriber watch "C:\audio\incoming" --model turbo --once
```

For a long-running workflow:

```powershell
massscriber watch "C:\audio\incoming" --model turbo --archive-dir "C:\audio\done"
```

### Check system health

```powershell
massscriber doctor
```

## Workflow Profiles

If you reuse the same combinations of model, glossary rules, watch folder, or subtitle settings, save them as a profile in the UI.

- Save the current form values into a named reusable preset
- Reload a preset with one click before starting a job
- Delete or refresh saved profile lists without leaving the app

This is useful for keeping separate setups like:

- podcast cleanup
- meeting transcription
- TV episode subtitle prep
- folder-watch automation for incoming recordings

## Transcript Library and Batch Review

The UI now includes a transcript library panel that scans your output directory and builds a searchable review table.

- Search transcript text and metadata from previous runs
- Filter by review state: `pending`, `reviewed`, `needs-edit`, `approved`
- Preview transcript snippets before opening files manually
- Apply review status updates to selected transcript IDs
- Bulk-apply a review status to the visible filtered result set

This gives you a lightweight local review workflow without needing a separate database service.

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

### Better subtitles

- Subtitle max chars: `36` to `48`
- Subtitle max duration: `4.0` to `6.0`
- Pause split: enabled

### Glossary cleanup

- Use `Source => Target` format, one rule per line
- Great for names, brands, product terms, and repeated OCR-like mistakes
- Works in UI, CLI, and watch workflows

Example:

```text
Open AI => OpenAI
Chat GPT => ChatGPT
Baris Manco => Barış Manço
```

### Experimental speaker diarization

- Enable only when you really need speaker labels
- Install the optional extra first:

```powershell
python -m pip install -e ".[diarization]"
```

- Provide a Hugging Face token either through `HUGGINGFACE_HUB_TOKEN` or the UI/CLI field
- Default model: `pyannote/speaker-diarization-3.1`

## Outputs

By default, transcripts are written to the `outputs` directory:

- `txt`
- `srt`
- `json`

You can also enable `vtt` from the UI or CLI.

SRT and VTT exports now use configurable subtitle regrouping, so long whisper segments can be re-cut into shorter subtitle cues.

## GPU Notes

If you use an NVIDIA GPU, `faster-whisper` may require CUDA runtime libraries on your machine.
CPU mode works too; it is simply slower.
If CUDA libraries such as `cublas64_12.dll` are missing, Massscriber now falls back to CPU automatically and logs a clear warning instead of stopping the job.
On Windows, Massscriber also auto-registers NVIDIA runtime DLL folders from installed Python packages before loading `ctranslate2`, which helps GPU mode come up cleanly after `install.bat`.

According to the `faster-whisper` recommendations:

- `int8` is a good default for CPU execution
- `float16` or `int8_float16` are good GPU options

If you hit CUDA DLL issues on Windows, follow the Windows notes in the official `faster-whisper` documentation.

## Desktop Builds

To create a Windows desktop bundle locally:

```bat
build_desktop.bat
```

Or manually:

```powershell
python -m pip install -e ".[desktop]"
python build_desktop.py
```

GitHub Actions also includes a Windows desktop build workflow that uploads a `Massscriber-windows` artifact for tagged releases and manual runs.

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

## Roadmap Progress

- Experimental speaker diarization support is now wired in as an optional extra.
- Subtitle exports now have configurable regrouping controls for better cue sizing.
- Folder watch and auto-transcribe workflows now exist in the CLI with persistent state and optional archiving.
- Folder watch is now available in the UI for repeated scan sessions.
- Glossary-aware cleanup and a built-in doctor/status surface are now part of the core app.
- Saved workflow profiles are now available for recurring transcription setups.
- Transcript library search and batch review are now built into the UI.
- Desktop packaging now has a local build script and a Windows artifact workflow.

## Next Roadmap

- Improve diarization with speaker-aware word-level subtitle cues
- Produce signed desktop installers instead of raw bundles
- Add project-level transcript libraries for large collections
- Add persistent batch actions such as export queues and review assignments
- Add richer transcript editing and glossary-assisted correction workflows

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## References

- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [openai/whisper](https://github.com/openai/whisper)
