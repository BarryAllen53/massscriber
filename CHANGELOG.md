# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog and the project follows Semantic Versioning.

## [Unreleased]

## [0.4.0] - 2026-03-06

- Added provider-side remote URL ingest for Deepgram and AssemblyAI, so large hosted jobs can run without uploading the file through the browser first.
- Added hosted-provider-to-local fallback, letting jobs continue with the local engine when an API rejects the file, times out, or otherwise fails.
- Added UI controls for remote audio URLs and hosted-to-local fallback.
- Added CLI flags for `--provider-remote-url` and `--provider-no-local-fallback`.
- Added provider help and doctor output details for remote URL support and upload-size limits.
- Added remote-URL provider tests plus hosted-to-local fallback tests.
- Fixed the remaining Linux CI issue in the Windows CUDA runtime test by patching the runtime module's `Path` constructor alongside `os.name`.
- Fixed hosted provider workflows so remote-URL-only CLI runs are supported without requiring a local input file.
- Fixed watch mode validation so remote URL settings cannot silently leak into folder-watch jobs.

## [0.3.3] - 2026-03-06

- Fixed the Linux CI failure by computing temporary Windows-path test inputs before patching `os.name` to `nt` in the CUDA runtime test.

## [0.3.2] - 2026-03-06

- Fixed the Windows CUDA runtime path test so it no longer fails on Linux runners where `os.add_dll_directory` is unavailable.

## [0.3.1] - 2026-03-06

- Fixed the GitHub release workflow to publish tagged releases through `softprops/action-gh-release`, avoiding the failing `gh` shell dependency on runners.

## [0.3.0] - 2026-03-06

- Added saved workflow profiles in the UI for reusable transcription, glossary, subtitle, and watch presets.
- Added a transcript library panel with text search, review-state filters, transcript previews, and batch review updates.
- Added a multi-provider transcription layer covering local Faster-Whisper, OpenAI, Groq, Deepgram, AssemblyAI, and ElevenLabs.
- Added hosted-provider controls to the UI and CLI for API keys, base URLs, timeout, polling, smart formatting, speaker labels, and keyword boosts.
- Added provider-aware exports that include provider names, remote request identifiers, and optional raw API payloads in JSON output.
- Added provider-aware diagnostics so the doctor panel shows environment-key readiness for each hosted API.
- Added provider-aware transcript library indexing so local and cloud results can be searched and batch-reviewed together.
- Added provider-aware workflow profiles so local and hosted transcription presets can be saved and reloaded.
- Added hosted-provider parsing tests plus provider utility tests.
- Added retry and timeout handling for transient hosted API failures.
- Fixed transcript review and result tables so provider source is visible alongside model and language metadata.
- Fixed CLI model selection so each provider now resolves to a sensible default model when `--model` is omitted.

## [0.2.0] - 2026-03-06

- Added configurable subtitle regrouping controls for cleaner `srt` and `vtt` exports.
- Added a `watch` CLI workflow with persistent state and optional archive moves.
- Added a UI watch panel for live folder scans and watch-history refreshes.
- Added experimental speaker diarization plumbing with optional `pyannote.audio` support.
- Added glossary-aware cleanup rules for transcript normalization across previews and subtitle exports.
- Added a `doctor` system-health command and a UI system status panel.
- Added Windows desktop bundle build scripts and a GitHub Actions desktop artifact workflow.

## [0.1.2] - 2026-03-06

- Added automatic CPU fallback when CUDA runtime DLLs like `cublas64_12.dll` are missing.
- Added automatic Windows DLL path registration for NVIDIA CUDA runtime Python packages before loading the whisper backend.
- Updated `install.bat` to auto-install NVIDIA CUDA runtime Python packages when an NVIDIA GPU is detected.

## [0.1.1] - 2026-03-06

### Added

- Added GitHub Actions CI for install, import, compile, and unit-test checks.
- Added tag-driven GitHub release automation.
- Added a dedicated release process document.
- Improved package metadata and project entry point configuration.
- Reworked the README for a more public, release-ready GitHub presentation.
- Added live stage-by-stage progress updates for long transcription jobs in the Gradio UI.
- Added streaming status logs so single large files no longer appear stuck at 0%.
- Added a local disk mode for direct file-path and folder-based transcription without browser upload.
- Enabled PWA support for the local Gradio app.
- Switched PyPI publishing workflow toward Trusted Publishing (OIDC) instead of a long-lived API token.

## [0.1.0] - 2026-03-06

### Added

- Initial local transcription app built with `faster-whisper`
- Gradio UI for batch transcription
- CLI mode for scripted usage
- Export support for `txt`, `srt`, `vtt`, and `json`
- Windows helper scripts for installation and startup
