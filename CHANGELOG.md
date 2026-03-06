# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog and the project follows Semantic Versioning.

## [Unreleased]

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
