from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from massscriber.ui import build_arg_parser, build_settings_from_args, collect_input_files


class UiInputTests(TestCase):
    def test_build_settings_from_cli_args_captures_subtitle_and_diarization_options(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "transcribe",
                "demo.mp3",
                "--provider",
                "openai",
                "--subtitle-max-chars",
                "48",
                "--subtitle-max-duration",
                "5.5",
                "--subtitle-pause-threshold",
                "0.8",
                "--enable-diarization",
                "--diarization-model",
                "pyannote/test-model",
                "--diarization-token",
                "token-123",
                "--glossary-text",
                "Open AI => OpenAI",
                "--glossary-case-sensitive",
                "--provider-api-key",
                "sk-demo",
                "--provider-base-url",
                "https://proxy.example/v1",
                "--provider-remote-url",
                "https://cdn.example/audio.mp3",
                "--provider-speaker-labels",
                "--provider-keywords",
                "OpenAI\nMassscriber",
                "--provider-keep-raw-response",
                "--provider-no-local-fallback",
            ]
        )

        settings = build_settings_from_args(args)

        self.assertEqual(settings.provider, "openai")
        self.assertEqual(settings.subtitle_max_chars, 48)
        self.assertEqual(settings.subtitle_max_duration, 5.5)
        self.assertEqual(settings.subtitle_pause_threshold, 0.8)
        self.assertTrue(settings.enable_diarization)
        self.assertEqual(settings.diarization_model, "pyannote/test-model")
        self.assertEqual(settings.diarization_token, "token-123")
        self.assertEqual(settings.glossary_text, "Open AI => OpenAI")
        self.assertTrue(settings.glossary_case_sensitive)
        self.assertEqual(settings.provider_api_key, "sk-demo")
        self.assertEqual(settings.provider_base_url, "https://proxy.example/v1")
        self.assertEqual(settings.provider_remote_url, "https://cdn.example/audio.mp3")
        self.assertTrue(settings.provider_speaker_labels)
        self.assertTrue(settings.provider_keep_raw_response)
        self.assertFalse(settings.provider_fallback_to_local)

    def test_collect_input_files_combines_upload_manual_and_folder_sources(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            upload_file = root / "upload.mp3"
            upload_file.write_bytes(b"upload")

            manual_file = root / "manual.wav"
            manual_file.write_bytes(b"manual")

            folder = root / "batch"
            folder.mkdir()
            folder_file = folder / "folder.flac"
            folder_file.write_bytes(b"folder")

            nested_folder = folder / "nested"
            nested_folder.mkdir()
            nested_file = nested_folder / "nested.ogg"
            nested_file.write_bytes(b"nested")

            ignored = folder / "ignore.txt"
            ignored.write_text("ignore", encoding="utf-8")

            files, warnings = collect_input_files(
                [str(upload_file)],
                f'"{manual_file}"',
                str(folder),
                recursive_scan=True,
            )

        self.assertEqual(
            [path.name for path in files],
            ["upload.mp3", "manual.wav", "folder.flac", "nested.ogg"],
        )
        self.assertEqual(warnings, [])

    def test_collect_input_files_reports_missing_or_unsupported_sources(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            unsupported = root / "bad.txt"
            unsupported.write_text("bad", encoding="utf-8")

            files, warnings = collect_input_files(
                None,
                str(unsupported),
                str(root / "missing-folder"),
                recursive_scan=False,
            )

        self.assertEqual(files, [])
        self.assertEqual(len(warnings), 2)
