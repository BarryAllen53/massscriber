import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from massscriber.types import SegmentData, TranscriptionResult, TranscriptionSettings
from massscriber.watcher import STATE_FILE_NAME, watch_folder


class WatcherTests(TestCase):
    @staticmethod
    def _stream_result(result: TranscriptionResult):
        yield 0.1, f"{result.audio_path.name}: basladi", None
        yield 1.0, f"{result.audio_path.name}: tamamlandi", result

    def test_watch_folder_processes_stable_files_once_and_persists_state(self):
        settings = TranscriptionSettings(model="tiny", output_formats=("txt",))

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            watch_dir = root / "watch"
            output_dir = root / "outputs"
            watch_dir.mkdir()
            audio_path = watch_dir / "episode.mp3"
            audio_path.write_bytes(b"fake")

            fake_result = TranscriptionResult(
                audio_path=audio_path,
                base_name="episode",
                model="tiny",
                language="en",
                language_probability=0.99,
                duration=12.0,
                task="transcribe",
                device="cpu",
                compute_type="int8",
                text="hello world",
                segments=[SegmentData(index=0, start=0.0, end=1.0, text=" hello world")],
                output_files={"txt": output_dir / "episode.txt"},
            )

            with patch(
                "massscriber.watcher.TranscriptionEngine.stream_file",
                return_value=self._stream_result(fake_result),
            ) as transcribe:
                events = list(
                    watch_folder(
                        watch_dir,
                        settings,
                        output_dir,
                        once=True,
                        stable_seconds=0.0,
                    )
                )

            self.assertTrue(any("[OK] episode.mp3" in event for event in events))
            transcribe.assert_called_once()

            state_file = output_dir / STATE_FILE_NAME
            self.assertTrue(state_file.exists())
            state_payload = json.loads(state_file.read_text(encoding="utf-8"))
            self.assertIn(str(audio_path.resolve()), state_payload)

            with patch("massscriber.watcher.TranscriptionEngine.stream_file") as transcribe_again:
                second_events = list(
                    watch_folder(
                        watch_dir,
                        settings,
                        output_dir,
                        once=True,
                        stable_seconds=0.0,
                    )
                )

            transcribe_again.assert_not_called()
            self.assertTrue(any("yeni dosya bulunmadi" in event for event in second_events))

    def test_watch_folder_can_archive_processed_files(self):
        settings = TranscriptionSettings(model="tiny", output_formats=("txt",))

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            watch_dir = root / "watch"
            archive_dir = root / "archive"
            output_dir = root / "outputs"
            watch_dir.mkdir()
            audio_path = watch_dir / "nested" / "meeting.wav"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"fake")

            fake_result = TranscriptionResult(
                audio_path=audio_path,
                base_name="meeting",
                model="tiny",
                language="tr",
                language_probability=0.99,
                duration=3.0,
                task="transcribe",
                device="cpu",
                compute_type="int8",
                text="merhaba",
                segments=[SegmentData(index=0, start=0.0, end=1.0, text=" merhaba")],
                output_files={"txt": output_dir / "meeting.txt"},
            )

            with patch(
                "massscriber.watcher.TranscriptionEngine.stream_file",
                return_value=self._stream_result(fake_result),
            ):
                events = list(
                    watch_folder(
                        watch_dir,
                        settings,
                        output_dir,
                        once=True,
                        stable_seconds=0.0,
                        archive_dir=archive_dir,
                    )
                )

            self.assertTrue(any("[ARSIV] meeting.wav" in event for event in events))
            self.assertFalse(audio_path.exists())
            self.assertTrue((archive_dir / "nested" / "meeting.wav").exists())
