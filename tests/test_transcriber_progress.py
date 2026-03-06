from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from massscriber.transcriber import TranscriptionEngine, WhisperRuntime
from massscriber.types import TranscriptionSettings


class FakeModel:
    def transcribe(self, _source: str, **_kwargs):
        segments = [
            SimpleNamespace(start=0.0, end=5.0, text=" Merhaba", words=[]),
            SimpleNamespace(start=5.0, end=10.0, text=" dunya", words=[]),
        ]
        info = SimpleNamespace(language="tr", language_probability=0.99, duration=10.0)
        return iter(segments), info


class TranscriberProgressTests(TestCase):
    def test_stream_file_emits_intermediate_progress_and_result(self):
        settings = TranscriptionSettings(model="tiny", output_formats=("txt",))
        engine = TranscriptionEngine()

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            audio_path = tmp_path / "sample.wav"
            audio_path.write_bytes(b"fake")

            with patch.object(
                WhisperRuntime,
                "get_backend",
                return_value=(FakeModel(), None, "cpu", "int8"),
            ):
                events = list(engine.stream_file(audio_path, settings, tmp_path / "outputs"))

                self.assertTrue(any("model hazirlaniyor" in message for _, message, _ in events))
                self.assertTrue(any("transkribe ediliyor" in message for _, message, _ in events))
                self.assertTrue(any(0.0 < progress < 1.0 for progress, _, _ in events))

                result = events[-1][2]
                self.assertIsNotNone(result)
                self.assertEqual(result.text, "Merhaba dunya")
                self.assertTrue(result.output_files["txt"].exists())
