import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from massscriber import transcriber
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


class FakeCudaFailingModel:
    def transcribe(self, _source: str, **_kwargs):
        raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")


class TranscriberProgressTests(TestCase):
    def test_configure_windows_cuda_runtime_paths_registers_nvidia_bins(self):
        with TemporaryDirectory() as tmpdir:
            site_packages = Path(tmpdir) / "site-packages"
            cublas_dir = site_packages / "nvidia" / "cublas" / "bin"
            cudnn_dir = site_packages / "nvidia" / "cudnn" / "bin"
            cublas_dir.mkdir(parents=True)
            cudnn_dir.mkdir(parents=True)

            fake_handles = [object(), object()]
            with patch.object(transcriber.os, "name", "nt"), patch.object(
                transcriber.os,
                "add_dll_directory",
                side_effect=fake_handles,
            ) as add_dll_directory, patch.object(
                transcriber.site,
                "getsitepackages",
                return_value=[str(site_packages)],
            ), patch.object(
                transcriber.site,
                "getusersitepackages",
                return_value=str(site_packages),
            ), patch.object(
                transcriber.sys,
                "prefix",
                str(Path(tmpdir) / "venv"),
            ), patch.object(
                transcriber.sys,
                "base_prefix",
                str(Path(tmpdir) / "base"),
            ), patch.dict(
                transcriber.os.environ,
                {"PATH": "C:\\Windows\\System32"},
                clear=True,
            ):
                transcriber.WINDOWS_DLL_HANDLES.clear()
                added_dirs = transcriber.configure_windows_cuda_runtime_paths()
                path_entries = transcriber.os.environ["PATH"].split(os.pathsep)

            self.assertEqual(added_dirs, [cublas_dir.resolve(), cudnn_dir.resolve()])
            self.assertEqual(add_dll_directory.call_count, 2)
            self.assertEqual(transcriber.WINDOWS_DLL_HANDLES, fake_handles)
            self.assertEqual(path_entries[0], str(cudnn_dir.resolve()))
            self.assertEqual(path_entries[1], str(cublas_dir.resolve()))

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

    def test_stream_file_falls_back_to_cpu_when_cuda_runtime_is_missing(self):
        settings = TranscriptionSettings(model="tiny", device="auto", output_formats=("txt",))
        engine = TranscriptionEngine()

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            audio_path = tmp_path / "sample.wav"
            audio_path.write_bytes(b"fake")

            with patch.object(
                WhisperRuntime,
                "get_backend",
                side_effect=[
                    (FakeCudaFailingModel(), None, "cuda", "float16"),
                    (FakeModel(), None, "cpu", "int8"),
                ],
            ), patch.object(WhisperRuntime, "clear_backend") as clear_backend:
                events = list(engine.stream_file(audio_path, settings, tmp_path / "outputs"))

                self.assertTrue(any("[UYARI]" in message for _, message, _ in events))
                self.assertTrue(any("CPU moduna geciliyor" in message for _, message, _ in events))

                result = events[-1][2]
                self.assertIsNotNone(result)
                self.assertEqual(result.device, "cpu")
                self.assertEqual(result.compute_type, "int8")
                clear_backend.assert_called_once()
