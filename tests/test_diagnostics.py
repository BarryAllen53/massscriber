from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from massscriber.diagnostics import detect_system_status, render_system_status


class DiagnosticsTests(TestCase):
    def test_detect_system_status_reports_output_writability(self):
        with TemporaryDirectory() as tmpdir:
            status = detect_system_status(tmpdir)

        self.assertEqual(status["output_dir"], str(Path(tmpdir).resolve()))
        self.assertTrue(status["output_writable"])

    def test_render_system_status_includes_core_sections(self):
        markdown = render_system_status(
            {
                "python": "3.11.0",
                "platform": "Windows",
                "cuda_devices": 1,
                "nvidia_smi": "C:\\Windows\\nvidia-smi.exe",
                "cuda_runtime_dirs": ["C:\\cuda"],
                "pyinstaller_available": True,
                "diarization_available": False,
                "output_dir": "D:\\outputs",
                "output_writable": True,
                "providers": {
                    "local": {"env_keys": [], "configured": True},
                    "openai": {"env_keys": ["OPENAI_API_KEY"], "configured": False},
                },
            }
        )

        self.assertIn("Sistem Durumu", markdown)
        self.assertIn("CUDA", markdown)
        self.assertIn("Desktop build araci", markdown)
        self.assertIn("openai", markdown)

    def test_detect_system_status_handles_missing_optional_modules(self):
        with TemporaryDirectory() as tmpdir, patch(
            "massscriber.diagnostics._module_available",
            return_value=False,
        ):
            status = detect_system_status(tmpdir)

        self.assertFalse(status["diarization_available"])
