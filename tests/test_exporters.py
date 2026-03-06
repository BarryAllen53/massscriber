import unittest

from massscriber.exporters import build_srt, format_timestamp, sanitize_name, to_plain_text
from massscriber.types import SegmentData, TranscriptionResult


class ExporterTests(unittest.TestCase):
    def test_format_timestamp_for_srt(self):
        self.assertEqual(format_timestamp(65.432, decimal_marker=","), "00:01:05,432")

    def test_plain_text_collapses_extra_spaces(self):
        segments = [
            SegmentData(index=0, start=0.0, end=1.0, text=" Merhaba   "),
            SegmentData(index=1, start=1.0, end=2.0, text=" dunya"),
        ]
        self.assertEqual(to_plain_text(segments), "Merhaba dunya")

    def test_sanitize_name_removes_unsafe_characters(self):
        self.assertEqual(sanitize_name("Meeting Notes / Final?"), "Meeting_Notes_Final")

    def test_build_srt_emits_expected_blocks(self):
        result = TranscriptionResult(
            audio_path=__file__,
            base_name="demo",
            model="tiny",
            language="en",
            language_probability=0.99,
            duration=2.0,
            task="transcribe",
            device="cpu",
            compute_type="int8",
            text="Hello world",
            segments=[
                SegmentData(index=0, start=0.0, end=1.0, text=" Hello"),
                SegmentData(index=1, start=1.0, end=2.0, text=" world"),
            ],
            output_files={},
        )
        srt = build_srt(result)
        self.assertIn("1\n00:00:00,000 --> 00:00:01,000\nHello", srt)
        self.assertIn("2\n00:00:01,000 --> 00:00:02,000\nworld", srt)


if __name__ == "__main__":
    unittest.main()
