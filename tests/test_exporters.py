import unittest
from pathlib import Path

from massscriber.exporters import format_timestamp, to_plain_text
from massscriber.types import SegmentData


class ExporterTests(unittest.TestCase):
    def test_format_timestamp_for_srt(self):
        self.assertEqual(format_timestamp(65.432, decimal_marker=","), "00:01:05,432")

    def test_plain_text_collapses_extra_spaces(self):
        segments = [
            SegmentData(index=0, start=0.0, end=1.0, text=" Merhaba   "),
            SegmentData(index=1, start=1.0, end=2.0, text=" dunya"),
        ]
        self.assertEqual(to_plain_text(segments), "Merhaba dunya")


if __name__ == "__main__":
    unittest.main()
