import unittest

from massscriber.exporters import build_srt, build_vtt, format_timestamp, sanitize_name, to_plain_text
from massscriber.types import SegmentData, TranscriptionResult, TranscriptionSettings, WordTiming


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
            provider="local",
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
        self.assertIn("1\n00:00:00,000 --> 00:00:02,000\nHello world", srt)

    def test_build_srt_regroups_word_timestamps_into_shorter_cues(self):
        result = TranscriptionResult(
            audio_path=__file__,
            base_name="demo",
            provider="local",
            model="tiny",
            language="en",
            language_probability=0.99,
            duration=4.0,
            task="transcribe",
            device="cpu",
            compute_type="int8",
            text="Hello world again after pause",
            segments=[
                SegmentData(
                    index=0,
                    start=0.0,
                    end=4.0,
                    text=" Hello world again after pause",
                    words=[
                        WordTiming(start=0.0, end=0.4, word=" Hello"),
                        WordTiming(start=0.4, end=0.8, word=" world"),
                        WordTiming(start=0.8, end=1.2, word=" again"),
                        WordTiming(start=2.2, end=2.6, word=" after"),
                        WordTiming(start=2.6, end=3.0, word=" pause"),
                    ],
                ),
            ],
            output_files={},
        )
        settings = TranscriptionSettings(
            subtitle_max_chars=20,
            subtitle_max_duration=2.0,
            subtitle_split_on_pause=True,
            subtitle_pause_threshold=0.7,
        )

        srt = build_srt(result, settings)

        self.assertIn("1\n00:00:00,000 --> 00:00:01,200\nHello world again", srt)
        self.assertIn("2\n00:00:02,200 --> 00:00:03,000\nafter pause", srt)

    def test_build_vtt_prefixes_speaker_labels(self):
        result = TranscriptionResult(
            audio_path=__file__,
            base_name="demo",
            provider="local",
            model="tiny",
            language="en",
            language_probability=0.99,
            duration=2.0,
            task="transcribe",
            device="cpu",
            compute_type="int8",
            text="Hello there",
            segments=[
                SegmentData(index=0, start=0.0, end=1.0, text=" Hello", speaker="Speaker 1"),
                SegmentData(index=1, start=1.0, end=2.0, text=" there", speaker="Speaker 1"),
            ],
            output_files={},
        )

        vtt = build_vtt(result, TranscriptionSettings(subtitle_max_chars=32))

        self.assertIn("[Speaker 1] Hello there", vtt)


if __name__ == "__main__":
    unittest.main()
