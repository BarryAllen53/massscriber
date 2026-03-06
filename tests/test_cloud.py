from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from massscriber.cloud import (
    _parse_assemblyai_payload,
    _parse_deepgram_payload,
    _parse_elevenlabs_payload,
    _parse_openai_family_payload,
    _finalize_result,
)
from massscriber.types import TranscriptionSettings


class CloudParsingTests(TestCase):
    def test_parse_openai_verbose_json_payload(self):
        segments, text = _parse_openai_family_payload(
            {
                "text": "hello world",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.2,
                        "text": " hello world",
                        "words": [
                            {"start": 0.0, "end": 0.5, "word": " hello"},
                            {"start": 0.5, "end": 1.2, "word": " world"},
                        ],
                    }
                ],
            }
        )
        self.assertEqual(text, "hello world")
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0].words), 2)

    def test_parse_deepgram_payload_with_utterances(self):
        segments, text, language, duration = _parse_deepgram_payload(
            {
                "metadata": {"duration": 12.5},
                "results": {
                    "utterances": [
                        {
                            "start": 0.0,
                            "end": 1.1,
                            "speaker": 0,
                            "transcript": "hello there",
                            "words": [
                                {"start": 0.0, "end": 0.5, "word": "hello"},
                                {"start": 0.5, "end": 1.1, "word": " there"},
                            ],
                        }
                    ],
                    "channels": [
                        {
                            "alternatives": [
                                {
                                    "transcript": "hello there",
                                    "detected_language": "en",
                                }
                            ]
                        }
                    ],
                },
            }
        )
        self.assertEqual(text, "hello there")
        self.assertEqual(language, "en")
        self.assertEqual(duration, 12.5)
        self.assertEqual(segments[0].speaker, "speaker_0")

    def test_parse_assemblyai_payload_with_utterances(self):
        segments, text = _parse_assemblyai_payload(
            {
                "text": "merhaba dunya",
                "utterances": [
                    {
                        "start": 0,
                        "end": 1200,
                        "speaker": "A",
                        "text": "merhaba dunya",
                        "words": [
                            {"start": 0, "end": 500, "text": "merhaba"},
                            {"start": 500, "end": 1200, "text": " dunya"},
                        ],
                    }
                ],
            }
        )
        self.assertEqual(text, "merhaba dunya")
        self.assertEqual(segments[0].speaker, "speaker_A")

    def test_parse_elevenlabs_payload_with_word_timestamps(self):
        segments, text = _parse_elevenlabs_payload(
            {
                "text": "test line",
                "words": [
                    {"start": 0.0, "end": 0.4, "text": "test "},
                    {"start": 0.4, "end": 0.8, "text": "line"},
                ],
            }
        )
        self.assertEqual(text, "test line")
        self.assertEqual(len(segments), 1)

    def test_finalize_result_adds_provider_metadata(self):
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "demo.mp3"
            source.write_bytes(b"demo")
            result = _finalize_result(
                source=source,
                settings=TranscriptionSettings(output_formats=("json",)),
                output_dir=tmpdir,
                provider="openai",
                model="whisper-1",
                text="hello",
                segments=[],
                language="en",
                duration=1.0,
                request_id="req_123",
                metadata={"cost_hint": "remote"},
            )
        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.provider_request_id, "req_123")
        self.assertEqual(result.metadata["cost_hint"], "remote")
