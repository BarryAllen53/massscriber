from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from massscriber.cloud import (
    _parse_assemblyai_payload,
    _parse_deepgram_payload,
    _parse_elevenlabs_payload,
    _parse_openai_family_payload,
    _finalize_result,
    RemoteTranscriptionEngine,
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

    def test_deepgram_remote_url_mode_posts_json_url(self):
        engine = RemoteTranscriptionEngine()
        settings = TranscriptionSettings(
            provider="deepgram",
            provider_remote_url="https://cdn.example.com/audio.mp3",
        )
        source = Path("audio.mp3")
        captured: dict[str, object] = {}

        class FakeResponse:
            headers = {"dg-request-id": "dg_req_1"}

            def json(self):
                return {"results": {"channels": [{"alternatives": [{"transcript": "hello"}]}]}}

        def fake_request(**kwargs):
            captured.update(kwargs)
            return FakeResponse()

        with patch.object(engine, "_request", side_effect=fake_request):
            payload, request_id = engine._call_deepgram(
                source=source,
                settings=settings,
                api_key="dg-secret",
                model="nova-3",
                remote_audio_url=settings.provider_remote_url or "",
            )

        self.assertEqual(request_id, "dg_req_1")
        self.assertEqual(captured["json"], {"url": "https://cdn.example.com/audio.mp3"})
        self.assertNotIn("content", captured)
        self.assertEqual(payload["results"]["channels"][0]["alternatives"][0]["transcript"], "hello")

    def test_assemblyai_remote_url_skips_upload_step(self):
        engine = RemoteTranscriptionEngine()
        settings = TranscriptionSettings(
            provider="assemblyai",
            provider_remote_url="https://cdn.example.com/audio.mp3",
            provider_poll_interval=0.01,
        )
        source = Path("audio.mp3")
        requested_urls: list[str] = []

        class FakeResponse:
            def __init__(self, payload: dict[str, object]):
                self._payload = payload
                self.headers: dict[str, str] = {}

            def json(self):
                return self._payload

        responses = [
            FakeResponse({"id": "tr_123"}),
            FakeResponse({"status": "completed", "id": "tr_123", "text": "hello world"}),
        ]

        def fake_request(**kwargs):
            requested_urls.append(str(kwargs["url"]))
            return responses.pop(0)

        with patch.object(engine, "_request", side_effect=fake_request):
            payload, request_id = engine._call_assemblyai(
                source=source,
                settings=settings,
                api_key="aa-secret",
                model="universal",
                remote_audio_url=settings.provider_remote_url or "",
            )

        self.assertEqual(request_id, "tr_123")
        self.assertEqual(payload["text"], "hello world")
        self.assertEqual(len(requested_urls), 2)
        self.assertTrue(requested_urls[0].endswith("/transcript"))
        self.assertNotIn("/upload", "".join(requested_urls))
