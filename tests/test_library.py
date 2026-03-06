import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from massscriber.library import (
    REVIEW_STATE_FILE_NAME,
    build_preview,
    extract_transcript_ids,
    records_to_rows,
    search_transcripts,
    update_review_status,
)


class LibraryTests(TestCase):
    def test_index_and_search_transcripts_reads_json_and_txt_sources(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "demo.json").write_text(
                json.dumps(
                    {
                        "audio_path": "C:/audio/demo.mp3",
                        "language": "tr",
                        "model": "large-v3",
                        "text": "OpenAI ile test transcript",
                    }
                ),
                encoding="utf-8",
            )
            (root / "fallback.txt").write_text("plain text transcript", encoding="utf-8")

            records, summary = search_transcripts(root, query="openai", status_filter="all")

        self.assertEqual(len(records), 1)
        self.assertIn("1 transcript", summary)
        self.assertEqual(records[0].language, "tr")
        self.assertEqual(len(records_to_rows(records)), 1)

    def test_update_review_status_persists_and_filters(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "demo.json").write_text(
                json.dumps(
                    {
                        "audio_path": "C:/audio/demo.mp3",
                        "language": "en",
                        "model": "turbo",
                        "text": "review me",
                    }
                ),
                encoding="utf-8",
            )

            count = update_review_status(root, ["demo"], status="approved", note="looks good")
            self.assertEqual(count, 1)
            self.assertTrue((root / REVIEW_STATE_FILE_NAME).exists())

            approved_records, _ = search_transcripts(root, status_filter="approved")
            self.assertEqual(len(approved_records), 1)
            self.assertEqual(approved_records[0].status, "approved")
            self.assertIn("looks good", build_preview(approved_records))

    def test_extract_transcript_ids_splits_lines(self):
        self.assertEqual(extract_transcript_ids("a\n\nb\n"), ["a", "b"])
