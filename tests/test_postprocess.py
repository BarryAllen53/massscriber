import unittest

from massscriber.postprocess import (
    apply_glossary_to_segments,
    apply_glossary_to_text,
    build_glossary_summary,
    parse_glossary_rules,
)
from massscriber.types import SegmentData, TranscriptionSettings, WordTiming


class PostprocessTests(unittest.TestCase):
    def test_parse_glossary_rules_supports_comments_and_multiple_arrows(self):
        rules = parse_glossary_rules(
            """
            # comment
            Open AI => OpenAI
            GPT - > ignored
            Baris Manco -> Barış Manço
            """
        )

        self.assertEqual(len(rules), 2)
        self.assertEqual(rules[0].source, "Open AI")
        self.assertEqual(rules[1].target, "Barış Manço")

    def test_apply_glossary_to_text_respects_case_and_phrase_modes(self):
        settings = TranscriptionSettings(
            glossary_text="open ai => OpenAI",
            glossary_case_sensitive=False,
            glossary_whole_word=True,
        )
        self.assertEqual(apply_glossary_to_text("open ai labs", settings), "OpenAI labs")

        phrase_settings = TranscriptionSettings(
            glossary_text="trans => X",
            glossary_whole_word=False,
        )
        self.assertEqual(apply_glossary_to_text("transcript", phrase_settings), "Xcript")

    def test_apply_glossary_to_segments_updates_segment_and_word_tokens(self):
        settings = TranscriptionSettings(glossary_text="Open AI => OpenAI")
        segments = [
            SegmentData(
                index=0,
                start=0.0,
                end=1.0,
                text=" Open AI demos",
                words=[WordTiming(start=0.0, end=0.5, word=" Open"), WordTiming(start=0.5, end=1.0, word=" AI")],
            )
        ]

        apply_glossary_to_segments(segments, settings)

        self.assertEqual(segments[0].text.strip(), "OpenAI demos")
        self.assertEqual("".join(word.word for word in segments[0].words).strip(), "Open AI")

    def test_build_glossary_summary_reports_enabled_rule_count(self):
        settings = TranscriptionSettings(glossary_text="A => B\nC => D")
        self.assertIn("2 kural", build_glossary_summary(settings) or "")


if __name__ == "__main__":
    unittest.main()
