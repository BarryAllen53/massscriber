import unittest

from massscriber.diarization import SpeakerTurn, assign_speakers_to_segments, normalize_speaker_labels
from massscriber.types import SegmentData


class DiarizationTests(unittest.TestCase):
    def test_normalize_speaker_labels_makes_labels_human_readable(self):
        turns = [
            SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_00"),
            SpeakerTurn(start=1.0, end=2.0, speaker="SPEAKER_01"),
            SpeakerTurn(start=2.0, end=3.0, speaker="SPEAKER_00"),
        ]

        normalized = normalize_speaker_labels(turns)

        self.assertEqual(normalized[0].speaker, "Speaker 1")
        self.assertEqual(normalized[1].speaker, "Speaker 2")
        self.assertEqual(normalized[2].speaker, "Speaker 1")

    def test_assign_speakers_to_segments_uses_best_overlap(self):
        segments = [
            SegmentData(index=0, start=0.0, end=1.4, text=" hello"),
            SegmentData(index=1, start=1.4, end=3.0, text=" world"),
        ]
        turns = [
            SpeakerTurn(start=0.0, end=1.0, speaker="Speaker 1"),
            SpeakerTurn(start=1.0, end=3.0, speaker="Speaker 2"),
        ]

        assign_speakers_to_segments(segments, turns)

        self.assertEqual(segments[0].speaker, "Speaker 1")
        self.assertEqual(segments[1].speaker, "Speaker 2")


if __name__ == "__main__":
    unittest.main()
