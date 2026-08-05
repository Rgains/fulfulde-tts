import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from fub_tts.common_voice import audit_common_voice, inspect_audio, is_usable
from fub_tts.split import build_manifests


CLIP_FIELDS = [
    "client_id",
    "path",
    "sentence_id",
    "sentence",
    "sentence_domain",
    "up_votes",
    "down_votes",
    "age",
    "gender",
    "accents",
    "variant",
    "locale",
    "segment",
]


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def clip_row(index: int, sentence: str, sentence_id: str, speaker: str = "speaker-a") -> dict[str, str]:
    return {
        "client_id": speaker,
        "path": f"clip-{index}.mp3",
        "sentence_id": sentence_id,
        "sentence": sentence,
        "sentence_domain": "",
        "up_votes": "2",
        "down_votes": "0",
        "age": "",
        "gender": "",
        "accents": "",
        "variant": "",
        "locale": "fub",
        "segment": "",
    }


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "cv-corpus-test" / "fub"
        (self.root / "clips").mkdir(parents=True)
        (self.root / "README.md").write_text("# fub\n", encoding="utf-8")

        validated: list[dict[str, str]] = []
        for sentence_number in range(30):
            sentence = f"Ɓiŋgel jooɗi ɗoo {sentence_number}."
            sentence_id = f"sentence-{sentence_number}"
            for repeat in range(2):
                index = sentence_number * 2 + repeat
                row = clip_row(index, sentence, sentence_id, speaker=f"speaker-{repeat}")
                validated.append(row)
                (self.root / "clips" / row["path"]).write_bytes(b"not-decoded-in-metadata-test")

        too_long = clip_row(999, "Ƴiɗde ko wooɗi.", "sentence-long")
        validated.append(too_long)
        (self.root / "clips" / too_long["path"]).write_bytes(b"long")

        write_tsv(self.root / "validated.tsv", CLIP_FIELDS, validated)
        write_tsv(self.root / "invalidated.tsv", CLIP_FIELDS, [])
        write_tsv(self.root / "other.tsv", CLIP_FIELDS, [])
        write_tsv(self.root / "train.tsv", CLIP_FIELDS, validated[:10])
        write_tsv(self.root / "dev.tsv", CLIP_FIELDS, validated[10:20])
        write_tsv(self.root / "test.tsv", CLIP_FIELDS, validated[20:30])
        duration_rows = [
            {"clip": row["path"], "duration[ms]": "15120" if row is too_long else "5000"}
            for row in validated
        ]
        write_tsv(self.root / "clip_durations.tsv", ["clip", "duration[ms]"], duration_rows)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_audit_reports_presence_without_claiming_decode(self) -> None:
        report = audit_common_voice(self.root)
        self.assertEqual(report["counts"]["validated_rows"], 61)
        self.assertEqual(report["counts"]["validated_speakers"], 3)
        self.assertEqual(report["integrity"]["validated_missing_audio"], [])
        self.assertEqual(report["audio_quality"]["status"], "not_run")

    def test_manifests_are_deterministic_and_sentence_disjoint(self) -> None:
        first = Path(self.temporary.name) / "first"
        second = Path(self.temporary.name) / "second"
        first_summary = build_manifests(self.root, first)
        second_summary = build_manifests(self.root, second)

        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first_summary["accepted_rows"], 60)
        self.assertEqual(first_summary["rejected_reason_counts"], {"duration_over_15000ms": 1})
        self.assertEqual(first_summary["checks"]["normalized_sentence_leakage"], 0)
        self.assertEqual((first / "train.tsv").read_bytes(), (second / "train.tsv").read_bytes())

        assignments: dict[str, set[str]] = {}
        for split in ("train", "validation", "test"):
            with (first / f"{split}.tsv").open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    assignments.setdefault(row["normalized_text"], set()).add(split)
        self.assertTrue(assignments)
        self.assertTrue(all(len(splits) == 1 for splits in assignments.values()))

        rejected = (first / "rejected.tsv").read_text(encoding="utf-8")
        self.assertIn("clip-999.mp3", rejected)
        json.loads((first / "summary.json").read_text(encoding="utf-8"))


class AudioScanTests(unittest.TestCase):
    """inspect_audio/is_usable and the opt-in full decoded-audio scan."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.clips_dir = Path(self.temporary.name) / "clips"
        self.clips_dir.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_tone(self, name: str, seconds: float = 2.0, sample_rate: int = 24000) -> Path:
        path = self.clips_dir / name
        t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
        tone = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        sf.write(path, tone, sample_rate, format="WAV")
        return path

    def _write_silence(self, name: str, seconds: float = 2.0, sample_rate: int = 24000) -> Path:
        path = self.clips_dir / name
        sf.write(path, np.zeros(int(seconds * sample_rate), dtype=np.float32), sample_rate, format="WAV")
        return path

    def test_inspect_audio_measures_a_real_clip(self) -> None:
        path = self._write_tone("tone.mp3")
        audio = inspect_audio(path)
        self.assertEqual(audio["decode_status"], "ok")
        self.assertEqual(audio["sample_rate"], 24000)
        self.assertAlmostEqual(audio["duration_ms"], 2000, delta=5)
        self.assertLess(audio["silence_ratio"], 0.1)
        self.assertTrue(is_usable(audio))

    def test_inspect_audio_flags_decode_errors(self) -> None:
        path = self.clips_dir / "garbage.mp3"
        path.write_bytes(b"not actually audio")
        audio = inspect_audio(path)
        self.assertEqual(audio["decode_status"], "error")
        self.assertFalse(is_usable(audio))

    def test_is_usable_rejects_near_silent_clips(self) -> None:
        path = self._write_silence("silent.mp3")
        audio = inspect_audio(path)
        self.assertEqual(audio["decode_status"], "ok")
        self.assertGreaterEqual(audio["silence_ratio"], 0.60)
        self.assertFalse(is_usable(audio))

    def test_full_scan_fills_in_audio_quality_and_usable_duration(self) -> None:
        root = Path(self.temporary.name) / "cv-corpus-test" / "fub"
        (root / "clips").mkdir(parents=True)
        (root / "README.md").write_text("# fub\n", encoding="utf-8")
        self.clips_dir = root / "clips"

        good = self._write_tone("clip-0.mp3")
        quiet = self._write_silence("clip-1.mp3")
        rows = [
            clip_row(0, "Ɓiŋgel jooɗi ɗoo.", "sentence-0", speaker="speaker-a"),
            clip_row(1, "Ɓiŋgel jooɗi ɗoo.", "sentence-0", speaker="speaker-a"),
        ]
        write_tsv(root / "validated.tsv", CLIP_FIELDS, rows)
        write_tsv(root / "invalidated.tsv", CLIP_FIELDS, [])
        write_tsv(root / "other.tsv", CLIP_FIELDS, [])
        write_tsv(root / "train.tsv", CLIP_FIELDS, rows)
        write_tsv(root / "dev.tsv", CLIP_FIELDS, [])
        write_tsv(root / "test.tsv", CLIP_FIELDS, [])
        write_tsv(
            root / "clip_durations.tsv",
            ["clip", "duration[ms]"],
            [{"clip": "clip-0.mp3", "duration[ms]": "2000"}, {"clip": "clip-1.mp3", "duration[ms]": "2000"}],
        )
        self.assertTrue(good.exists() and quiet.exists())

        report = audit_common_voice(root, run_audio_scan=True)
        self.assertEqual(report["audio_quality"]["status"], "ok")
        self.assertEqual(report["audio_quality"]["decoded_clips"], 2)
        self.assertEqual(report["audio_quality"]["usable_clips"], 1)
        ranking = {row["speaker_id"]: row for row in report["speaker_analysis"]["ranking"]}
        (only_speaker,) = ranking.values()
        self.assertEqual(only_speaker["usable_clips"], 1)
        self.assertAlmostEqual(only_speaker["usable_duration_hours"], 2000 / 3_600_000, places=6)


if __name__ == "__main__":
    unittest.main()
