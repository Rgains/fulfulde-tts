import csv
import json
import tempfile
import unittest
from pathlib import Path

from fub_tts.common_voice import audit_common_voice
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


if __name__ == "__main__":
    unittest.main()
