"""Deterministic, sentence-disjoint Common Voice manifest generation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common_voice import (
    EXPECTED_LOCALE,
    SOURCE_DATASET,
    SOURCE_LICENCE,
    anonymous_speaker_id,
    dataset_version,
    read_tsv,
)
from .text import analyze_text

MANIFEST_FIELDS = (
    "audio_path",
    "raw_text",
    "normalized_text",
    "sentence_id",
    "speaker_id",
    "source_dataset",
    "dataset_version",
    "language_code",
    "licence",
    "duration_ms",
    "split",
    "text_flags",
)


def _score_group(normalized_text: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{normalized_text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _assign_split(normalized_text: str, seed: str, validation_fraction: float, test_fraction: float) -> str:
    score = _score_group(normalized_text, seed)
    if score < test_fraction:
        return "test"
    if score < test_fraction + validation_fraction:
        return "validation"
    return "train"


def _load_duration_map(path: Path) -> dict[str, int]:
    durations: dict[str, int] = {}
    for row in read_tsv(path):
        durations[row["clip"]] = int(row["duration[ms]"])
    return durations


def _write_tsv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_manifests(
    dataset_dir: Path,
    output_dir: Path,
    *,
    seed: str = "fub-common-voice-v1",
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    maximum_duration_ms: int = 15_000,
) -> dict[str, Any]:
    """Create deterministic training manifests from validated Common Voice rows."""

    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")

    dataset_dir = dataset_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_tsv(dataset_dir / "validated.tsv")
    durations = _load_duration_map(dataset_dir / "clip_durations.tsv")
    physical_audio = {path.name for path in (dataset_dir / "clips").glob("*.mp3")}
    version = dataset_version(dataset_dir)

    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for source_row in rows:
        path = source_row.get("path", "")
        sentence_id = source_row.get("sentence_id", "")
        analysis = analyze_text(source_row.get("sentence", ""))
        reasons: list[str] = []
        if source_row.get("locale") != EXPECTED_LOCALE:
            reasons.append("unexpected_locale")
        if path not in physical_audio:
            reasons.append("missing_audio")
        if path not in durations:
            reasons.append("missing_duration")
        elif durations[path] <= 0:
            reasons.append("nonpositive_duration")
        elif durations[path] > maximum_duration_ms:
            reasons.append(f"duration_over_{maximum_duration_ms}ms")
        if not analysis.normalized:
            reasons.append("empty_normalized_text")
        if not sentence_id:
            reasons.append("missing_sentence_id")
        if not source_row.get("client_id"):
            reasons.append("missing_speaker_id")

        if reasons:
            rejected.append(
                {
                    "audio_path": f"clips/{path}" if path else "",
                    "sentence_id": sentence_id,
                    "reasons": ",".join(reasons),
                }
            )
            continue

        split = _assign_split(
            analysis.normalized,
            seed,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
        accepted.append(
            {
                "audio_path": f"clips/{path}",
                "raw_text": source_row["sentence"],
                "normalized_text": analysis.normalized,
                "sentence_id": sentence_id,
                "speaker_id": anonymous_speaker_id(source_row["client_id"]),
                "source_dataset": SOURCE_DATASET,
                "dataset_version": version,
                "language_code": EXPECTED_LOCALE,
                "licence": SOURCE_LICENCE,
                "duration_ms": str(durations[path]),
                "split": split,
                "text_flags": ",".join(analysis.flags),
            }
        )

    accepted.sort(key=lambda row: (row["split"], row["normalized_text"], row["audio_path"]))
    rejected.sort(key=lambda row: (row["audio_path"], row["sentence_id"]))
    split_rows = {
        name: [row for row in accepted if row["split"] == name]
        for name in ("train", "validation", "test")
    }

    sentence_splits: dict[str, set[str]] = defaultdict(set)
    audio_splits: dict[str, set[str]] = defaultdict(set)
    for row in accepted:
        sentence_splits[row["normalized_text"]].add(row["split"])
        audio_splits[row["audio_path"]].add(row["split"])
    leaked_sentences = sorted(text for text, assigned in sentence_splits.items() if len(assigned) > 1)
    leaked_audio = sorted(path for path, assigned in audio_splits.items() if len(assigned) > 1)
    if leaked_sentences or leaked_audio:
        raise AssertionError("Manifest leakage check failed")

    for name, manifest_rows in split_rows.items():
        _write_tsv(output_dir / f"{name}.tsv", manifest_rows, MANIFEST_FIELDS)
    _write_tsv(
        output_dir / "rejected.tsv",
        rejected,
        ("audio_path", "sentence_id", "reasons"),
    )

    rejected_reason_counts: Counter[str] = Counter()
    for row in rejected:
        rejected_reason_counts.update(row["reasons"].split(","))

    summary: dict[str, Any] = {
        "source_dataset": SOURCE_DATASET,
        "dataset_version": version,
        "language_code": EXPECTED_LOCALE,
        "licence": SOURCE_LICENCE,
        "seed": seed,
        "split_method": "SHA-256 assignment grouped by NFC-normalized sentence text",
        "maximum_duration_ms": maximum_duration_ms,
        "fractions": {
            "train": 1 - validation_fraction - test_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "source_validated_rows": len(rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "rejected_reason_counts": dict(sorted(rejected_reason_counts.items())),
        "splits": {
            name: {
                "rows": len(manifest_rows),
                "hours": round(
                    sum(int(row["duration_ms"]) for row in manifest_rows) / 3_600_000, 6
                ),
                "unique_normalized_sentences": len(
                    {row["normalized_text"] for row in manifest_rows}
                ),
                "speakers": len({row["speaker_id"] for row in manifest_rows}),
            }
            for name, manifest_rows in split_rows.items()
        },
        "checks": {
            "normalized_sentence_leakage": len(leaked_sentences),
            "audio_path_leakage": len(leaked_audio),
            "all_rows_are_fub": all(row["language_code"] == EXPECTED_LOCALE for row in accepted),
            "raw_and_normalized_text_preserved": all(
                "raw_text" in row and "normalized_text" in row for row in accepted
            ),
        },
        "limitations": [
            "Audio-quality filtering is limited to metadata duration until the full PCM scan runs.",
            "Primary sentence-disjoint splits are not speaker-disjoint; create a separate speaker-disjoint benchmark where practical.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary

