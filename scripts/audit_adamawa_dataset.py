#!/usr/bin/env python3
"""Audit and split the separate Adamawa-Fulfulde-TTS-Dataset corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from fub_tts.text import analyze_text, normalize_text, words


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a JSON object.")
    return value


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_mapping(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["audio_filename", "sentence"]:
            raise ValueError(
                "Expected Mapping_MP3.tsv columns ['audio_filename', 'sentence']; "
                f"got {reader.fieldnames}."
            )
        return [dict(row) for row in reader]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def text_review_flags(text: str) -> tuple[str, ...]:
    analysis = analyze_text(text)
    flags = list(analysis.flags)
    if re.search(r"['ʼ’]{2,}", analysis.normalized):
        flags.append("repeated_apostrophe")
    if "\\" in analysis.normalized:
        flags.append("literal_backslash")
    if any(character in analysis.normalized for character in "“”‘’«»"):
        flags.append("typographic_punctuation")
    return tuple(dict.fromkeys(flags))


def inspect_audio(path: Path) -> dict[str, Any]:
    try:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as error:  # libsndfile raises several backend-specific types
        return {"decode_status": "error", "decode_error": f"{type(error).__name__}: {error}"}
    if samples.size == 0:
        return {"decode_status": "error", "decode_error": "empty decoded audio"}
    mono = np.mean(samples, axis=1, dtype=np.float32)
    finite = bool(np.isfinite(mono).all())
    if not finite:
        return {"decode_status": "error", "decode_error": "non-finite decoded samples"}
    absolute = np.abs(mono)
    peak = float(absolute.max())
    rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
    clipping_ratio = float(np.mean(absolute >= 0.999))
    silence_ratio = float(np.mean(absolute <= 10 ** (-50 / 20)))
    duration_ms = round(len(mono) * 1000 / int(sample_rate))
    return {
        "decode_status": "ok",
        "sample_rate": int(sample_rate),
        "channels": int(samples.shape[1]),
        "frames": int(samples.shape[0]),
        "duration_ms": duration_ms,
        "peak": peak,
        "rms": rms,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "dc_offset": float(np.mean(mono, dtype=np.float64)),
    }


def split_for_text(
    text: str, seed: int, validation_fraction: float, test_fraction: float
) -> str:
    digest = hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).digest()
    score = int.from_bytes(digest[:8], "big") / 2**64
    if score < test_fraction:
        return "test"
    if score < test_fraction + validation_fraction:
        return "validation"
    return "train"


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    audio = report["audio"]
    manifests = report["manifests"]
    source = report["source"]
    return f"""# Adamawa-Fulfulde-TTS-Dataset audit

## Source and authorization

- Language code: `{source['language_code']}`
- Variety scope: {source['variety_scope']}
- Dataset version: `{source['dataset_version']}`
- Speaker coverage: {source['speaker_coverage']}
- Licence record: {source['licence']}
- Licence status: {source['licence_status']}
- Redistribution status: {source['redistribution_status']}

The licence was confirmed by the user for local research training, but the
exact licence identifier and evidence file are not yet present in the
workspace. This report does not infer redistribution or model-release rights.

## Mapping integrity

- Mapping rows: {counts['mapping_rows']:,}
- Physical MP3 files: {counts['physical_mp3_files']:,}
- Missing mapped files: {counts['missing_mapped_audio']}
- Unreferenced audio files: {counts['unreferenced_audio']}
- Unique normalized transcripts: {counts['unique_normalized_texts']:,}
- Duplicate mapped filenames: {counts['duplicate_mapped_filenames']}
- Duplicate normalized transcripts: {counts['duplicate_normalized_texts']}
- Duplicate audio hashes: {counts['duplicate_audio_hashes']}

## Full audio scan

- Successfully decoded mapped clips: {audio['decoded_mapped_clips']:,}
- Decode failures: {audio['decode_failures']}
- Total mapped duration: {audio['total_mapped_hours']:.3f} hours
- Usable duration after 1-15 second duration gate: {audio['duration_gate_usable_hours']:.3f} hours
- Source sample rates: {audio['sample_rate_distribution']}
- Source channel counts: {audio['channel_distribution']}
- Median duration: {audio['duration_distribution_ms']['median'] / 1000:.3f} seconds
- 95th-percentile duration: {audio['duration_distribution_ms']['p95'] / 1000:.3f} seconds
- Clips over 15 seconds: {audio['clips_over_15000_ms']}
- Clips below 1 second: {audio['clips_below_1000_ms']}
- Clips with at least 0.1% near-full-scale samples: {audio['clips_clipping_ratio_ge_0_001']}
- Clips with at least 60% near-silence samples: {audio['clips_silence_ratio_ge_0_60']}

## Training-ready metadata manifests

| Split | Rows | Hours | Unique text |
|---|---:|---:|---:|
| Train | {manifests['splits']['train']['rows']:,} | {manifests['splits']['train']['hours']:.3f} | {manifests['splits']['train']['unique_normalized_texts']:,} |
| Validation | {manifests['splits']['validation']['rows']:,} | {manifests['splits']['validation']['hours']:.3f} | {manifests['splits']['validation']['unique_normalized_texts']:,} |
| Test | {manifests['splits']['test']['rows']:,} | {manifests['splits']['test']['hours']:.3f} | {manifests['splits']['test']['unique_normalized_texts']:,} |

- Accepted rows: {manifests['accepted_rows']:,}
- Rejected rows: {manifests['rejected_rows']:,}
- Normalized-text leakage: {manifests['checks']['normalized_text_leakage']}
- Audio-path leakage: {manifests['checks']['audio_path_leakage']}

## Important limitation

`Mapping_MP3.tsv` contains no speaker field. These files must not be described
as a verified single-speaker corpus, and a speaker-disjoint evaluation cannot
be created from the supplied metadata. Local training may use the neutral
source-group identifier, but speaker identity or consistency claims require
provenance evidence or acoustic clustering followed by authorized review.
"""


def run(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = config_path.resolve().parent.parent
    dataset_dir = root / config["dataset_dir"]
    mapping_path = dataset_dir / config["mapping_file"]
    audio_dir = dataset_dir / config["audio_dir"]
    report_dir = root / config["report_dir"]
    manifest_dir = root / config["manifest_dir"]
    source = config["source"]
    filters = config["filters"]
    split_config = config["splits"]
    mapping_hash_before = sha256_file(mapping_path)
    mapping = read_mapping(mapping_path)
    mapped_names = [row["audio_filename"] for row in mapping]
    physical_paths = sorted(audio_dir.glob("*.mp3"))
    physical_by_name = {path.name: path for path in physical_paths}

    records: list[dict[str, Any]] = []
    audio_hashes: dict[str, str] = {}
    character_counts: Counter[str] = Counter()
    text_flag_counts: Counter[str] = Counter()
    for row in mapping:
        filename = row["audio_filename"]
        raw_text = row["sentence"]
        normalized = normalize_text(raw_text)
        flags = text_review_flags(raw_text)
        character_counts.update(character for character in normalized if not character.isspace())
        text_flag_counts.update(flags)
        path = physical_by_name.get(filename)
        if path is None:
            audio = {"decode_status": "not_found", "decode_error": "missing mapped audio"}
            digest = None
        else:
            audio = inspect_audio(path)
            digest = sha256_file(path)
            audio_hashes[filename] = digest
        rejection_reasons: list[str] = []
        if path is None:
            rejection_reasons.append("missing_audio")
        elif audio["decode_status"] != "ok":
            rejection_reasons.append("decode_error")
        else:
            duration_ms = int(audio["duration_ms"])
            if duration_ms < int(filters["minimum_duration_ms"]):
                rejection_reasons.append(
                    f"duration_below_{filters['minimum_duration_ms']}ms"
                )
            if duration_ms > int(filters["maximum_duration_ms"]):
                rejection_reasons.append(
                    f"duration_over_{filters['maximum_duration_ms']}ms"
                )
        if not normalized:
            rejection_reasons.append("empty_normalized_text")
        records.append(
            {
                "audio_filename": filename,
                "audio_path": f"{config['audio_dir']}/{filename}",
                "raw_text": raw_text,
                "normalized_text": normalized,
                "text_flags": list(flags),
                "audio_sha256": digest,
                "audio": audio,
                "rejection_reasons": rejection_reasons,
            }
        )

    orphan_names = sorted(set(physical_by_name) - set(mapped_names))
    for filename in orphan_names:
        if filename not in audio_hashes:
            audio_hashes[filename] = sha256_file(physical_by_name[filename])

    good_audio = [
        record for record in records if record["audio"]["decode_status"] == "ok"
    ]
    accepted = [record for record in records if not record["rejection_reasons"]]
    durations_ms = [float(record["audio"]["duration_ms"]) for record in good_audio]
    hash_groups: dict[str, list[str]] = defaultdict(list)
    for filename, digest in audio_hashes.items():
        hash_groups[digest].append(filename)
    duplicate_hash_groups = [names for names in hash_groups.values() if len(names) > 1]

    manifest_fields = [
        "audio_path",
        "raw_text",
        "normalized_text",
        "sentence_id",
        "speaker_id",
        "speaker_metadata_status",
        "source_dataset",
        "dataset_version",
        "language_code",
        "licence",
        "duration_ms",
        "split",
        "text_flags",
    ]
    manifest_rows: list[dict[str, Any]] = []
    for record in accepted:
        split = split_for_text(
            record["normalized_text"],
            int(config["seed"]),
            float(split_config["validation_fraction"]),
            float(split_config["test_fraction"]),
        )
        manifest_rows.append(
            {
                "audio_path": record["audio_path"],
                "raw_text": record["raw_text"],
                "normalized_text": record["normalized_text"],
                "sentence_id": hashlib.sha256(
                    record["normalized_text"].encode("utf-8")
                ).hexdigest(),
                "speaker_id": "unknown",
                "speaker_metadata_status": "not supplied",
                "source_dataset": source["dataset"],
                "dataset_version": source["dataset_version"],
                "language_code": source["language_code"],
                "licence": source["licence"],
                "duration_ms": str(record["audio"]["duration_ms"]),
                "split": split,
                "text_flags": ",".join(record["text_flags"]),
            }
        )

    # A character-level model must not encounter a corpus character only in a
    # held-out split. Deterministically move whole normalized-text groups into
    # training until its inventory covers the complete accepted corpus.
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        grouped_rows[row["normalized_text"]].append(row)
    full_characters = set(
        "".join(row["normalized_text"] for row in manifest_rows)
    )
    train_characters = set(
        "".join(
            row["normalized_text"]
            for row in manifest_rows
            if row["split"] == "train"
        )
    )
    missing_train_characters = full_characters - train_characters
    character_coverage_moves: list[dict[str, Any]] = []
    while missing_train_characters:
        candidates = [
            (text, rows, set(text) & missing_train_characters)
            for text, rows in grouped_rows.items()
            if rows[0]["split"] != "train" and set(text) & missing_train_characters
        ]
        if not candidates:
            raise RuntimeError(
                "Could not place all accepted characters in training: "
                f"{sorted(missing_train_characters, key=ord)}"
            )
        text, rows, gained = max(
            candidates,
            key=lambda item: (len(item[2]), -len(item[0]), item[0]),
        )
        old_split = rows[0]["split"]
        for row in rows:
            row["split"] = "train"
        train_characters.update(set(text))
        missing_train_characters = full_characters - train_characters
        character_coverage_moves.append(
            {
                "normalized_text_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "from_split": old_split,
                "rows_moved": len(rows),
                "characters_added": sorted(gained, key=ord),
            }
        )
    manifest_rows.sort(
        key=lambda row: (row["split"], row["normalized_text"], row["audio_path"])
    )
    split_rows = {
        split: [row for row in manifest_rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    for split, rows in split_rows.items():
        write_tsv(manifest_dir / f"{split}.tsv", manifest_fields, rows)
    rejected_rows = [
        {
            "audio_path": record["audio_path"],
            "reasons": ",".join(record["rejection_reasons"]),
        }
        for record in records
        if record["rejection_reasons"]
    ]
    rejected_rows.extend(
        {
            "audio_path": f"{config['audio_dir']}/{filename}",
            "reasons": "unreferenced_audio",
        }
        for filename in orphan_names
    )
    write_tsv(manifest_dir / "rejected.tsv", ["audio_path", "reasons"], rejected_rows)

    text_splits: dict[str, set[str]] = defaultdict(set)
    path_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        text_splits[row["normalized_text"]].add(row["split"])
        path_splits[row["audio_path"]].add(row["split"])
    manifests = {
        "accepted_rows": len(manifest_rows),
        "rejected_rows": len(rejected_rows),
        "rejection_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in rejected_rows
                    for reason in row["reasons"].split(",")
                ).items()
            )
        ),
        "splits": {
            split: {
                "rows": len(rows),
                "hours": round(
                    sum(int(row["duration_ms"]) for row in rows) / 3_600_000, 6
                ),
                "unique_normalized_texts": len(
                    {row["normalized_text"] for row in rows}
                ),
            }
            for split, rows in split_rows.items()
        },
        "checks": {
            "normalized_text_leakage": sum(
                len(splits) > 1 for splits in text_splits.values()
            ),
            "audio_path_leakage": sum(len(splits) > 1 for splits in path_splits.values()),
            "speaker_disjoint_check": "not possible; speaker metadata absent",
            "full_character_inventory_covered_by_train": (
                set(
                    "".join(
                        row["normalized_text"]
                        for row in manifest_rows
                        if row["split"] == "train"
                    )
                )
                == full_characters
            ),
            "character_coverage_moves": character_coverage_moves,
        },
    }
    report = {
        "report_type": "full mapped text and decoded-audio audit",
        "source": source,
        "counts": {
            "mapping_rows": len(mapping),
            "physical_mp3_files": len(physical_paths),
            "missing_mapped_audio": len(set(mapped_names) - set(physical_by_name)),
            "unreferenced_audio": len(orphan_names),
            "unreferenced_audio_paths": orphan_names,
            "unique_normalized_texts": len(
                {normalize_text(row["sentence"]) for row in mapping}
            ),
            "duplicate_mapped_filenames": sum(
                count > 1 for count in Counter(mapped_names).values()
            ),
            "duplicate_normalized_texts": sum(
                count > 1
                for count in Counter(
                    normalize_text(row["sentence"]) for row in mapping
                ).values()
            ),
            "duplicate_audio_hashes": len(duplicate_hash_groups),
            "duplicate_audio_hash_groups": duplicate_hash_groups,
        },
        "audio": {
            "decoded_mapped_clips": len(good_audio),
            "decode_failures": len(records) - len(good_audio),
            "total_mapped_hours": round(sum(durations_ms) / 3_600_000, 6),
            "duration_gate_usable_hours": round(
                sum(record["audio"]["duration_ms"] for record in accepted)
                / 3_600_000,
                6,
            ),
            "sample_rate_distribution": dict(
                sorted(Counter(str(row["audio"]["sample_rate"]) for row in good_audio).items())
            ),
            "channel_distribution": dict(
                sorted(Counter(str(row["audio"]["channels"]) for row in good_audio).items())
            ),
            "duration_distribution_ms": {
                "minimum": min(durations_ms) if durations_ms else None,
                "median": statistics.median(durations_ms) if durations_ms else None,
                "p95": percentile(durations_ms, 0.95),
                "maximum": max(durations_ms) if durations_ms else None,
                "mean": statistics.fmean(durations_ms) if durations_ms else None,
            },
            "clips_over_15000_ms": sum(value > 15_000 for value in durations_ms),
            "clips_below_1000_ms": sum(value < 1_000 for value in durations_ms),
            "clips_clipping_ratio_ge_0_001": sum(
                row["audio"]["clipping_ratio"] >= 0.001 for row in good_audio
            ),
            "clips_silence_ratio_ge_0_60": sum(
                row["audio"]["silence_ratio"] >= 0.60 for row in good_audio
            ),
            "clipping_ratio_distribution": {
                "median": statistics.median(
                    row["audio"]["clipping_ratio"] for row in good_audio
                ),
                "p95": percentile(
                    [row["audio"]["clipping_ratio"] for row in good_audio], 0.95
                ),
                "maximum": max(row["audio"]["clipping_ratio"] for row in good_audio),
            },
            "silence_ratio_distribution": {
                "median": statistics.median(
                    row["audio"]["silence_ratio"] for row in good_audio
                ),
                "p95": percentile(
                    [row["audio"]["silence_ratio"] for row in good_audio], 0.95
                ),
                "maximum": max(row["audio"]["silence_ratio"] for row in good_audio),
            },
        },
        "text": {
            "normalization": "Unicode NFC plus whitespace collapse; no transliteration",
            "character_inventory": [
                {"character": character, "count": count}
                for character, count in sorted(character_counts.items())
            ],
            "review_flag_counts": dict(sorted(text_flag_counts.items())),
            "word_types": len(
                {
                    word.casefold()
                    for row in mapping
                    for word in words(normalize_text(row["sentence"]))
                }
            ),
        },
        "speaker_analysis": {
            "status": "not possible from supplied metadata",
            "speaker_field_present": False,
            "safe_source_group_label": "unknown",
            "limitations": [
                "Do not claim this is a verified single-speaker corpus.",
                "Do not attempt speaker identification without explicit scope and authorization.",
            ],
        },
        "manifests": manifests,
        "lineage": {
            "mapping_sha256_before": mapping_hash_before,
            "mapping_sha256_after": sha256_file(mapping_path),
            "source_modified": mapping_hash_before != sha256_file(mapping_path),
        },
        "intended_use": "local Adamawa Fulfulde TTS research baseline",
        "limitations": [
            "Exact licence identifier and evidence file are pending repository registration.",
            "Speaker coverage is absent from the supplied mapping.",
            "Noise variability and native-speaker transcript correctness were not assessed automatically.",
            "Duration-only rejection does not establish studio-quality TTS suitability.",
        ],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "audit.md").write_text(render_markdown(report), encoding="utf-8")
    with (report_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (manifest_dir / "summary.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/adamawa_dataset.json")
    )
    args = parser.parse_args()
    report = run(args.config)
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "audio": report["audio"],
                "text": {"review_flag_counts": report["text"]["review_flag_counts"]},
                "speaker_analysis": report["speaker_analysis"],
                "manifests": report["manifests"],
                "lineage": report["lineage"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
