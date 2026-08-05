"""Mozilla Common Voice metadata inspection for Adamawa Fulfulde."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

from .text import analyze_text, normalize_text, words

EXPECTED_LOCALE = "fub"
SOURCE_DATASET = "Mozilla Common Voice Scripted Speech"
SOURCE_LICENCE = "CC0-1.0"
DIALECT_SCOPE = "Cameroon Adamawa Fulfulde; Ngaoundéré prompt variety"

REQUIRED_CLIP_COLUMNS = {
    "client_id",
    "path",
    "sentence_id",
    "sentence",
    "locale",
}

# Same thresholds used by scripts/audit_adamawa_dataset.py, so "usable
# duration" means the same thing across both source datasets.
USABLE_MINIMUM_DURATION_MS = 1_000
USABLE_MAXIMUM_DURATION_MS = 15_000
USABLE_MAXIMUM_CLIPPING_RATIO = 0.001
USABLE_MAXIMUM_SILENCE_RATIO = 0.60


def inspect_audio(path: Path) -> dict[str, Any]:
    """Decode one clip and measure it. Never modifies the source file."""

    try:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as error:  # libsndfile raises several backend-specific types
        return {"decode_status": "error", "decode_error": f"{type(error).__name__}: {error}"}
    if samples.size == 0:
        return {"decode_status": "error", "decode_error": "empty decoded audio"}
    mono = np.mean(samples, axis=1, dtype=np.float32)
    if not np.isfinite(mono).all():
        return {"decode_status": "error", "decode_error": "non-finite decoded samples"}
    absolute = np.abs(mono)
    return {
        "decode_status": "ok",
        "sample_rate": int(sample_rate),
        "channels": int(samples.shape[1]),
        "duration_ms": round(len(mono) * 1000 / int(sample_rate)),
        "peak": float(absolute.max()),
        "rms": float(np.sqrt(np.mean(np.square(mono), dtype=np.float64))),
        "clipping_ratio": float(np.mean(absolute >= 0.999)),
        "silence_ratio": float(np.mean(absolute <= 10 ** (-50 / 20))),
        "dc_offset": float(np.mean(mono, dtype=np.float64)),
    }


def is_usable(audio: dict[str, Any]) -> bool:
    """Apply the same duration/clipping/silence gate used for the separate dataset."""

    if audio.get("decode_status") != "ok":
        return False
    duration_ms = audio["duration_ms"]
    return (
        USABLE_MINIMUM_DURATION_MS <= duration_ms <= USABLE_MAXIMUM_DURATION_MS
        and audio["clipping_ratio"] < USABLE_MAXIMUM_CLIPPING_RATIO
        and audio["silence_ratio"] < USABLE_MAXIMUM_SILENCE_RATIO
    )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 TSV file and preserve all source fields."""

    if path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return []
        return [dict(row) for row in reader]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def anonymous_speaker_id(client_id: str) -> str:
    """Create a stable project-local pseudonym without exposing the source ID."""

    digest = hashlib.sha256(f"common-voice-fub\0{client_id}".encode()).hexdigest()
    return f"cv_{digest[:16]}"


def dataset_version(dataset_dir: Path) -> str:
    parent_name = dataset_dir.resolve().parent.name
    if parent_name.startswith("cv-corpus-"):
        return parent_name.removeprefix("cv-corpus-")
    return parent_name


def _duration_distribution(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "maximum_ms": None,
            "mean_ms": None,
            "over_15000_ms": 0,
        }
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "count": len(values),
        "minimum_ms": min(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[p95_index],
        "maximum_ms": max(values),
        "mean_ms": round(statistics.fmean(values), 3),
        "over_15000_ms": sum(value > 15_000 for value in values),
    }


def _load_durations(path: Path) -> tuple[dict[str, int], list[str]]:
    rows = read_tsv(path)
    durations: dict[str, int] = {}
    duplicates: list[str] = []
    for row in rows:
        clip = row.get("clip", "")
        if clip in durations:
            duplicates.append(clip)
        try:
            durations[clip] = int(row.get("duration[ms]", ""))
        except ValueError as error:
            raise ValueError(f"Invalid duration for {clip!r}") from error
    return durations, duplicates


def _validate_schema(rows: Iterable[dict[str, str]], source: Path) -> None:
    first = next(iter(rows), None)
    if first is None:
        return
    missing = REQUIRED_CLIP_COLUMNS - first.keys()
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def _audio_quality_section(
    audio_scan: dict[str, dict[str, Any]], run_audio_scan: bool
) -> dict[str, Any]:
    if not run_audio_scan:
        return {
            "status": "not_run",
            "decoded_clips": None,
            "corrupt_clips": None,
            "sample_rate_distribution": None,
            "clipping_statistics": None,
            "silence_statistics": None,
            "noise_variability": None,
            "required_next_step": "Run a full FFmpeg/PCM audio scan before training.",
        }
    good = [audio for audio in audio_scan.values() if audio.get("decode_status") == "ok"]
    usable = [audio for audio in audio_scan.values() if is_usable(audio)]
    durations_ms = [float(audio["duration_ms"]) for audio in good]
    clipping_values = [audio["clipping_ratio"] for audio in good]
    silence_values = [audio["silence_ratio"] for audio in good]
    return {
        "status": "ok",
        "decoded_clips": len(good),
        "decode_failures": len(audio_scan) - len(good),
        "usable_clips": len(usable),
        "usable_hours": round(sum(audio["duration_ms"] for audio in usable) / 3_600_000, 6),
        "usable_definition": (
            f"decode ok, {USABLE_MINIMUM_DURATION_MS}-{USABLE_MAXIMUM_DURATION_MS} ms, "
            f"clipping_ratio < {USABLE_MAXIMUM_CLIPPING_RATIO}, "
            f"silence_ratio < {USABLE_MAXIMUM_SILENCE_RATIO}"
        ),
        "sample_rate_distribution": dict(
            sorted(Counter(str(audio["sample_rate"]) for audio in good).items())
        ),
        "channel_distribution": dict(
            sorted(Counter(str(audio["channels"]) for audio in good).items())
        ),
        "duration_distribution_ms": {
            "minimum": min(durations_ms) if durations_ms else None,
            "median": statistics.median(durations_ms) if durations_ms else None,
            "p95": _percentile(durations_ms, 0.95),
            "maximum": max(durations_ms) if durations_ms else None,
        },
        "clipping_ratio_distribution": {
            "median": statistics.median(clipping_values) if clipping_values else None,
            "p95": _percentile(clipping_values, 0.95),
            "maximum": max(clipping_values) if clipping_values else None,
        },
        "silence_ratio_distribution": {
            "median": statistics.median(silence_values) if silence_values else None,
            "p95": _percentile(silence_values, 0.95),
            "maximum": max(silence_values) if silence_values else None,
        },
        "clips_clipping_ratio_at_or_above_threshold": sum(
            value >= USABLE_MAXIMUM_CLIPPING_RATIO for value in clipping_values
        ),
        "clips_silence_ratio_at_or_above_threshold": sum(
            value >= USABLE_MAXIMUM_SILENCE_RATIO for value in silence_values
        ),
    }


def audit_common_voice(dataset_dir: Path, *, run_audio_scan: bool = False) -> dict[str, Any]:
    """Build a metadata-level integrity and suitability audit.

    With ``run_audio_scan=False`` (the default), this function deliberately
    does not claim to decode or assess MP3 quality -- it is fast and safe to
    call repeatedly, including in tests with placeholder clip files. Pass
    ``run_audio_scan=True`` to additionally decode every validated clip with
    a physically present file and fill in real audio-quality measurements
    and per-speaker usable duration.
    """

    dataset_dir = dataset_dir.resolve()
    required_files = [
        "README.md",
        "validated.tsv",
        "invalidated.tsv",
        "other.tsv",
        "train.tsv",
        "dev.tsv",
        "test.tsv",
        "clip_durations.tsv",
    ]
    missing_metadata = [name for name in required_files if not (dataset_dir / name).is_file()]
    if missing_metadata:
        raise FileNotFoundError(f"Missing Common Voice files: {missing_metadata}")
    clips_dir = dataset_dir / "clips"
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"Missing clips directory: {clips_dir}")

    bucket_names = ("validated", "invalidated", "other")
    buckets = {name: read_tsv(dataset_dir / f"{name}.tsv") for name in bucket_names}
    splits = {name: read_tsv(dataset_dir / f"{name}.tsv") for name in ("train", "dev", "test")}
    for name, rows in {**buckets, **splits}.items():
        _validate_schema(rows, dataset_dir / f"{name}.tsv")

    durations, duplicate_duration_rows = _load_durations(dataset_dir / "clip_durations.tsv")
    physical_audio = {path.name for path in clips_dir.glob("*.mp3")}
    validated = buckets["validated"]
    validated_paths = [row["path"] for row in validated]
    validated_duration_values = [
        durations[path] for path in validated_paths if path in durations
    ]

    # Decode every validated clip with a physically present file. Skipped by
    # default so metadata-only callers (including the fast unit test) stay
    # fast and do not require real decodable audio on disk.
    audio_scan: dict[str, dict[str, Any]] = {}
    if run_audio_scan:
        for path_name in dict.fromkeys(validated_paths):
            if path_name not in physical_audio:
                audio_scan[path_name] = {"decode_status": "missing_file"}
            else:
                audio_scan[path_name] = inspect_audio(clips_dir / path_name)

    speaker_data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "clips": 0,
            "duration_ms": 0,
            "duration_values_ms": [],
            "sentence_ids": set(),
            "words": set(),
            "chars": set(),
            "usable_clips": 0,
            "usable_duration_ms": 0,
        }
    )
    character_counts: Counter[str] = Counter()
    unknown_script_counts: Counter[str] = Counter()
    text_flag_counts: Counter[str] = Counter()
    normalized_to_sentence_ids: dict[str, set[str]] = defaultdict(set)

    for row in validated:
        analysis = analyze_text(row["sentence"])
        normalized = analysis.normalized
        normalized_to_sentence_ids[normalized].add(row["sentence_id"])
        character_counts.update(character for character in normalized if not character.isspace())
        unknown_script_counts.update(analysis.unknown_script_characters)
        text_flag_counts.update(analysis.flags)

        speaker = speaker_data[anonymous_speaker_id(row["client_id"])]
        speaker["clips"] += 1
        speaker["duration_ms"] += durations.get(row["path"], 0)
        if row["path"] in durations:
            speaker["duration_values_ms"].append(durations[row["path"]])
        speaker["sentence_ids"].add(row["sentence_id"])
        speaker["words"].update(word.casefold() for word in words(normalized))
        speaker["chars"].update(character for character in normalized if not character.isspace())
        if run_audio_scan and is_usable(audio_scan.get(row["path"], {})):
            speaker["usable_clips"] += 1
            speaker["usable_duration_ms"] += audio_scan[row["path"]]["duration_ms"]

    speaker_ranking = []
    for speaker_id, values in speaker_data.items():
        speaker_ranking.append(
            {
                "speaker_id": speaker_id,
                "duration_hours": round(values["duration_ms"] / 3_600_000, 6),
                "median_clip_duration_ms": (
                    statistics.median(values["duration_values_ms"])
                    if values["duration_values_ms"]
                    else None
                ),
                "validated_clips": values["clips"],
                "unique_sentences": len(values["sentence_ids"]),
                "lexical_coverage_words": len(values["words"]),
                "character_coverage": len(values["chars"]),
                "usable_clips": values["usable_clips"] if run_audio_scan else None,
                "usable_duration_hours": (
                    round(values["usable_duration_ms"] / 3_600_000, 6) if run_audio_scan else None
                ),
            }
        )
    speaker_ranking.sort(
        key=lambda row: (row["duration_hours"], row["validated_clips"]), reverse=True
    )

    strongest_hours = speaker_ranking[0]["duration_hours"] if speaker_ranking else 0.0
    if strongest_hours >= 3:
        provisional_case = "A"
    elif strongest_hours >= 1:
        provisional_case = "B"
    else:
        provisional_case = "C"

    split_sentence_ids = {
        name: {row["sentence_id"] for row in rows} for name, rows in splits.items()
    }
    split_speakers = {
        name: {anonymous_speaker_id(row["client_id"]) for row in rows}
        for name, rows in splits.items()
    }

    metadata_paths = sorted(dataset_dir.glob("*.tsv")) + [dataset_dir / "README.md"]
    metadata_checksums = {
        path.name: sha256_file(path) for path in metadata_paths if path.is_file()
    }

    all_bucket_speakers = {
        anonymous_speaker_id(row["client_id"])
        for rows in buckets.values()
        for row in rows
        if row["client_id"]
    }
    locale_counts = Counter(row["locale"] for row in validated)
    duplicate_validated_paths = sorted(
        path for path, count in Counter(validated_paths).items() if count > 1
    )

    return {
        "report_type": "Common Voice metadata audit",
        "audit_scope": "metadata and file-presence checks; audio decoding/quality scan not run",
        "language": {
            "code": EXPECTED_LOCALE,
            "name": "Adamawa Fulfulde",
            "dialect_scope": DIALECT_SCOPE,
            "must_not_merge_with": "fuv",
        },
        "source": {
            "dataset": SOURCE_DATASET,
            "dataset_version": dataset_version(dataset_dir),
            "dataset_directory_name": dataset_dir.parent.name,
            "source_archive": {
                "name": None,
                "sha256": None,
                "status": "not_available; corpus was supplied as an extracted directory",
            },
            "licence": SOURCE_LICENCE,
            "speaker_reidentification_prohibited": True,
            "metadata_sha256": metadata_checksums,
        },
        "counts": {
            "physical_mp3_files": len(physical_audio),
            "duration_entries": len(durations),
            "validated_rows": len(validated),
            "invalidated_rows": len(buckets["invalidated"]),
            "other_rows": len(buckets["other"]),
            "validated_unique_sentences": len({row["sentence_id"] for row in validated}),
            "validated_unique_normalized_texts": len(normalized_to_sentence_ids),
            "validated_speakers": len(speaker_data),
            "all_bucket_speakers": len(all_bucket_speakers),
        },
        "integrity": {
            "validated_missing_audio": sorted(set(validated_paths) - physical_audio),
            "validated_missing_duration": sorted(set(validated_paths) - durations.keys()),
            "duration_entries_missing_audio": sorted(durations.keys() - physical_audio),
            "audio_without_duration": sorted(physical_audio - durations.keys()),
            "duplicate_duration_rows": sorted(set(duplicate_duration_rows)),
            "duplicate_validated_paths": duplicate_validated_paths,
            "normalized_text_with_multiple_sentence_ids": sum(
                len(sentence_ids) > 1 for sentence_ids in normalized_to_sentence_ids.values()
            ),
            "validated_locale_counts": dict(sorted(locale_counts.items())),
        },
        "duration": {
            "all_hours_from_clip_durations": round(sum(durations.values()) / 3_600_000, 6),
            "validated_hours_from_clip_durations": round(
                sum(validated_duration_values) / 3_600_000, 6
            ),
            "validated_distribution": _duration_distribution(validated_duration_values),
        },
        "official_splits": {
            name: {
                "rows": len(rows),
                "hours": round(
                    sum(durations.get(row["path"], 0) for row in rows) / 3_600_000, 6
                ),
                "speakers": len(split_speakers[name]),
                "unique_sentences": len(split_sentence_ids[name]),
            }
            for name, rows in splits.items()
        }
        | {
            "sentence_overlap": {
                "train_dev": len(split_sentence_ids["train"] & split_sentence_ids["dev"]),
                "train_test": len(split_sentence_ids["train"] & split_sentence_ids["test"]),
                "dev_test": len(split_sentence_ids["dev"] & split_sentence_ids["test"]),
            },
            "speaker_overlap": {
                "train_dev": len(split_speakers["train"] & split_speakers["dev"]),
                "train_test": len(split_speakers["train"] & split_speakers["test"]),
                "dev_test": len(split_speakers["dev"] & split_speakers["test"]),
            },
        },
        "text": {
            "normalization": "Unicode NFC plus whitespace collapse; no transliteration",
            "character_inventory": [
                {"character": character, "count": count}
                for character, count in sorted(character_counts.items())
            ],
            "unknown_script_inventory": dict(sorted(unknown_script_counts.items())),
            "review_flag_counts": dict(sorted(text_flag_counts.items())),
        },
        "speaker_analysis": {
            "decision_case": provisional_case,
            "decision_is_provisional": True,
            "reason": (
                "Audio-quality filtering has been applied to usable-duration figures, "
                "but the decision remains provisional pending native-speaker review."
                if run_audio_scan
                else "Duration ranking precedes audio-quality filtering and native-speaker review."
            ),
            "ranking": speaker_ranking,
        },
        "audio_quality": _audio_quality_section(audio_scan, run_audio_scan),
        "limitations": [
            "Common Voice was collected primarily for ASR rather than studio-quality TTS.",
            "Speaker duration does not establish recording quality, consent for identity cloning, or production suitability.",
            "The metadata duration table may not match rounded duration statistics in the supplied datasheet.",
            "Native-speaker evaluation has not been performed.",
        ],
        "intended_use": "Adamawa Fulfulde Common Voice research baseline",
    }


def audit_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    duration = report["duration"]
    integrity = report["integrity"]
    splits = report["official_splits"]
    speaker = report["speaker_analysis"]
    missing_audio = len(integrity["validated_missing_audio"])
    missing_duration = len(integrity["validated_missing_duration"])
    lines = [
        "# Common Voice Adamawa Fulfulde metadata audit",
        "",
        f"- Language: Adamawa Fulfulde (`{report['language']['code']}`)",
        f"- Variety: {report['language']['dialect_scope']}",
        f"- Dataset version: `{report['source']['dataset_version']}`",
        f"- Licence: `{report['source']['licence']}`",
        f"- Intended use: {report['intended_use']}",
        "",
        "## Corpus summary",
        "",
        f"- Physical MP3 files: {counts['physical_mp3_files']:,}",
        f"- Validated rows: {counts['validated_rows']:,}",
        f"- Invalidated rows: {counts['invalidated_rows']:,}",
        f"- Other rows: {counts['other_rows']:,}",
        f"- Validated duration from `clip_durations.tsv`: {duration['validated_hours_from_clip_durations']:.3f} hours",
        f"- Validated anonymous speakers: {counts['validated_speakers']}",
        f"- Unique validated sentences: {counts['validated_unique_sentences']:,}",
        f"- Validated clips over 15 seconds: {duration['validated_distribution']['over_15000_ms']}",
        "",
        "## Integrity",
        "",
        f"- Validated records missing audio: {missing_audio}",
        f"- Validated records missing duration: {missing_duration}",
        f"- Duplicate validated paths: {len(integrity['duplicate_validated_paths'])}",
        f"- Validated locale counts: {integrity['validated_locale_counts']}",
        "",
        "## Official split assessment",
        "",
        "| Split | Rows | Hours | Speakers | Sentences |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("train", "dev", "test"):
        row = splits[name]
        lines.append(
            f"| {name} | {row['rows']:,} | {row['hours']:.3f} | {row['speakers']} | {row['unique_sentences']:,} |"
        )
    lines.extend(
        [
            "",
            f"Sentence overlap: `{splits['sentence_overlap']}`",
            "",
            f"Speaker overlap: `{splits['speaker_overlap']}`",
            "",
            "The official splits are suitable as a small benchmark, but they use only a fraction of validated recordings. Build sentence-grouped manifests from all suitable validated rows for multi-speaker training.",
            "",
            "## Provisional speaker decision",
            "",
            f"Decision case: **{speaker['decision_case']}**. {speaker['reason']}",
            "",
            "| Rank | Anonymous speaker | Hours | Usable hours | Clips | Usable clips | Unique sentences |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(speaker["ranking"][:10], start=1):
        usable_hours = row["usable_duration_hours"]
        usable_hours_text = f"{usable_hours:.3f}" if usable_hours is not None else "-"
        usable_clips = row["usable_clips"] if row["usable_clips"] is not None else "-"
        lines.append(
            f"| {rank} | `{row['speaker_id']}` | {row['duration_hours']:.3f} | "
            f"{usable_hours_text} | {row['validated_clips']:,} | "
            f"{usable_clips} | {row['unique_sentences']:,} |"
        )
    audio_quality = report["audio_quality"]
    if audio_quality["status"] == "ok":
        audio_quality_lines = [
            "",
            "## Audio-quality status",
            "",
            f"- Decoded clips: {audio_quality['decoded_clips']:,}",
            f"- Decode failures: {audio_quality['decode_failures']}",
            f"- Usable clips ({audio_quality['usable_definition']}): {audio_quality['usable_clips']:,}",
            f"- Usable duration: {audio_quality['usable_hours']:.3f} hours",
            f"- Source sample rates: {audio_quality['sample_rate_distribution']}",
            f"- Source channel counts: {audio_quality['channel_distribution']}",
            f"- Median duration: {audio_quality['duration_distribution_ms']['median'] / 1000:.3f} seconds",
            f"- 95th-percentile duration: {audio_quality['duration_distribution_ms']['p95'] / 1000:.3f} seconds",
            f"- Clips at/above the clipping threshold: {audio_quality['clips_clipping_ratio_at_or_above_threshold']}",
            f"- Clips at/above the silence threshold: {audio_quality['clips_silence_ratio_at_or_above_threshold']}",
            "",
            "Noise-floor variability beyond peak/RMS/clipping/silence has not been separately measured.",
        ]
    else:
        audio_quality_lines = [
            "",
            "## Audio-quality status",
            "",
            "Audio decoding, corrupt-file detection, sample-rate, clipping, silence, and noise measurements have **not yet run**. File presence is not treated as proof of clean or decodable audio.",
        ]
    lines.extend(audio_quality_lines)
    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def write_audit(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "audit.json"
    markdown_path = output_dir / "audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(audit_markdown(report), encoding="utf-8")
    return json_path, markdown_path
