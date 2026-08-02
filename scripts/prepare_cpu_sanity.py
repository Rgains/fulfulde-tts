#!/usr/bin/env python3
"""Prepare a deterministic 60-clip Fulfulde subset for the CPU VITS gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio


SPECIAL_TOKENS = ("<pad>", "<unk>")


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")
    return config


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def eligible_rows(
    rows: list[dict[str, str]],
    minimum_seconds: float,
    maximum_seconds: float,
    allowed_text_flags: set[str],
) -> list[dict[str, str]]:
    result = []
    for row in rows:
        duration = int(row["duration_ms"]) / 1000
        if not minimum_seconds <= duration <= maximum_seconds:
            continue
        if row["language_code"] != "fub":
            continue
        row_flags = {flag for flag in row["text_flags"].split(",") if flag}
        if row_flags - allowed_text_flags:
            continue
        result.append(row)
    return result


def row_features(
    row: dict[str, str], characters: set[str], sequences: set[str]
) -> set[str]:
    text = row["normalized_text"]
    features = {f"char:{character}" for character in set(text) & characters}
    features.update(f"sequence:{sequence}" for sequence in sequences if sequence in text.casefold())
    return features


def stable_order(rows: list[dict[str, str]], seed: int, label: str) -> list[dict[str, str]]:
    rng = random.Random(f"{seed}:{label}")
    ordered = list(rows)
    rng.shuffle(ordered)
    return ordered


def coverage_select(
    rows: list[dict[str, str]],
    count: int,
    *,
    characters: set[str],
    sequences: set[str],
    seed: int,
    label: str,
) -> tuple[list[dict[str, str]], set[str]]:
    """Greedily cover features, then deterministically fill the requested size."""

    if len(rows) < count:
        raise RuntimeError(f"Only {len(rows)} eligible {label} rows; {count} requested.")
    ordered = stable_order(rows, seed, label)
    target = {
        feature
        for row in ordered
        for feature in row_features(row, characters, sequences)
    }
    missing = set(target)
    selected: list[dict[str, str]] = []
    remaining = list(ordered)
    while missing and remaining and len(selected) < count:
        best_index, best_row = max(
            enumerate(remaining),
            key=lambda item: (
                len(row_features(item[1], characters, sequences) & missing),
                -item[0],
            ),
        )
        gain = row_features(best_row, characters, sequences) & missing
        if not gain:
            break
        selected.append(best_row)
        missing -= gain
        remaining.pop(best_index)
    selected.extend(remaining[: count - len(selected)])
    return selected, missing


def decode_resample(source: Path, target: Path, sample_rate: int) -> dict[str, Any]:
    samples, source_rate = sf.read(source, dtype="float32", always_2d=True)
    mono = np.mean(samples, axis=1, dtype=np.float32)
    waveform = torch.from_numpy(mono).unsqueeze(0)
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f"Unreadable or non-finite audio: {source}")
    source_peak = float(waveform.abs().max())
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    output_peak = float(waveform.abs().max())
    if output_peak > 1:
        waveform = waveform.clamp(-1, 1)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, waveform.squeeze(0).numpy(), sample_rate, subtype="PCM_16")
    info = sf.info(target)
    return {
        "source_sample_rate": int(source_rate),
        "sample_rate": int(info.samplerate),
        "frames": int(info.frames),
        "duration_seconds": float(info.duration),
        "source_peak": source_peak,
        "resampled_peak": output_peak,
        "range_clamped": output_peak > 1,
    }


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def prepare(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    project_root = config_path.resolve().parent.parent
    dataset_dir = project_root / config["dataset_dir"]
    manifest_dir = project_root / config["manifest_dir"]
    output_dir = project_root / config["output_dir"]
    selection = config["selection"]
    seed = int(config["seed"])
    minimum_seconds = float(selection["minimum_duration_seconds"])
    maximum_seconds = float(selection["maximum_duration_seconds"])
    required_characters = set(selection["required_characters"])
    required_sequences = set(selection["required_sequences"])
    allowed_text_flags = set(selection.get("allowed_text_flags", []))

    source_manifest_paths = {
        split: manifest_dir / f"{split}.tsv"
        for split in ("train", "validation", "test")
    }
    manifest_hashes_before = {
        split: sha256_file(path) for split, path in source_manifest_paths.items()
    }
    candidates = {
        split: eligible_rows(
            load_manifest(path),
            minimum_seconds,
            maximum_seconds,
            allowed_text_flags,
        )
        for split, path in source_manifest_paths.items()
    }

    # Cover every character available in eligible training text, not merely the
    # language-specific subset. This makes punctuation and case failures visible.
    training_characters = set(
        "".join(row["normalized_text"] for row in candidates["train"])
    )
    selected_train, missing_train_features = coverage_select(
        candidates["train"],
        int(selection["train_count"]),
        characters=training_characters,
        sequences=required_sequences,
        seed=seed,
        label="train",
    )
    if missing_train_features:
        raise RuntimeError(
            "The 50-row training sample could not cover: "
            f"{sorted(missing_train_features)}"
        )

    selected: dict[str, list[dict[str, str]]] = {"train": selected_train}
    for split in ("validation", "test"):
        selected[split], _ = coverage_select(
            candidates[split],
            int(selection[f"{split}_count"]),
            characters=required_characters,
            sequences=required_sequences,
            seed=seed,
            label=split,
        )

    train_vocabulary_characters = sorted(
        set("".join(row["normalized_text"] for row in selected_train)), key=ord
    )
    vocabulary = [*SPECIAL_TOKENS, *train_vocabulary_characters]
    char_to_id = {character: index for index, character in enumerate(vocabulary)}
    unknowns: dict[str, list[str]] = {}
    for split, rows in selected.items():
        unknowns[split] = sorted(
            {
                character
                for row in rows
                for character in row["normalized_text"]
                if character not in char_to_id
            },
            key=ord,
        )
    if any(unknowns.values()):
        raise RuntimeError(f"Held-out sample contains unknown characters: {unknowns}")

    records: list[dict[str, Any]] = []
    source_audio_hashes_before: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for index, row in enumerate(selected[split]):
            relative_audio_path = Path(row["audio_path"])
            source_audio = dataset_dir / relative_audio_path
            source_audio_hashes_before[str(source_audio)] = sha256_file(source_audio)
            key = f"{split}_{index:03d}_{source_audio.stem}"
            derived_audio = output_dir / "wavs" / f"{key}.wav"
            audio = decode_resample(
                source_audio,
                derived_audio,
                int(config["audio"]["sample_rate"]),
            )
            encoded = [char_to_id.get(character, char_to_id["<unk>"]) for character in row["normalized_text"]]
            if char_to_id["<unk>"] in encoded:
                raise RuntimeError(f"Tokenizer produced <unk> for {row['audio_path']}")
            decoded = "".join(vocabulary[token_id] for token_id in encoded)
            if decoded != row["normalized_text"]:
                raise RuntimeError(f"Tokenizer round trip failed for {row['audio_path']}")
            records.append(
                {
                    "key": key,
                    "source_audio_path": str(source_audio),
                    "audio_path": str(derived_audio),
                    "sentence_id": row["sentence_id"],
                    "speaker_id": row["speaker_id"],
                    "source_dataset": row["source_dataset"],
                    "dataset_version": row["dataset_version"],
                    "language_code": row["language_code"],
                    "licence": row["licence"],
                    "raw_text": row["raw_text"],
                    "text": row["normalized_text"],
                    "split": split,
                    "audio": audio,
                }
            )

    source_audio_hashes_after = {
        path: sha256_file(Path(path)) for path in source_audio_hashes_before
    }
    source_audio_unchanged = source_audio_hashes_before == source_audio_hashes_after
    manifest_hashes_after = {
        split: sha256_file(path) for split, path in source_manifest_paths.items()
    }
    if not source_audio_unchanged or manifest_hashes_before != manifest_hashes_after:
        raise RuntimeError("A source audio file or manifest changed during preparation.")

    write_lines(
        output_dir / "manifest.jsonl",
        [json.dumps(record, ensure_ascii=False) for record in records],
    )
    (output_dir / "vocab.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for split in ("train", "validation", "test"):
        subset = [record for record in records if record["split"] == split]
        write_lines(
            output_dir / "coqui" / f"metadata_{split}.csv",
            [f"{record['key']}|{record['raw_text']}|{record['text']}" for record in subset],
        )

    counts = Counter(record["split"] for record in records)
    required_character_coverage = {
        character: sum(character in record["text"] for record in records)
        for character in sorted(required_characters, key=ord)
    }
    required_sequence_coverage = {
        sequence: sum(sequence in record["text"].casefold() for record in records)
        for sequence in sorted(required_sequences)
    }
    summary = {
        "config": str(config_path),
        "source_dataset": records[0]["source_dataset"],
        "dataset_version": records[0]["dataset_version"],
        "language_code": "fub",
        "licence": records[0]["licence"],
        "samples": len(records),
        "splits": dict(counts),
        "duration_seconds": round(sum(record["audio"]["duration_seconds"] for record in records), 6),
        "sample_rate": int(config["audio"]["sample_rate"]),
        "vocabulary_size": len(vocabulary),
        "vocabulary": vocabulary,
        "unknown_characters": unknowns,
        "required_character_coverage": required_character_coverage,
        "required_sequence_coverage": required_sequence_coverage,
        "source_sample_rates": dict(
            Counter(str(record["audio"]["source_sample_rate"]) for record in records)
        ),
        "source_peak_over_one": sum(record["audio"]["source_peak"] > 1 for record in records),
        "resampled_peak_over_one": sum(record["audio"]["range_clamped"] for record in records),
        "manifest_sha256_before": manifest_hashes_before,
        "manifest_sha256_after": manifest_hashes_after,
        "source_audio_hashes_unchanged": source_audio_unchanged,
        "raw_dataset_modified": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/fub_cpu_sanity.json")
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
