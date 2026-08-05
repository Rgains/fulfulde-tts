#!/usr/bin/env python3
"""Prepare one shared 24 kHz clean pool plus smoke/full JSONL manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torchaudio


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["audio_filename"]: row for row in rows}


def derive_wav(source: Path, destination: Path, sample_rate: int) -> float:
    waveform, source_rate = torchaudio.load(str(source))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    peak = float(waveform.abs().max()) if waveform.numel() else 0.0
    if peak > 0.98:
        waveform *= 0.98 / peak
    destination.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(destination), waveform, sample_rate)
    return waveform.shape[-1] / sample_rate


def select_smoke(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quotas = {"train": 225, "validation": 20, "test": 20}
    selected: list[dict[str, Any]] = []
    for split, quota in quotas.items():
        candidates = sorted(
            (row for row in rows if row["split"] == split),
            key=lambda row: row["audio_unique_name"],
        )
        selected.extend(candidates[:quota])

    # Ensure the bounded training subset exercises the corpus's important
    # graphemes and doubled vowels. Add rows instead of replacing them so the
    # selection remains deterministic and stays below the 300-clip gate.
    required = ["ɓ", "Ɓ", "ɗ", "Ɗ", "ƴ", "Ƴ", "ŋ", "’", "aa", "ee", "ii", "oo", "uu"]
    train_selected = [row for row in selected if row["split"] == "train"]
    train_candidates = sorted(
        (row for row in rows if row["split"] == "train"),
        key=lambda row: row["audio_unique_name"],
    )
    selected_names = {row["audio_unique_name"] for row in selected}
    for token in required:
        if any(token in row["text"] for row in train_selected):
            continue
        row = next(row for row in train_candidates if token in row["text"])
        if row["audio_unique_name"] not in selected_names:
            selected.append(row)
            train_selected.append(row)
            selected_names.add(row["audio_unique_name"])
    return selected


def write_bundle(path: Path, rows: list[dict[str, Any]], mode: str, sample_rate: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (row["split"], row["audio_unique_name"]))
    (path / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8",
    )
    vocabulary = sorted({character for row in ordered for character in row["text"]})
    (path / "vocab.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "mode": mode,
        "source_dataset": "Adamawa-Fulfulde-TTS-Dataset",
        "speaker_metadata_status": "not supplied",
        "speaker_conditioning": False,
        "selected_clips": len(ordered),
        "splits": {
            split: sum(row["split"] == split for row in ordered)
            for split in ("train", "validation", "test")
        },
        "total_duration_seconds": sum(row["duration_seconds"] for row in ordered),
        "sample_rate": sample_rate,
        "vocabulary_size": len(vocabulary),
        "raw_dataset_modified": False,
        "filters": ["1-15 seconds", "silence_ratio < 0.60", "no repeated_apostrophe flag"],
    }
    (path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    project = args.project_root.resolve()
    source_audio = project / "Adamawa-Fulfulde-TTS-Dataset/audio_files_mp3"
    manifests = project / "data/manifests/adamawa_fulfulde_tts_dataset"
    records = load_records(
        project / "reports/generated/adamawa_fulfulde_tts_dataset/records.jsonl"
    )
    rows: list[dict[str, Any]] = []
    wav_dir = args.pool_dir.resolve() / "wavs"
    for split in ("train", "validation", "test"):
        for row in read_tsv(manifests / f"{split}.tsv"):
            filename = Path(row["audio_path"]).name
            record = records[filename]
            if record["audio"]["silence_ratio"] >= 0.60:
                continue
            if "repeated_apostrophe" in record["text_flags"]:
                continue
            destination = wav_dir / f"{Path(filename).stem}.wav"
            if destination.is_file():
                duration = torchaudio.info(str(destination)).num_frames / args.sample_rate
            else:
                duration = derive_wav(source_audio / filename, destination, args.sample_rate)
            rows.append(
                {
                    "text": row["normalized_text"],
                    "raw_text": row["raw_text"],
                    "audio_file": str(destination.resolve()),
                    "audio_unique_name": f"adamawa_fub#{destination.name}",
                    "speaker_name": "unknown",
                    "speaker_metadata_status": "not supplied",
                    "language": "fub",
                    "split": split,
                    "duration_seconds": duration,
                    "source_audio_path": str((source_audio / filename).resolve()),
                    "source_dataset": "Adamawa-Fulfulde-TTS-Dataset",
                    "licence": "user-confirmed licence; exact identifier pending",
                }
            )
    write_bundle(args.pool_dir.resolve(), rows, "full", args.sample_rate)
    write_bundle(args.smoke_dir.resolve(), select_smoke(rows), "smoke", args.sample_rate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--pool-dir", type=Path, default=Path("data/derived/adamawa_training"))
    parser.add_argument("--smoke-dir", type=Path, default=Path("data/derived/adamawa_smoke"))
    parser.add_argument("--sample-rate", type=int, default=24_000)
    args = parser.parse_args()
    torch.manual_seed(20260802)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
