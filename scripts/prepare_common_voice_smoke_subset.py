#!/usr/bin/env python3
"""Select a 100-300 clip, real multi-speaker subset of Common Voice for the
GPU smoke gate, derive 24 kHz mono WAVs, and write a training-ready manifest.

Speaker selection uses usable_duration_hours from the full decoded-audio
scan (reports/generated/common_voice_fub_v26_0/audit.json), not raw
duration: quality filtering reorders the top speakers, so the two rankings
disagree. See docs/paper.md section 2.1.
Every candidate clip is re-decoded and usability-checked here rather than
trusted from the manifest, since the manifest's duration gate does not
include the clipping/silence gate.

Never modifies or writes into the source cv-corpus directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torchaudio

from fub_tts.common_voice import inspect_audio, is_usable


def read_manifest_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def top_speakers_by_usable_hours(audit_json: Path, count: int) -> list[str]:
    report = json.loads(audit_json.read_text(encoding="utf-8"))
    ranking = report["speaker_analysis"]["ranking"]
    if any(row["usable_duration_hours"] is None for row in ranking):
        raise ValueError(
            f"{audit_json} has no usable_duration_hours -- rerun the audit "
            "with --full-audio-scan first."
        )
    ranked = sorted(ranking, key=lambda row: row["usable_duration_hours"], reverse=True)
    return [row["speaker_id"] for row in ranked[:count]]


def derive_wav(source_path: Path, destination: Path, sample_rate: int) -> float:
    waveform, source_rate = torchaudio.load(str(source_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(destination), waveform, sample_rate)
    return waveform.shape[-1] / sample_rate


def run(args: argparse.Namespace) -> dict[str, Any]:
    speakers = top_speakers_by_usable_hours(args.audit_json, args.num_speakers)
    print(f"Selected speakers by usable hours: {speakers}")

    rows_by_split: dict[str, list[dict[str, str]]] = {
        split: read_manifest_tsv(args.manifest_dir / f"{split}.tsv")
        for split in ("train", "validation", "test")
    }
    per_split_quota = {
        "train": args.train_per_speaker,
        "validation": args.eval_per_speaker,
        "test": args.eval_per_speaker,
    }

    clips_dir = args.dataset_dir / "clips"
    selected: list[dict[str, Any]] = []
    scanned = 0
    for split, quota in per_split_quota.items():
        by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows_by_split[split]:
            if row["speaker_id"] in speakers:
                by_speaker[row["speaker_id"]].append(row)
        for speaker in speakers:
            candidates = sorted(by_speaker[speaker], key=lambda row: row["audio_path"])
            taken = 0
            for row in candidates:
                if taken >= quota:
                    break
                scanned += 1
                source_path = clips_dir / Path(row["audio_path"]).name
                audio = inspect_audio(source_path)
                if not is_usable(audio):
                    continue
                selected.append({**row, "split": split, "speaker_name": speaker})
                taken += 1
            if taken < quota:
                print(
                    f"warning: only found {taken}/{quota} usable {split} clips "
                    f"for speaker {speaker}"
                )

    wav_dir = args.output_dir / "wavs"
    manifest_rows: list[dict[str, Any]] = []
    unique_names: set[str] = set()
    for row in selected:
        source_name = Path(row["audio_path"]).name
        destination = wav_dir / f"{Path(source_name).stem}.wav"
        duration_seconds = derive_wav(clips_dir / source_name, destination, args.sample_rate)
        unique_name = f"common_voice_fub_smoke#{destination.name}"
        assert unique_name not in unique_names
        unique_names.add(unique_name)
        manifest_rows.append(
            {
                "audio_file": str(destination.resolve()),
                "audio_unique_name": unique_name,
                "text": row["normalized_text"],
                "raw_text": row["raw_text"],
                "speaker_name": row["speaker_name"],
                "language": "fub",
                "split": row["split"],
                "duration_seconds": duration_seconds,
                "source_audio_path": str((clips_dir / source_name).resolve()),
                "source_dataset": "Mozilla Common Voice Scripted Speech",
                "licence": "CC0-1.0",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )

    vocabulary = sorted({character for row in manifest_rows for character in row["text"]})
    (args.output_dir / "vocab.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "speakers": speakers,
        "num_speakers": len(speakers),
        "candidates_decoded_for_selection": scanned,
        "selected_clips": len(manifest_rows),
        "splits": {
            split: sum(row["split"] == split for row in manifest_rows)
            for split in ("train", "validation", "test")
        },
        "total_duration_seconds": sum(row["duration_seconds"] for row in manifest_rows),
        "sample_rate": args.sample_rate,
        "vocabulary_size": len(vocabulary),
        "vocabulary": vocabulary,
        "source_modified": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("cv-corpus-26.0-2026-06-12/fub")
    )
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("data/manifests/common_voice_fub_v26_0")
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("reports/generated/common_voice_fub_v26_0/audit.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/derived/common_voice_smoke"),
    )
    parser.add_argument("--num-speakers", type=int, default=5)
    parser.add_argument("--train-per-speaker", type=int, default=45)
    parser.add_argument("--eval-per-speaker", type=int, default=4)
    parser.add_argument("--sample-rate", type=int, default=24_000)
    args = parser.parse_args()
    torch.manual_seed(20260802)
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
