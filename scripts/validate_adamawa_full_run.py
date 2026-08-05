#!/usr/bin/env python3
"""Validate completed full-corpus artifacts and write machine-readable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import soundfile as sf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def latest_run(output_root: Path) -> Path:
    candidates = [path for path in output_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directory under {output_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project_root.resolve()
    run_dir = latest_run(args.output_root.resolve())
    metrics_path = run_dir / "metrics.json"
    require(metrics_path.is_file(), f"Missing final metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    require(metrics["device"] == "cuda", "Full run did not report CUDA")
    require(metrics["precision"] == "bf16", "Full run did not report BF16")
    require(metrics["steps"] == args.expected_steps, "Unexpected completed step count")
    require(metrics["train_clips"] == 844, "Unexpected training clip count")
    require(metrics["validation_clips"] == 112, "Unexpected validation clip count")
    require(metrics["test_clips"] == 99, "Unexpected test clip count")
    require(metrics["speaker_conditioning"] is False, "Unexpected speaker conditioning")
    require(metrics["checkpoint_reload"] is True, "Strict checkpoint reload did not pass")
    require(not metrics["unknown_characters_before_training"], "Pretraining unknown characters")
    require(not metrics["unknown_characters_at_inference"], "Inference unknown characters")
    require(len(metrics["generated_samples"]) == 3, "Expected exactly three samples")
    require(metrics["raw_dataset_modified"] is False, "Run reports modifying source data")

    checkpoint = Path(metrics["checkpoint"])
    require(checkpoint.is_file(), f"Missing checkpoint: {checkpoint}")
    samples: list[dict[str, Any]] = []
    for row in metrics["generated_samples"]:
        path = Path(row["audio_path"])
        require(path.is_file(), f"Missing sample: {path}")
        info = sf.info(path)
        require(info.samplerate == 24_000, f"Unexpected sample rate: {path}")
        require(info.channels == 1, f"Unexpected channel count: {path}")
        require(info.frames > 0, f"Empty generated sample: {path}")
        samples.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "duration_seconds": info.duration,
                "format": info.format,
                "subtype": info.subtype,
                "text": row["text"],
            }
        )

    summary_path = project / "data/derived/adamawa_training/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary["selected_clips"] == 1055, "Unexpected clean-pool size")
    require(summary["splits"] == {"train": 844, "validation": 112, "test": 99},
            "Unexpected clean-pool split counts")

    source_mapping = project / "Adamawa-Fulfulde-TTS-Dataset/Mapping_MP3.tsv"
    result = {
        "status": "passed",
        "run_directory": str(run_dir),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256(metrics_path),
        "steps": metrics["steps"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_reload": metrics["checkpoint_reload"],
        "unknown_characters_before_training": metrics["unknown_characters_before_training"],
        "unknown_characters_at_inference": metrics["unknown_characters_at_inference"],
        "samples": samples,
        "clean_pool_summary": summary,
        "clean_pool_summary_sha256": sha256(summary_path),
        "source_mapping_sha256_after_training": sha256(source_mapping),
        "expected_source_mapping_sha256": (
            "9d43f2616f656bd3d46178381dc83dbd3fddd1dbf39e63833a6ab2d8db73a3b5"
        ),
        "source_mapping_unchanged": sha256(source_mapping)
        == "9d43f2616f656bd3d46178381dc83dbd3fddd1dbf39e63833a6ab2d8db73a3b5",
    }
    require(result["source_mapping_unchanged"], "Source mapping changed during training")
    output = run_dir / "validation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-root", type=Path, default=Path("/home/ubuntu/fulfulde-tts/checkpoints/adamawa-full")
    )
    parser.add_argument("--expected-steps", type=int, default=53_000)
    validate(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
