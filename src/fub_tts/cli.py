"""Command-line interface for the Adamawa Fulfulde data pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .common_voice import audit_common_voice, write_audit
from .split import build_manifests


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fub-tts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="audit Common Voice metadata and file presence")
    audit.add_argument("--dataset-dir", type=Path, required=True)
    audit.add_argument("--report-dir", type=Path, required=True)
    audit.add_argument(
        "--full-audio-scan",
        action="store_true",
        help="decode every validated clip and measure real audio quality (slow)",
    )

    manifests = subparsers.add_parser(
        "manifests", help="build deterministic sentence-disjoint manifests"
    )
    manifests.add_argument("--dataset-dir", type=Path, required=True)
    manifests.add_argument("--manifest-dir", type=Path, required=True)
    manifests.add_argument("--seed", default="fub-common-voice-v1")
    manifests.add_argument("--maximum-duration-ms", type=int, default=15_000)

    prepare = subparsers.add_parser("prepare", help="run the audit and build manifests")
    prepare.add_argument("--dataset-dir", type=Path, required=True)
    prepare.add_argument("--report-dir", type=Path, required=True)
    prepare.add_argument("--manifest-dir", type=Path, required=True)
    prepare.add_argument("--seed", default="fub-common-voice-v1")
    prepare.add_argument("--maximum-duration-ms", type=int, default=15_000)
    prepare.add_argument(
        "--full-audio-scan",
        action="store_true",
        help="decode every validated clip and measure real audio quality (slow)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command in {"audit", "prepare"}:
        report = audit_common_voice(args.dataset_dir, run_audio_scan=args.full_audio_scan)
        json_path, markdown_path = write_audit(report, args.report_dir)
        print(f"Wrote {json_path}")
        print(f"Wrote {markdown_path}")

    if args.command in {"manifests", "prepare"}:
        summary = build_manifests(
            args.dataset_dir,
            args.manifest_dir,
            seed=args.seed,
            maximum_duration_ms=args.maximum_duration_ms,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0

