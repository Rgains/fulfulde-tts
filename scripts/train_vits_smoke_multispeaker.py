#!/usr/bin/env python3
"""VITS GPU smoke/full trainer for prepared Adamawa Fulfulde JSONL data.

This exercises real speaker conditioning (multiple anonymous Common Voice
speaker IDs, not the single-sample character path used by the earlier CPU
sanity gates) and captures the run metadata this project requires: fixed seed,
BF16 stability, peak VRAM, step time, strict checkpoint reload, and three
generated samples. It does not claim anything about voice quality or
naturalness -- see docs/common_voice_gpu_smoke_results.md for that caveat
in context.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models import vits as vits_module
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.text import cleaners as coqui_cleaners
from TTS.vocoder.utils.generic_utils import plot_results as _plot_results

SEED = 20260802


def fub_character_cleaner(text: str) -> str:
    """Preserve Fulfulde graphemes; only collapse file-format whitespace."""
    return " ".join(text.split())


setattr(coqui_cleaners, "fub_character_cleaner", fub_character_cleaner)


def _bf16_safe_create_logs(self, batch, outputs):
    """Coqui's Vits._create_logs calls .numpy() on raw BF16 model outputs in
    several places for tensorboard previews; numpy has no bfloat16 dtype.
    Cast to float32 first. Only affects preview images/audio, not training.
    """
    y_hat = outputs[1]["model_outputs"].float()
    y = outputs[1]["waveform_seg"].float()
    figures = _plot_results(y_hat, y, self.ap)
    sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
    audios = {"audio": sample_voice}
    alignments = outputs[1]["alignments"]
    align_img = alignments[0].float().data.cpu().numpy().T
    figures.update({"alignment": vits_module.plot_alignment(align_img, output_fig=False)})
    return figures, audios


vits_module.Vits._create_logs = _bf16_safe_create_logs


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "text": row["text"],
            "audio_file": row["audio_file"],
            "audio_unique_name": row["audio_unique_name"],
            "speaker_name": row["speaker_name"],
            "language": row["language"],
            "root_path": "",
        }
        for row in rows
    ]


def character_config(vocab: list[str]) -> CharactersConfig:
    punctuation_set = {" ", ",", "-", ".", ":", ";", "!", "?", "'", "ʼ", "’", "«", "»"}
    characters = "".join(symbol for symbol in vocab if symbol not in punctuation_set)
    punctuations = "".join(symbol for symbol in vocab if symbol in punctuation_set)
    return CharactersConfig(
        pad="<PAD>",
        eos=None,
        bos=None,
        blank="<BLNK>",
        characters=characters,
        punctuations=punctuations,
        is_unique=False,
        is_sorted=False,
    )


def save_waveform(path: Path, waveform: torch.Tensor, sample_rate: int) -> sf.SoundFile:
    array = waveform.detach().cpu().squeeze().numpy().astype(np.float32)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 0.98:
        array *= 0.98 / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, array, sample_rate)
    return sf.info(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_started = time.perf_counter()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    manifest_dir = args.manifest_dir.resolve()
    rows = load_manifest(manifest_dir / "manifest.jsonl")
    vocab = json.loads((manifest_dir / "vocab.json").read_text(encoding="utf-8"))
    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]
    speakers = sorted({row["speaker_name"] for row in rows})

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    use_speaker_embedding = not args.unconditioned
    config = VitsConfig(
        output_path=str(output_root),
        run_name=args.run_name,
        run_description=args.purpose,
        dashboard_logger="tensorboard",
        audio=VitsAudioConfig(
            sample_rate=24_000,
            fft_size=1_024,
            win_length=1_024,
            hop_length=256,
            num_mels=80,
            mel_fmin=0,
            mel_fmax=8_000,
        ),
        characters=character_config(vocab),
        use_phonemes=False,
        text_cleaner="fub_character_cleaner",
        add_blank=True,
        enable_eos_bos_chars=False,
        # get_from_config_or_model_args_with_default() (TTS/config/__init__.py)
        # checks config.model_args FIRST whenever the key exists there, so a
        # top-level use_speaker_embedding=True on VitsConfig is silently
        # ignored -- VitsArgs.use_speaker_embedding (defaulting False) wins.
        # Confirmed the hard way: a full run trained with no speaker_manager
        # attached at all (single-speaker in effect) before this was caught.
        # Must set it on model_args for SpeakerManager.init_from_config to
        # actually see it.
        model_args=VitsArgs(use_speaker_embedding=use_speaker_embedding),
        use_speaker_embedding=use_speaker_embedding,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_loader_workers=2,
        num_eval_loader_workers=2,
        epochs=args.epochs,
        run_eval=True,
        test_delay_epochs=999,
        print_step=10,
        plot_step=100_000,
        save_step=args.save_step,
        # Smoke checkpoints are ~1 GB each. Retain only the latest rolling
        # checkpoint so a bounded validation run cannot exhaust the EC2 disk.
        save_n_checkpoints=1,
        save_all_best=False,
        # BF16 tensors reaching TTS.vocoder.utils.generic_utils.plot_results
        # crash on `.numpy()` (unsupported ScalarType BFloat16). log_model_step
        # defaults to save_step and fires that path; push it past this run's
        # total steps so BF16 training itself is unaffected.
        log_model_step=1_000_000,
        save_checkpoints=True,
        model_param_stats=False,
        mixed_precision=True,
        precision="bf16",
        training_seed=SEED,
        test_sentences=[],
        cudnn_benchmark=True,
        cudnn_deterministic=False,
    )

    train_samples = to_samples(train_rows)
    eval_samples = to_samples(eval_rows)
    model = Vits.init_from_config(config, samples=train_samples + eval_samples)
    if use_speaker_embedding and model.speaker_manager is None:
        raise RuntimeError("Multi-speaker model initialized without a speaker manager")
    speaker_name_to_id = (
        dict(model.speaker_manager.name_to_id) if model.speaker_manager is not None else {}
    )

    unknown_before = set()
    for sample in train_samples + eval_samples + to_samples(test_rows):
        ids = model.tokenizer.text_to_ids(sample["text"], language=sample["language"])
        unknown_before.update(model.tokenizer.not_found_characters)
    if unknown_before:
        raise RuntimeError(f"Unknown characters found before training: {sorted(unknown_before)}")

    parameter_count = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    trainer = Trainer(
        TrainerArgs(),
        config,
        str(output_root),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        parse_command_line_args=False,
    )
    device = "cuda" if trainer.use_cuda else "cpu"
    trainer.fit()
    training_elapsed = time.perf_counter() - started
    trainer.save_checkpoint()
    checkpoint_path = max(
        Path(trainer.output_path).glob("checkpoint_*.pth"),
        key=lambda path: path.stat().st_mtime,
    )
    peak_allocated = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2) if device == "cuda" else None

    reloaded = Vits.init_from_config(config, samples=train_samples + eval_samples)
    reloaded.load_checkpoint(config, checkpoint_path, eval=True, strict=True)
    if use_speaker_embedding and reloaded.speaker_manager is None:
        # Re-use the exact SpeakerManager the checkpoint's embedding indices
        # were trained against, rather than trust a second name_to_id build
        # (observed None here on reload; embeddings are index-based, so any
        # mismatched re-derivation would silently mix up speaker identity).
        reloaded.speaker_manager = model.speaker_manager
    if device == "cuda":
        reloaded = reloaded.to("cuda")

    generated_samples: list[dict[str, Any]] = []
    unknown_at_inference: set[str] = set()
    for index, row in enumerate(test_rows[:3]):
        ids = reloaded.tokenizer.text_to_ids(row["text"], language="fub")
        unknown_at_inference.update(reloaded.tokenizer.not_found_characters)
        # Use the exact name-to-index mapping captured before training. The
        # checkpoint contains the indexed embedding weights but Coqui does not
        # reliably reconstruct SpeakerManager when loading this configuration.
        tokens = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        aux_input: dict[str, torch.Tensor] = {}
        if use_speaker_embedding:
            speaker_id = speaker_name_to_id[row["speaker_name"]]
            speaker_ids = torch.tensor([speaker_id], dtype=torch.long)
            aux_input["speaker_ids"] = speaker_ids
        if device == "cuda":
            tokens = tokens.to("cuda")
            aux_input = {key: value.to("cuda") for key, value in aux_input.items()}
        with torch.inference_mode():
            generated = reloaded.inference(tokens, aux_input=aux_input)["model_outputs"]
        audio_path = Path(trainer.output_path) / f"sample_{index}.wav"
        info = save_waveform(audio_path, generated, config.audio.sample_rate)
        generated_samples.append(
            {
                "text": row["text"],
                "speaker_name": row["speaker_name"],
                "audio_path": str(audio_path),
                "duration_seconds": info.duration,
                "sample_rate": info.samplerate,
            }
        )
    if unknown_at_inference:
        raise RuntimeError(f"Unknown characters at inference: {sorted(unknown_at_inference)}")

    result = {
        "purpose": args.purpose,
        "device": device,
        "torch_version": torch.__version__,
        "precision": config.precision,
        "mixed_precision": config.mixed_precision,
        "seed": SEED,
        "speaker_conditioning": use_speaker_embedding,
        "num_speakers": len(speakers) if use_speaker_embedding else None,
        "speaker_names": speakers,
        "batch_size": args.batch_size,
        "epochs_configured": args.epochs,
        "steps": trainer.total_steps_done,
        "parameter_count": parameter_count,
        "train_clips": len(train_samples),
        "validation_clips": len(eval_samples),
        "test_clips": len(test_rows),
        # Training time ends at fit(); total time also includes final save,
        # strict reload, synthesis, and artifact serialization up to this point.
        "training_elapsed_seconds": training_elapsed,
        "total_elapsed_seconds": time.perf_counter() - run_started,
        "seconds_per_step": training_elapsed / max(trainer.total_steps_done, 1),
        "peak_vram_allocated_mib": peak_allocated,
        "peak_vram_reserved_mib": peak_reserved,
        "unknown_characters_before_training": sorted(unknown_before),
        "unknown_characters_at_inference": sorted(unknown_at_inference),
        "checkpoint_reload": True,
        "checkpoint": str(checkpoint_path),
        "generated_samples": generated_samples,
        "vocabulary_size": len(vocab),
        "raw_dataset_modified": False,
    }
    metrics_path = Path(trainer.output_path) / "metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    import shutil

    shutil.copy(metrics_path, log_dir / args.metrics_name)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("data/derived/common_voice_smoke")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ubuntu/fulfulde-tts/checkpoints/common-voice-smoke-multispeaker"),
    )
    parser.add_argument("--log-dir", type=Path, default=Path("/home/ubuntu/fulfulde-tts/logs"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--save-step", type=int, default=100)
    parser.add_argument("--metrics-name", default="metrics.json")
    parser.add_argument("--run-name", default="fub_common_voice_smoke_multispeaker")
    parser.add_argument(
        "--purpose",
        default=(
            "Multi-speaker GPU smoke gate: real speaker conditioning and data flow, "
            "not a quality or naturalness result."
        ),
    )
    parser.add_argument(
        "--unconditioned",
        action="store_true",
        help="train without speaker embeddings when the corpus has no speaker metadata",
    )
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
