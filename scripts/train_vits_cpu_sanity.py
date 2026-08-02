#!/usr/bin/env python3
"""Run the bounded 50-step character-level VITS CPU sanity check."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text import cleaners as coqui_cleaners


def fub_character_cleaner(text: str) -> str:
    """Preserve Adamawa Fulfulde graphemes while normalizing whitespace."""

    return " ".join(unicodedata.normalize("NFC", text).split())


setattr(coqui_cleaners, "fub_character_cleaner", fub_character_cleaner)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def character_config(vocab_path: Path) -> CharactersConfig:
    vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
    punctuation = []
    characters = []
    for symbol in vocabulary:
        if symbol in {"<pad>", "<unk>"}:
            continue
        category = unicodedata.category(symbol)
        if symbol.isspace() or category.startswith("P"):
            punctuation.append(symbol)
        else:
            characters.append(symbol)
    return CharactersConfig(
        pad="<PAD>",
        eos=None,
        bos=None,
        blank="<BLNK>",
        characters="".join(characters),
        punctuations="".join(punctuation),
        is_unique=False,
        is_sorted=False,
    )


def build_configuration(
    data_dir: Path, output_root: Path, steps: int, seed: int
) -> VitsConfig:
    dataset = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="fub_cpu_sanity",
        path=str(data_dir),
        meta_file_train="coqui/metadata_train.csv",
        meta_file_val="coqui/metadata_validation.csv",
        language="fub",
    )
    return VitsConfig(
        output_path=str(output_root),
        run_name=f"fub_vits_{steps}_steps",
        run_description="Bounded Adamawa Fulfulde character-level VITS CPU sanity run.",
        dashboard_logger="tensorboard",
        audio=VitsAudioConfig(
            sample_rate=24_000,
            fft_size=1_024,
            win_length=1_024,
            hop_length=256,
            num_mels=80,
            mel_fmin=0,
            mel_fmax=11_000,
        ),
        characters=character_config(data_dir / "vocab.json"),
        datasets=[dataset],
        use_phonemes=False,
        text_cleaner="fub_character_cleaner",
        add_blank=True,
        enable_eos_bos_chars=False,
        batch_size=1,
        eval_batch_size=1,
        num_loader_workers=0,
        num_eval_loader_workers=0,
        epochs=1,
        run_eval=True,
        test_delay_epochs=999,
        print_step=10,
        plot_step=100_000,
        save_step=max(steps, 1),
        save_n_checkpoints=1,
        save_all_best=False,
        save_checkpoints=True,
        model_param_stats=False,
        mixed_precision=False,
        training_seed=seed,
        test_sentences=[],
        cudnn_benchmark=False,
        cudnn_deterministic=True,
    )


def safe_json_values(values: dict[str, Any] | None) -> dict[str, float]:
    if not values:
        return {}
    result: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result[key] = float(value)
    return result


def save_waveform(path: Path, waveform: torch.Tensor) -> sf.SoundFile:
    array = waveform.detach().cpu().squeeze().numpy().astype(np.float32)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 0.98:
        array *= 0.98 / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, array, 24_000)
    return sf.info(path)


def manifest_texts(data_dir: Path) -> list[str]:
    records = [
        json.loads(line)
        for line in (data_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    return [record["text"] for record in records]


def tokenizer_audit(model: Vits, texts: list[str]) -> dict[str, Any]:
    """Exercise Coqui's actual cleaner/encoder/decoder for every transcript."""

    model.tokenizer.not_found_characters = []
    round_trip_failures: list[dict[str, str]] = []
    for text in texts:
        cleaned = fub_character_cleaner(text)
        token_ids = model.tokenizer.text_to_ids(text, language="fub")
        decoded = model.tokenizer.ids_to_text(token_ids).replace("<BLNK>", "")
        if decoded != cleaned:
            round_trip_failures.append(
                {"input": text, "cleaned": cleaned, "decoded": decoded}
            )
    return {
        "unknown_characters": sorted(
            set(model.tokenizer.not_found_characters), key=ord
        ),
        "round_trip_failures": round_trip_failures,
        "texts_checked": len(texts),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    project_root = config_path.parent.parent
    sanity = load_json(config_path)
    data_dir = project_root / sanity["output_dir"]
    seed = int(sanity["seed"])
    requested_steps = args.steps or int(sanity["training"]["steps"])
    if requested_steps != 50:
        raise ValueError("This gate is intentionally fixed at exactly 50 steps.")
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(args.cpu_threads)

    model_config = build_configuration(data_dir, output_root, requested_steps, seed)
    train_samples, validation_samples = load_tts_samples(
        model_config.datasets, eval_split=True
    )
    if len(train_samples) != requested_steps:
        raise ValueError(
            f"Expected exactly {requested_steps} one-sample training rows, got {len(train_samples)}."
        )
    model = Vits.init_from_config(
        model_config, samples=train_samples + validation_samples
    )
    texts = manifest_texts(data_dir)
    pretraining_tokenizer_audit = tokenizer_audit(model, texts)
    unknown_before_training = pretraining_tokenizer_audit["unknown_characters"]
    if unknown_before_training or pretraining_tokenizer_audit["round_trip_failures"]:
        raise RuntimeError(
            "VITS tokenizer did not preserve the Fulfulde sample: "
            f"{pretraining_tokenizer_audit}"
        )

    required_characters = set(sanity["selection"]["required_characters"])
    model_symbols = set(model.tokenizer.characters.vocab)
    missing_required_characters = sorted(required_characters - model_symbols, key=ord)
    if missing_required_characters:
        raise RuntimeError(
            f"Required Fulfulde characters missing from VITS: {missing_required_characters}"
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    history: list[dict[str, Any]] = []

    def record_step(trainer: Trainer) -> None:
        values = safe_json_values(
            trainer.keep_avg_train.avg_values
            if trainer.keep_avg_train is not None
            else None
        )
        values["step"] = trainer.total_steps_done + 1
        history.append(values)

    trainer = Trainer(
        TrainerArgs(),
        model.config,
        str(output_root),
        model=model,
        train_samples=train_samples,
        eval_samples=validation_samples,
        callbacks={"on_train_step_end": record_step},
        parse_command_line_args=False,
    )
    started = time.perf_counter()
    trainer.fit()
    elapsed = time.perf_counter() - started
    if trainer.total_steps_done != requested_steps:
        raise RuntimeError(
            f"Expected {requested_steps} completed steps, got {trainer.total_steps_done}."
        )
    trainer.save_checkpoint()
    checkpoint_path = max(
        Path(trainer.output_path).glob("checkpoint_*.pth"),
        key=lambda path: path.stat().st_mtime,
    )

    reloaded = Vits.init_from_config(
        model.config, samples=train_samples + validation_samples
    )
    reloaded.load_checkpoint(model.config, checkpoint_path, eval=True, strict=True)
    records = [
        json.loads(line)
        for line in (data_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    test_record = next(record for record in records if record["split"] == "test")
    test_text = test_record["text"]
    reloaded.tokenizer.not_found_characters = []
    ids = reloaded.tokenizer.text_to_ids(test_text, language="fub")
    inference_unknowns = sorted(set(reloaded.tokenizer.not_found_characters), key=ord)
    if inference_unknowns:
        raise RuntimeError(f"VITS discarded inference characters: {inference_unknowns}")
    tokens = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        generated = reloaded.inference(tokens)["model_outputs"]
    inference_elapsed = time.perf_counter() - inference_started
    audio_path = Path(trainer.output_path) / "sample.wav"
    audio_info = save_waveform(audio_path, generated)

    train_averages = safe_json_values(
        trainer.keep_avg_train.avg_values
        if trainer.keep_avg_train is not None
        else None
    )
    validation_averages = safe_json_values(
        trainer.keep_avg_eval.avg_values
        if trainer.keep_avg_eval is not None
        else None
    )
    finite_history = all(
        math.isfinite(value)
        for item in history
        for value in item.values()
        if isinstance(value, float)
    )
    all_gates_passed = all(
        [
            trainer.total_steps_done == 50,
            not unknown_before_training,
            not missing_required_characters,
            finite_history,
            audio_info.frames > 0,
            audio_info.samplerate == 24_000,
        ]
    )
    result = {
        "purpose": "Fulfulde orthography and pipeline sanity check; not architecture selection",
        "model": "VITS",
        "implementation": "coqui-tts 0.27.5",
        "device": "cpu",
        "torch_version": torch.__version__,
        "steps": trainer.total_steps_done,
        "sample_counts": {"train": len(train_samples), "validation": len(validation_samples), "test": 5},
        "parameter_count": parameter_count,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / max(trainer.total_steps_done, 1),
        "train_averages": train_averages,
        "validation_averages": validation_averages,
        "all_losses_finite": finite_history,
        "required_fulfulde_characters": sorted(required_characters, key=ord),
        "missing_required_characters": missing_required_characters,
        "unknown_characters_before_training": unknown_before_training,
        "tokenizer_texts_checked": pretraining_tokenizer_audit["texts_checked"],
        "tokenizer_round_trip_failures": pretraining_tokenizer_audit[
            "round_trip_failures"
        ],
        "unknown_characters_at_inference": inference_unknowns,
        "checkpoint_reload": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "sample_audio": str(audio_path),
        "sample_text": test_text,
        "sample_duration_seconds": audio_info.duration,
        "sample_rate": audio_info.samplerate,
        "inference_seconds": inference_elapsed,
        "vocoder": "VITS integrated waveform decoder",
        "symbols": reloaded.tokenizer.characters.vocab,
        "raw_dataset_modified": False,
        "all_gates_passed": all_gates_passed,
        "history": history,
    }
    metrics_path = Path(trainer.output_path) / "metrics.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "history"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/fub_cpu_sanity.json")
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/cpu_sanity/vits")
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
