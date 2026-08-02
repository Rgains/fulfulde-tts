# Adamawa Fulfulde 50-step CPU sanity gate

## Purpose

This is the same bounded character-path gate previously used before the Tiv
GPU run, adapted to Adamawa Fulfulde (`fub`). It does not reopen the
architecture decision and does not measure naturalness. VITS remains the
selected primary architecture.

The check answers only whether the selected implementation can:

- decode a small deterministic Common Voice subset;
- preserve direct Fulfulde graphemes without phonemization or transliteration;
- cover `ɓ`, `Ɓ`, `ɗ`, `Ɗ`, `ƴ`, `Ƴ`, `ŋ`, `Ŋ`, and `ʼ`;
- preserve doubled vowels;
- complete exactly 50 finite CPU optimization steps;
- save and strictly reload a checkpoint;
- tokenize held-out Fulfulde text without unknown characters; and
- emit a valid 24 kHz waveform after reload.

## Controlled sample

- 60 validated Common Voice clips: 50 train, 5 validation, 5 test.
- Deterministic seed: `20260802`.
- Eligible source duration: 2-10 seconds.
- Training selection greedily covers every character present in the eligible
  training corpus, then fills deterministically.
- Audio is derived as mono 24 kHz PCM WAV; source MP3s and manifests are hashed
  before and after preparation.
- Direct character input, no pretrained weights, batch size 1, CPU only.

## Commands

The locked `cpu-sanity` dependency group provides Python 3.11-compatible
PyTorch 2.8, torchaudio, SoundFile, Trainer, and Coqui-TTS 0.27.5.

```bash
uv run --group cpu-sanity python scripts/prepare_cpu_sanity.py \
  --config configs/fub_cpu_sanity.json
uv run --group cpu-sanity python scripts/train_vits_cpu_sanity.py \
  --config configs/fub_cpu_sanity.json \
  --steps 50 \
  --cpu-threads 4
```

All derived audio, checkpoints, and metrics are written below
`outputs/cpu_sanity/`, which is excluded from Git.
