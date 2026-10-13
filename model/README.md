# Adamawa Fulfulde (`fub`) VITS research baseline — model card

A character-level VITS text-to-speech model for Adamawa Fulfulde, trained from
the locally supplied `Adamawa-Fulfulde-TTS-Dataset`. It is a **research
baseline**, not a production model: no native-speaker listening evaluation has
been performed, so no claim is made about intelligibility, pronunciation,
speaker consistency, or naturalness.

The full method, measurements, and engineering record are in
[`docs/adamawa_full_training_report.md`](../docs/adamawa_full_training_report.md).
This card states only what a user of the weights needs before using them.

## Licence

**The repository's MIT licence does not cover these weights.** MIT applies to
this repository's own source and documentation. The checkpoint is a derived work
of a NOODL-1.0 dataset and is governed by that lineage:

- The source `Adamawa-Fulfulde-TTS-Dataset` is distributed under **NOODL-1.0**.
  Redistribution of the source speech dataset is **prohibited**, and the dataset
  is kept outside Git.
- Release of this derived checkpoint for this research demo was authorised by a
  **separate email agreement** with the dataset holder, confirmed 2026-08-07.
  That private correspondence is not stored in the repository. The permission
  covers this research-demo release; it is not a general redistribution grant.
- A formal licence for the weights will be declared at the point of public
  release, once the dataset permission, dependency review, and native-speaker
  evaluation are complete. Until then, treat the checkpoint as
  research-use-only.

If you need to redistribute or build on these weights, ask first rather than
assuming MIT terms.

## Language variety

Adamawa Fulfulde (`fub`), recorded in Nigeria (confirmed 2026-08-10). Adamawa
Fulfulde is spoken across northeastern Nigeria in Adamawa and Taraba States, as
well as in Cameroon and Chad; the region is transnational, named for the
19th-century Adamawa Emirate that spanned both sides of the present border.

The corpus keeps its precise ISO 639-3 code, `fub`, and is never relabelled as
`fuv`, the separate code for the Fulfulde varieties of northwestern and
north-central Nigeria. Both varieties are spoken in Nigeria; the codes
distinguish which variety was recorded, not which country.

The pipeline preserves Fulfulde orthography: `ɓ`, `ɗ`, `ƴ`, `ŋ` are not
transliterated, doubled vowels are not collapsed, and the typographic apostrophe
`’` is preserved where accepted text contains it.

This corpus is distinct from the Common Voice `fub` data also audited in this
repository, which was recorded in Cameroon. The two are never merged.

## Lineage and training summary

| | |
|---|---|
| Architecture | Character-level VITS (Kim, Kong & Son, 2021), direct graphemes, no phonemiser |
| Implementation | Coqui TTS 0.27.5, PyTorch 2.8.0 |
| Parameters | 83,045,932 |
| Audio | 24 kHz mono, 1024-sample FFT/window, 256 hop, 80 mel channels, 8 kHz mel ceiling |
| Training data | `Adamawa-Fulfulde-TTS-Dataset`, local snapshot 2026-08-02 |
| Clean pool | 1,055 clips / 1.931 h — 844 train, 112 validation, 99 test |
| Splits | Deterministic, grouped by normalised transcript; zero text and audio-path leakage |
| Speaker conditioning | **None.** The mapping supplies no speaker field, so speaker coverage is unknown and single-speaker status is not claimed |
| Run | 53,000 optimizer steps, 1,000 epochs, batch 16, BF16, seed 20260802, single NVIDIA L4, 6 h 54 min |
| Vocabulary | 60 characters; zero unknown characters before training and at inference |
| Checkpoint | 997,762,755 bytes, SHA-256 `7dfd08b18b968f41ba4a5a09e7d61c197817647b7747aaf013ad18610b2940a8`, strict reload verified |

Final held-out optimisation diagnostics were generator aggregate 35.9234, mel
23.8307, KL 3.0755, duration 2.6026, feature 4.4774, discriminator 2.5840.
These are training diagnostics, not perceptual scores, and must not be reported
as quality measurements.

## Getting the weights

The weights are not in this repository. They are hosted on the Hugging Face Hub
and the demo downloads them at runtime:

```bash
export FUB_HF_REPO=ejnuma/adamawa-fulfulde-tts   # default
export HF_TOKEN=...                              # access is not yet public
streamlit run scripts/demo_streamlit_adamawa.py
```

The Hub repository is currently private pending the licence declaration above,
so a token is required. `model/` otherwise holds only generated artifacts
(`metrics.json`, `validation.json`, sample WAVs), which are excluded from Git;
this card is the one tracked file in it.

## Limitations

- No native-speaker listening evaluation. This is the blocking gap before any
  quality claim.
- No speaker metadata, so evaluation cannot be speaker-disjoint and speaker
  consistency cannot be assessed. The held-out split is transcript-disjoint
  only.
- 1.931 hours of clean audio is small for training a modern TTS model from
  scratch.
- Valid waveform output and finite losses demonstrate that the pipeline runs.
  They are not evidence of intelligibility or naturalness.
- Not validated for disaster or emergency messaging. Terminology coverage for
  that domain has not been tested.
