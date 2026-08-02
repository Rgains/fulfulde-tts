# Adamawa Fulfulde 50-step CPU sanity results

## Outcome

The bounded VITS gate passed. No Adamawa Fulfulde orthography-specific
character loss was found in sample preparation, Coqui tokenization, training,
checkpoint reload, or held-out inference.

This result confirms only that the character-level data path and optimization
pipeline work. Fifty updates from random initialization do not establish
naturalness, intelligibility, speaker consistency, or production readiness.

## Controlled run

- Date: 2026-08-02
- Device: CPU, four PyTorch threads
- Python: 3.11.15
- PyTorch: 2.8.0
- Implementation: Coqui-TTS 0.27.5 VITS
- Pretrained weights: none
- Sample: 60 validated Common Voice clips (50 train, 5 validation, 5 test)
- Derived audio: mono 24 kHz PCM WAV from 32 kHz source MP3
- Derived sample duration: 367.956 seconds
- Optimization budget: exactly 50 one-sample steps
- Parameters: 83,047,276

## Orthography checks

- Vocabulary: 69 entries, including special tokens.
- Required characters retained: `ɓ`, `Ɓ`, `ɗ`, `Ɗ`, `ƴ`, `Ƴ`, `ŋ`, `Ŋ`, and
  `ʼ`.
- Doubled-vowel coverage retained: `aa`, `ee`, `ii`, `oo`, and `uu`.
- Unknown characters during sample preparation: 0.
- Actual Coqui encode/decode round trips checked: 60.
- Coqui round-trip failures: 0.
- Unknown characters before training: 0.
- Unknown characters during held-out inference: 0.
- Source audio and manifest hashes were unchanged after preparation.

The sample also exercised `é`, `ü`, guillemets, case variants, and the complete
character inventory available in eligible training transcripts.

## Optimization and reload checks

- Completed steps: 50.
- Total training/evaluation wall time: 41.526 seconds.
- Mean wall time: 0.831 seconds per step including validation overhead.
- Initial generator aggregate loss: 238.759.
- Fifty-step running-average generator aggregate loss: 114.027.
- Validation generator aggregate loss: 77.870.
- Fifty-step running-average discriminator loss: 3.017.
- Validation discriminator loss: 2.809.
- All recorded losses finite: pass.
- Strict checkpoint reload: pass.
- Generated waveform: valid 24 kHz WAV, 8.128 seconds.
- End-to-end gate status: pass.

Loss values are useful only as finite-optimization diagnostics at this budget;
they are not comparable with a converged model or a different architecture.

## Decision

No Fulfulde character-handling blocker was found. The selected multi-speaker,
character-level VITS direction remains unchanged. A full GPU run should still
wait for the full audio-quality scan and the planned GPU smoke run; this CPU
gate does not replace either requirement.

Machine-readable preparation and training results are generated under
`outputs/cpu_sanity/` and excluded from Git.

