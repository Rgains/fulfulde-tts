# A Character-Level VITS Research Baseline for Adamawa Fulfulde

## Abstract

This report describes an auditable text-to-speech (TTS) research baseline for
Adamawa Fulfulde (`fub`) trained from the locally supplied
`Adamawa-Fulfulde-TTS-Dataset`. The workflow preserves Fulfulde orthography,
keeps source recordings unchanged, filters unsuitable recordings using explicit
audio and transcript criteria, and prevents normalized transcripts from crossing
the training, validation, and test partitions. A bounded GPU smoke experiment
was completed before full-corpus training. The dataset mapping does not contain
speaker labels, so the model is trained without speaker conditioning and this
work makes no claim that the recordings represent one speaker. Native-speaker
evaluation remains required before claims about intelligibility, pronunciation,
speaker consistency, or naturalness can be made.

## 1. Scope and research question

The practical question is whether a direct-grapheme VITS pipeline can ingest a
small, licensed Adamawa Fulfulde corpus without losing language-specific Unicode
characters, train stably on an NVIDIA L4, reload its checkpoint strictly, and
generate valid waveforms for held-out text. This is a data-flow and baseline
experiment, not evidence of production readiness.

Adamawa Fulfulde (`fub`) is kept distinct from Nigerian Fulfulde (`fuv`). The
pipeline does not transliterate `ɓ`, `ɗ`, `ƴ`, or `ŋ`, does not collapse doubled
vowels, and preserves the typographic apostrophe `’` where it occurs in accepted
text.

## 2. Dataset and authorization

The source snapshot contains 1,302 mapped MP3 recordings and one additional
unreferenced MP3. All 1,302 mapped files decoded successfully. Their combined
duration is 3.542 hours; the source format is mono 48 kHz MP3. The dataset is
distributed under NOODL-1.0. The user confirmed authorization for local
research training on 2026-08-02 and confirmed on 2026-08-07 that a separate
email agreement authorizes release of the derived checkpoint for this research
demo. The private correspondence remains outside the repository, and the
source speech dataset is not redistributed.

`Mapping_MP3.tsv` provides filenames and transcripts but no speaker field.
Consequently, the experiment uses an unconditioned VITS model and labels speaker
coverage as unknown. It does not infer speaker identity or claim a verified
single-speaker corpus.

## 3. Data preparation

Text normalization is deliberately conservative: Unicode NFC normalization [3]
and whitespace collapse only. Raw and normalized text are retained separately.
Recordings outside 1–15 seconds, recordings with at least 60% samples below the
-50 dBFS silence threshold, and transcripts flagged for repeated apostrophes are
excluded from the conservative training pool. Typographic punctuation is
preserved rather than silently rewritten.

Accepted audio is decoded, converted to mono, resampled from 48 kHz to 24 kHz,
and written as derived PCM WAV. Samples exceeding a peak magnitude of 0.98 after
decode/resampling are scaled only in the derived file; source MP3s remain
unchanged. A shared derived pool is used by smoke and full experiments to avoid
duplicating audio.

The clean pool contains 1,055 clips and 6,951.12 seconds (1.931 hours):

| Partition | Clips | Purpose |
|---|---:|---|
| Training | 844 | Parameter optimization |
| Validation | 112 | Per-epoch held-out loss monitoring |
| Test | 99 | Final held-out synthesis inputs |

Partitions are deterministic and grouped by normalized transcript. There is
zero normalized-text leakage and zero audio-path leakage. Every character in
accepted held-out text is represented in training.

## 4. Model and training configuration

The model is character-level VITS, following the end-to-end variational and
adversarial architecture introduced by Kim, Kong, and Son [1], implemented with
Coqui TTS 0.27.5 [2] and PyTorch 2.8.0. It has 83,045,932 parameters in the
full-corpus configuration. Audio is
represented at 24 kHz with a 1,024-sample FFT and window, 256-sample hop, 80 mel
channels, and an 8 kHz mel upper bound. Training uses direct characters rather
than phonemes.

The full experiment uses BF16 mixed precision, batch size 16, two training data
workers, two validation workers, seed 20260802, cuDNN benchmarking, and 1,000
epochs on a single NVIDIA L4. The progress display labels batches `0/52`, while
the global counter advances by 53 steps per epoch; the resulting target is
approximately 53,000 steps in total. Rolling
checkpoints are requested every 5,000 steps, with one rolling checkpoint retained.
Training runs inside detached `tmux` so an SSH or VS Code reconnect does not stop
the process.

## 5. Preliminary GPU smoke experiment

Before full training, a 266-clip subset (226 train, 20 validation, 20 test) was
trained for 375 optimizer steps. BF16 training completed with finite reported
losses. The final checkpoint reloaded with strict parameter matching, tokenizer
audits found zero unknown characters before training and at inference, and three
valid mono 24 kHz WAV files were generated. Peak allocated and reserved GPU
memory were 20,081 MiB and 20,876 MiB respectively. Trainer initialization and
fitting took 442.87 seconds; the original smoke metrics did not separately time
the subsequent final save, strict reload, and synthesis. The reusable trainer
has since been corrected to report training and end-to-end wall time separately.
The smoke result establishes pipeline execution, not speech quality.

## 6. Full-corpus results

Training completed all 1,000 configured epochs and exactly 53,000 optimizer
steps. The trainer-reported elapsed time was 24,879.91 seconds (6 h 54 min
39.91 s), or 0.469 seconds per step when initialization, per-epoch validation,
and checkpoint pauses are included. From process launch at approximately
12:59:38 UTC to the completed validation record at 19:54:51 UTC, the end-to-end
workflow took approximately 6 h 55 min 13 s.

The final held-out evaluation reported generator aggregate loss 35.9234, mel
loss 23.8307, KL loss 3.0755, duration loss 2.6026, feature loss 4.4774, and
discriminator loss 2.5840. These optimization diagnostics are not perceptual
quality scores and should not be interpreted as intelligibility or naturalness
measurements.

The 997,762,755-byte final checkpoint reloaded with strict parameter matching.
Its SHA-256 digest is
`7dfd08b18b968f41ba4a5a09e7d61c197817647b7747aaf013ad18610b2940a8`.
Tokenizer audits found zero unknown characters both before training and during
final inference. Three held-out prompts produced valid mono, 24 kHz, 16-bit PCM
WAV files lasting 2.027, 8.203, and 5.611 seconds. Automated post-run validation
passed, and all nine repository unit tests passed.

The source `Mapping_MP3.tsv` digest after training matched the expected digest,
`9d43f2616f656bd3d46178381dc83dbd3fddd1dbf39e63833a6ab2d8db73a3b5`,
confirming that the source mapping was not modified.

## 7. Resource utilization and engineering assessment

The full-corpus first epoch exposed the expected cuDNN benchmarking cost across
varied sequence lengths: its 52 training steps averaged approximately 12.8
seconds per step and were followed by a successful validation pass. Once kernel
choices were cached, epoch 1 training stabilized near 0.34 seconds per step,
about 38 times faster. This distinction matters: extrapolating from epoch 0
would substantially overstate total runtime. After warm-up, training steps were
typically near 0.36 seconds, while complete epochs also included validation and
occasional checkpoint writes. The measured trainer runtime was 6 h 54 min 40 s,
and the complete launch-to-validation workflow was approximately 6 h 55 min.

Batch size 16 reached 20,239.51 MiB peak allocated and 20,678 MiB peak reserved
GPU memory during the full run on a 23,034 MiB NVIDIA L4. This uses the
accelerator effectively while retaining limited safety headroom. The full corpus has nearly the same maximum and
95th-percentile clip durations as the smoke subset, reducing the risk that the
larger manifest introduces a new peak-memory shape. Two loader workers and two
validation workers are proportionate to the instance's eight CPUs. BF16 lowers
activation memory and improves tensor-core utilization. Checkpoint retention is
bounded to one rolling checkpoint, although Coqui also maintains its best-model
artifact separately. At global step 5,000, the run held `best_model.pth`, its
retained predecessor `best_model_3922.pth`, and `checkpoint_5000.pth`. Each file
was 997,762,755 bytes, for about 3.0 GB of model state in total. The root volume
still had 2.9 GB free immediately after the rolling checkpoint was written.
This confirms that retention is operating within the recovered headroom, while
continued monitoring remains necessary because best-model replacement can
briefly require another temporary write.

Before the first rolling checkpoint, root-volume headroom fell to approximately
2.0 GB. With explicit user approval, two superseded smoke-run optimizer
checkpoints (one Adamawa and one Common Voice) were deleted after their smoke
metrics, logs, configurations, and generated WAV samples had already been
preserved. This recovered approximately 1.9 GB and raised free space to 3.9 GB.
The operation did not remove any full-corpus artifact or modify either source
dataset.

The EC2 host also exposes a blank, unmounted 419 GB NVMe device
(`/dev/nvme1n1`). Read-only inspection found no filesystem or partition
signature. Formatting and mounting that device would provide a durable
experiment volume and remove the root-volume checkpoint risk, but this is a
destructive infrastructure action and therefore requires explicit user
authorization. Until authorized, the device is left untouched.

## 8. Limitations

- Speaker metadata is absent, preventing speaker-disjoint evaluation and
  speaker-consistency claims.
- Native-speaker listening evaluation has not yet been performed.
- The NOODL-1.0 source licence is recorded. Model release relies on separate
  email permission confirmed by the user on 2026-08-07; that private
  correspondence is not stored in the repository.
- The corpus is small for training a modern TTS model from scratch.
- Automatic loss values, successful synthesis, and valid WAV structure do not
  prove intelligibility or naturalness.
- The held-out split is transcript-disjoint but cannot be speaker-disjoint.

## 9. Reproducibility and artifacts

The preparation and trainer entry points are
`scripts/prepare_adamawa_gpu_data.py` and
`scripts/train_vits_smoke_multispeaker.py`. Generated manifests live under
`data/derived/adamawa_training/`; model artifacts are outside Git under
`/home/ubuntu/fulfulde-tts/checkpoints/adamawa-full/`. The full console log is
`/home/ubuntu/fulfulde-tts/logs/adamawa_full_training.stdout.log`.

The machine-readable result files are `metrics.json` and `validation.json` in
the timestamped run directory. The latter records the final checkpoint digest,
sample digests and audio properties, manifest summary digest, source mapping
integrity check, strict-reload result, and tokenizer coverage. All numerical
claims above were derived from saved manifests, trainer logs, metrics JSON,
audio headers, and cryptographic hashes rather than conversational memory.

## References

1. J. Kim, J. Kong, and J. Son, “Conditional Variational Autoencoder with
   Adversarial Learning for End-to-End Text-to-Speech,” *Proceedings of the 38th
   International Conference on Machine Learning*, PMLR 139, 2021.
   <https://proceedings.mlr.press/v139/kim21f.html>
2. Coqui AI, “TTS: a deep learning toolkit for text-to-speech.”
   <https://github.com/coqui-ai/TTS>
3. Unicode Consortium, “Unicode Standard Annex #15: Unicode Normalization
   Forms.” <https://unicode.org/reports/tr15/>
