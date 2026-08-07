# Adamawa Fulfulde (`fub`) TTS

This repository contains the auditable data-preparation foundation for an
Adamawa Fulfulde TTS research baseline. Source speech corpora remain unchanged
and are excluded from Git.

When continuing this work in a new Codex session or on EC2, read the portable
[session handoff](sessions.md) before taking action.

## Current corpus layout

Place the extracted Common Voice language directory at:

```text
cv-corpus-26.0-2026-06-12/fub/
  clips/
  validated.tsv
  clip_durations.tsv
  README.md
```

The corpus is Adamawa Fulfulde (`fub`) recorded in Cameroon, with Ngaoundéré
prompts. `fub` is also spoken in northeastern Nigeria, but these recordings are
Cameroonian, and `fub` is not Nigerian Fulfulde (`fuv`).

## Prepare the Common Voice data

The preparation command performs a metadata integrity audit and creates
deterministic, sentence-disjoint manifests from all suitable validated clips:

```bash
PYTHONPATH=src python3 -m fub_tts prepare \
  --dataset-dir cv-corpus-26.0-2026-06-12/fub \
  --report-dir reports/generated/common_voice_fub_v26_0 \
  --manifest-dir data/manifests/common_voice_fub_v26_0
```

It never modifies or converts the source MP3 files. Clips over 15 seconds,
missing files, missing durations, unexpected locale values, and empty text are
written to `rejected.tsv` with explicit reasons.

The generated primary split uses all suitable validated clips and groups by
normalized sentence, preventing repeated Common Voice prompts from crossing
train, validation, and test. The official Common Voice train/dev/test files
remain useful as a small, separately defined benchmark.

Add `--full-audio-scan` to also decode every validated clip and measure real
sample rate, clipping, and silence (slow -- several thousand files). Without
it, the audit reports file presence only and `audio_quality.status` stays
`"not_run"`. With it, `speaker_analysis.ranking` gains a `usable_clips` /
`usable_duration_hours` per speaker, which can rank differently from raw
duration once near-silent or clipped clips are excluded -- check
`usable_duration_hours`, not `duration_hours`, before picking a target
speaker for adaptation.

## Tests

The initial audit pipeline has no runtime dependencies:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Audio decoding, clipping, silence, sample-rate, and noise measurements require
a later full audio scan. The current report labels those measurements as not
run; it does not silently treat present MP3 files as decoded or clean.

## Cheap CPU gate before GPU training

The [50-step CPU sanity protocol](docs/cpu_sanity_protocol.md) adapts the
earlier Tiv gate to direct Adamawa Fulfulde characters. It is a bounded VITS
data-flow and orthography check, not an architecture comparison or speech-
quality evaluation.

The completed run and its limitations are recorded in the
[CPU sanity results](docs/cpu_sanity_results.md).

## Separate Adamawa-Fulfulde-TTS-Dataset

The non-Common-Voice corpus follows its own
[audit and lineage protocol](docs/adamawa_dataset_protocol.md). It retains a
separate source label and manifests; it is never silently merged with Common
Voice.

Its completed [audit and CPU sanity results](docs/adamawa_dataset_results.md)
record the usable duration, character-coverage correction, remaining review
flags, and bounded GPU-smoke recommendation.

The completed full-corpus model can run without an NVIDIA GPU. See the
[CPU live-demo guide](docs/adamawa_cpu_demo.md) for one-shot synthesis, a
localhost browser interface, artifact backup, and the EC2 termination checklist.
