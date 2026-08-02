# Adamawa Fulfulde (`fub`) TTS

This repository contains the auditable data-preparation foundation for an
Adamawa Fulfulde TTS research baseline. Source speech corpora remain unchanged
and are excluded from Git.

## Current corpus layout

Place the extracted Common Voice language directory at:

```text
cv-corpus-26.0-2026-06-12/fub/
  clips/
  validated.tsv
  clip_durations.tsv
  README.md
```

The corpus is Cameroon Adamawa Fulfulde (`fub`), with Ngaoundéré prompts. It is
not Nigerian Fulfulde (`fuv`).

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

## Tests

The initial audit pipeline has no runtime dependencies:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Audio decoding, clipping, silence, sample-rate, and noise measurements require
a later full audio scan. The current report labels those measurements as not
run; it does not silently treat present MP3 files as decoded or clean.

