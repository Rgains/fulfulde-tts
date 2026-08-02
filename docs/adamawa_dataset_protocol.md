# Separate Adamawa-Fulfulde-TTS-Dataset protocol

This corpus remains source-labelled and separate from Mozilla Common Voice.
It must not be merged into Common Voice manifests without an explicit combined
experiment and source-aware evaluation.

The user confirmed that a licence exists and authorized local research
training on 2026-08-02. The exact licence identifier and evidence file are not
yet present in this workspace, so redistribution and model-release rights are
not inferred.

Run the full mapped text/audio audit and deterministic split with:

```bash
uv run --group cpu-sanity env PYTHONPATH=src \
  python scripts/audit_adamawa_dataset.py \
  --config configs/adamawa_dataset.json
```

The audit decodes every mapped MP3 and reports duration, sample rate, channel
count, peak, RMS, near-full-scale ratio, near-silence ratio, file hashes,
mapping integrity, character inventory, text review flags, and rejection
reasons. The initial training gate accepts decoded clips between 1 and 15
seconds. Audio quality remains available in the record-level report rather
than being silently discarded.

The deterministic splitter keeps normalized-text groups intact and, when a
rare character would otherwise appear only in validation or test, moves that
whole text group into training. Every character-level model therefore receives
the complete accepted corpus inventory without row-level leakage.

`Mapping_MP3.tsv` has no speaker field. Treating the directory as one source
group is not evidence that it contains one speaker.
