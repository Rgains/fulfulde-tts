# Adamawa-Fulfulde-TTS-Dataset audit and CPU results

## Outcome

The separate dataset passed a full mapped-audio audit and the bounded 50-step
VITS CPU gate. It is suitable for a controlled GPU smoke run on its clean
subset. It is not yet ready for an unattended full GPU training run because
the supplied mapping has no speaker metadata and several transcript/audio
review flags remain.

This dataset was not merged with Mozilla Common Voice.

## Licence record

The source dataset is distributed under NOODL-1.0. The user confirmed on
2026-08-02 that local research training was authorized and, on 2026-08-07,
confirmed that a separate email agreement authorizes release of the derived
checkpoint for this research demo. The private correspondence is retained
outside the repository. This permission does not authorize redistribution of
the source speech dataset.

## Full audit

- Mapping rows: 1,302.
- Physical MP3 files: 1,303.
- Mapped MP3s decoded successfully: 1,302 of 1,302.
- Missing mapped audio: 0.
- Unreferenced audio: 1.
- Duplicate audio hashes: 0.
- Total mapped audio: 3.542 hours.
- Source format: mono 48 kHz MP3 throughout.
- Median clip duration: 6.510 seconds.
- Clips over the 15-second VITS gate: 171.
- Duration-gated subset: 1,131 clips / 2.043 hours.
- Clips with at least 60% near-silence: 43.
- Clips with at least 0.1% near-full-scale samples: 0.
- Transcripts with repeated apostrophes: 64 total; 33 in the duration-gated
  subset.
- Transcripts with typographic punctuation: 336 total; 230 in the
  duration-gated subset.
- Source mapping hash before and after audit: unchanged.

The conservative smoke-ready pool contains 1,055 duration-gated clips after
excluding high-silence and repeated-apostrophe records. Typographic `’` is
preserved and tested rather than silently replaced.

## Leakage-safe manifests

| Split | Clips | Hours |
|---|---:|---:|
| Train | 904 | 1.637 |
| Validation | 118 | 0.212 |
| Test | 109 | 0.194 |

- Normalized-text leakage: 0.
- Audio-path leakage: 0.
- Complete accepted character inventory represented in training: pass.
- Speaker-disjoint evaluation: unavailable because speaker metadata is absent.

The initial deterministic split placed the corpus's only uppercase `Ƴ` in the
test set. The audit caught this and deterministically moved that complete text
group into training. No held-out character is now absent from training.

## Fifty-step CPU gate

- Sample: 60 clips (50 train, 5 validation, 5 test).
- Derived duration: 315.660 seconds.
- Derived audio: mono 24 kHz PCM WAV.
- Direct grapheme vocabulary: 61 entries including special tokens.
- Required dataset characters exercised: `ɓ`, `Ɓ`, `ɗ`, `Ɗ`, `ƴ`, `Ƴ`, `ŋ`,
  and `’`.
- Doubled vowels exercised: `aa`, `ee`, `ii`, `oo`, and `uu`.
- Actual Coqui encode/decode checks: 60.
- Tokenizer round-trip failures: 0.
- Unknown characters before training: 0.
- Unknown characters during inference: 0.
- Steps completed: exactly 50.
- All recorded losses finite: pass.
- Training and evaluation wall time: 42.473 seconds.
- Mean wall time: 0.849 seconds per step.
- Initial generator aggregate loss: 260.187.
- Fifty-step running-average generator aggregate loss: 97.436.
- Validation generator aggregate loss: 67.937.
- Strict checkpoint reload: pass.
- Generated waveform: valid mono 24 kHz WAV, 5.611 seconds.
- Source audio and manifest hashes after preparation: unchanged.

Four selected source decodes had peaks slightly above one after MP3 decoding.
Only their derived 24 kHz WAVs were clamped to the valid PCM range; original
MP3 files were not modified.

## GPU decision

The orthography, data loading, optimization, checkpoint, and inference gates
are clear for a bounded GPU smoke run using the clean subset. Before a full
training run:

1. attach the exact licence evidence and provenance record;
2. determine whether the directory represents one or multiple speakers,
   without attempting unauthorized identification;
3. review or exclude repeated-apostrophe transcripts;
4. review the 43 high-silence clips; and
5. retain a source-labelled evaluation separate from Common Voice.

The 50-step checkpoint is diagnostic and says nothing about naturalness or
speaker consistency.
