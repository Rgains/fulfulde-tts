# Building a Text-to-Speech System for Adamawa Fulfulde: Two Corpora, One Architecture, and an Open Speaker Question

**Status:** research baseline, not production. No native Fulfulde speaker
has evaluated the output yet. **The corpus used here is Cameroon Adamawa
Fulfulde (`fub`), not Nigerian Fulfulde (`fuv`)** — see Section 7 before
assuming this serves a Nigeria-facing use case.

## Abstract

We build a character-level VITS text-to-speech system for Adamawa Fulfulde
from two independently sourced, separately licensed corpora: Mozilla
Common Voice 26.0 (`fub`, CC0, multi-speaker, ASR-oriented) and a private
corpus, `Adamawa-Fulfulde-TTS-Dataset` (single source group, speaker
identity unverified). We reuse the architecture decision (VITS over
Matcha-TTS) established for a sibling Tiv-language project rather than
re-litigating it, and instead focus engineering effort on staged
GPU validation and a genuine multi-speaker conditioning bug found and
fixed during that validation. The `Adamawa-Fulfulde-TTS-Dataset` track
reached full training (53,000 steps, 6h55m on one NVIDIA L4, all losses
finite) and a working CPU-only inference demo. The Common Voice track
completed a real multi-speaker smoke gate (5 speakers, verified speaker
conditioning) but has not yet been scaled to full training. We report
exact figures for both tracks and are explicit about what remains before
either is usable: native-speaker evaluation, and a still-unresolved
question about whether this project should target `fub` or `fuv` at all.

## 1. Introduction

This project began as an extension of a Tiv-language TTS effort built for
a proposed Nigeria-facing, NEMA-affiliated early-warning advisory system.
Fulfulde was a natural next target given its speaker population spans
Nigeria, Cameroon, and neighboring states. The corpus that was actually
available and audited, however, is **Cameroon Adamawa Fulfulde (`fub`)**,
with Ngaoundéré prompts — a distinct variety from **Nigerian Fulfulde
(`fuv`)**, which is what a Nigeria-facing system would presumably need.
This project's own operating rules (`AGENTS.md`) require never merging or
relabeling the two. We treat this as a research-baseline exercise in
building the pipeline correctly on available data, not a claim that the
result serves the original Nigeria-facing motivation — that requires
either a `fuv` corpus or an explicit decision that `fub` output is
acceptable for the intended audience, neither of which has happened.

Two corpora were used, kept source-labelled and never merged, per
`context.md`:

1. **Mozilla Common Voice 26.0, `fub`** — CC0, multi-speaker, ASR-oriented.
2. **`Adamawa-Fulfulde-TTS-Dataset`** — a separate, privately supplied
   corpus with no speaker field in its metadata.

## 2. Datasets

### 2.1 Mozilla Common Voice (`fub`)

| Field | Value |
|---|---|
| Source | Mozilla Common Voice 26.0, `cv-corpus-26.0-2026-06-12/fub` |
| Licence | CC0-1.0 |
| Variety | Cameroon Adamawa Fulfulde, Ngaoundéré prompts |
| Physical MP3 files | 7,924 |
| Validated rows | 7,686 (30 invalidated, 208 other/unresolved) |
| Validated duration (metadata) | 13.123 hours |
| Anonymous speakers (validated) | 14 (19 across all buckets) |
| Unique validated sentences | 1,067 |

A **full decoded-audio scan** (every validated clip actually decoded and
measured, not just metadata-checked) found:

- **7,686/7,686 clips decoded successfully — zero failures.**
- Uniform format: 32 kHz, mono, throughout.
- **7,354 usable clips (95.7%), 12.559 of 13.123 hours usable** after a
  duration (1–15 s), clipping (<0.1%), and silence (<60% near-silent
  samples) gate.
- Clipping was negligible corpus-wide (median 0%, max 0.068%). Silence
  was the binding constraint (331 clips at or above the 60% threshold).

This scan surfaced a finding relevant to any future speaker selection:
**ranking by usable duration reorders the top speakers relative to raw
duration.** The speaker with the most raw validated duration
(`cv_833357698662230d`, 2.031 h) drops to 1.922 h usable after quality
filtering, while the speaker ranked second by raw duration
(`cv_7a992eabfeff7576`, 2.019 h raw) is barely affected (2.018 h usable)
and becomes the strongest candidate once quality is accounted for. Deterministic
sentence-disjoint manifests were built from all suitable validated clips
(not just the official train/dev/test split, which uses only ~1,067
clips): 6,257 train / 691 validation / 737 test, zero sentence or
audio-path leakage.

### 2.2 `Adamawa-Fulfulde-TTS-Dataset`

| Field | Value |
|---|---|
| Mapping rows | 1,302 (`Mapping_MP3.tsv`, no speaker field) |
| Physical MP3 files | 1,303 (1 unreferenced) |
| Decoded successfully | 1,302 / 1,302 |
| Total mapped duration | 3.542 hours |
| Source format | mono 48 kHz MP3, throughout |
| Licence status | User-confirmed and authorized for local research training on 2026-08-02; **exact licence identifier and evidence file not yet recorded in the repository** |

A full mapped-audio-and-text audit (`scripts/audit_adamawa_dataset.py`)
applied a duration gate (1–15 s), a silence gate (<60%), and excluded
transcripts flagged for repeated apostrophes, yielding a **clean pool of
1,055 clips (6,951.12 s / 1.931 h)**, split 844 train / 112 validation /
99 test with zero normalized-text leakage and zero audio-path leakage.
The splitter's leakage-safe design caught a real edge case: the corpus's
only occurrence of uppercase `Ƴ` initially landed in the test split; the
splitter deterministically moved that whole text group into training so
no accepted character is ever absent from the training vocabulary.

**Speaker identity is explicitly not claimed.** With no speaker field in
the source mapping, this dataset is treated as one unverified source
group, and all experiments on it are unconditioned (no speaker
embedding) — a materially different setup from the Common Voice track.

## 3. Orthography and text handling

Both pipelines apply identical, deliberately conservative rules
(`src/fub_tts/text.py`): Unicode NFC normalization and whitespace
collapse only — no transliteration, no phoneme conversion. Both datasets'
audits verified retention of `ɓ Ɓ ɗ Ɗ ƴ Ƴ ŋ Ŋ ʼ`, doubled vowels
(`aa ee ii oo uu`), and the typographic apostrophe `’`, and confirmed zero
unknown-character tokenization failures at every training and inference
step across every experiment reported below.

## 4. Architecture: an inherited, not re-litigated, decision

This project does not run its own VITS-vs-alternatives comparison.
`AGENTS.md` and `docs/cpu_sanity_protocol.md` explicitly carry over the
architecture decision from the sibling Tiv-TTS project's controlled
bake-off (VITS selected over Matcha-TTS for its integrated waveform
decoder and simpler licensing surface) rather than re-opening it. This is
a deliberate scope decision worth stating plainly: **the claim "VITS
works for this data" is inherited, not independently re-validated for
Fulfulde** beyond the orthography- and pipeline-level checks in Section 5.
F5-TTS and MMS-TTS remain listed as future research-only comparisons
(Section 9), not selected alternatives.

## 5. Staged validation

Following the same discipline as the Tiv project — cheap gates before
expensive ones — each dataset passed, independently:

**5.1 50-step CPU sanity gates** (`docs/cpu_sanity_results.md`,
`docs/adamawa_dataset_results.md`): 60 clips each (50/5/5), direct
characters, no pretrained weights, batch size 1, CPU. Common Voice: 69-symbol
vocabulary, 0.831 s/step, all losses finite, strict reload passed. Adamawa
dataset: 61-symbol vocabulary, 0.849 s/step, all losses finite, strict
reload passed. Both are pipeline/orthography checks only — fifty updates
from random initialization establish nothing about voice quality.

**5.2 CUDA environment verification** (`reports/generated/environment/`):
NVIDIA L4, CUDA 12.8 (torch build), BF16 confirmed stable via an explicit
matmul probe before it was relied on for any real training.

**5.3 GPU smoke gates**, 375 steps each, both using
`scripts/train_vits_smoke_multispeaker.py`:

| | Common Voice | Adamawa dataset |
|---|---|---|
| Clips | 265 (225/20/20) | 266 (226/20/20) |
| Speakers | 5, real conditioning | 1 unverified group, unconditioned |
| Precision | BF16 | BF16 |
| Steps | 375 | 375 |
| s/step | 1.215 | 1.181 |
| Peak VRAM (allocated/reserved) | 20.07 / 20.75 GiB | 20.08 / 20.88 GiB |
| Unknown chars (train/inference) | 0 / 0 | 0 / 0 |
| Checkpoint reload | strict, pass | strict, pass |

### 5.4 A genuine bug, found and fixed

The first attempt at the Common Voice multi-speaker smoke gate silently
trained **without any speaker conditioning at all**, despite passing
`use_speaker_embedding=True` to `VitsConfig`. The root cause:
`TTS.config.get_from_config_or_model_args_with_default()` checks
`config.model_args` first whenever the key exists there, and
`VitsArgs.use_speaker_embedding` defaults to `False` independently of the
same-named top-level `VitsConfig` field — so the top-level flag was
silently ignored. This was only caught because the post-training
inference step required a populated `speaker_manager` and raised
`AttributeError` when it wasn't there, which prompted tracing why. The
fix is to pass `model_args=VitsArgs(use_speaker_embedding=True)`
explicitly. The corrected run (Section 5.3's Common Voice figures) shows
`num_speakers: 5` with real named anonymous speaker IDs attached to each
generated sample, confirming conditioning is genuinely active this
time — a distinction worth recording because a plausible-looking but
silently-wrong config is a realistic failure mode with this library, not
a hypothetical one.

## 6. Full training: `Adamawa-Fulfulde-TTS-Dataset`

Only the `Adamawa-Fulfulde-TTS-Dataset` track was scaled to full
training; Common Voice remains at the smoke-gate stage (Section 9).

| | |
|---|---|
| Hardware | NVIDIA L4, AWS `g6.2xlarge` |
| Steps | 53,000 (1,000 epochs) |
| Batch size | 16, BF16 |
| Wall-clock | 6h54m40s trainer time (24,879.9 s); ~6h55m end-to-end (12:59:38–19:54:51 UTC, 2026-08-03) |
| Sustained rate | 0.469 s/step including validation and checkpoint pauses |
| Parameters | 83,045,932 |
| Peak VRAM | 20,239.5 MiB allocated / 20,678 MiB reserved (of 23,034 MiB) |

**Convergence.** Mel loss fell from 93.73 (step 1) to a final validation
value of **23.83** (aggregate generator loss 35.92, KL 3.08, duration
2.60, feature 4.48, discriminator 2.58). All losses remained finite for
the entire run.

**A repeated lesson on calibration.** As with the earlier Tiv project,
epoch-0 step timing (~12.8 s/step, dominated by cuDNN kernel-selection
overhead across varied sequence lengths) was roughly 38× slower than the
steady-state rate after warm-up (~0.34–0.36 s/step). Extrapolating total
runtime from the first few steps of any of these runs would have
overstated it substantially — worth treating as a standing caveat for
this codebase rather than a one-off surprise.

**Verification.** The final checkpoint (`checkpoint_53000.pth`,
997,762,755 bytes, SHA-256 `7dfd08b1...`) reloaded with strict parameter
matching. Three held-out test sentences were synthesized and hashed
(`validation.json`); the source `Mapping_MP3.tsv` hash was confirmed
unchanged after training. Zero unknown characters at inference.

**Operational note.** Root-volume headroom fell to ~2.0 GB before the
first rolling checkpoint of this run. With explicit user approval, two
superseded smoke-run checkpoints (one per dataset) were deleted after
their metrics, logs, configs, and generated WAV samples were already
preserved elsewhere, recovering ~1.9 GB. A 419 GB unmounted NVMe device
was found on the host and left untouched pending explicit authorization
to format and mount it as a durable experiment volume — a live risk for
any future full run on this box that isn't yet resolved.

## 7. The `fub` vs. `fuv` question

This is the most consequential open item and is stated once more here
plainly: everything in this report trains and evaluates **Cameroon
Adamawa Fulfulde**. If the eventual deployment target is Nigerian
communities and Nigerian Fulfulde speakers, `fuv` is a linguistically
distinct variety and this model has not been shown to serve that
audience. Resolving this requires either sourcing a `fuv` corpus or an
explicit, informed decision from whoever owns the deployment goal that
`fub` output is an acceptable stand-in — a decision this report does not
make on its own.

## 8. CPU inference deployment

A CPU-only inference demo (`scripts/demo_adamawa_cpu.py`,
`docs/adamawa_cpu_demo.md`) was built and verified against the full
`Adamawa-Fulfulde-TTS-Dataset` checkpoint: model load in 1.60 s,
synthesis of a 1.888 s waveform in 0.652 s (real-time factor 0.345, i.e.
faster than real time on CPU for this short utterance). This is useful
for low-stakes local testing or environments without a GPU; it was not
benchmarked for concurrent or production-scale serving.

## 9. Limitations

- **Variety mismatch risk** (Section 7) — the single most important
  caveat, and blocking for any Nigeria-facing use.
- **No native-speaker evaluation** for either dataset's output. Every
  number in this report is a pipeline-correctness or optimization result.
- **Speaker identity unverified** for `Adamawa-Fulfulde-TTS-Dataset`;
  training proceeded unconditioned and no speaker-disjoint evaluation is
  possible from the supplied metadata.
- **Licence evidence incomplete.** Local research training was
  user-authorized for `Adamawa-Fulfulde-TTS-Dataset`, but the exact
  licence identifier and evidence file are not yet recorded in the
  repository — redistribution and model-release rights are not inferred.
- **Common Voice has not been scaled to full training** — only the
  375-step, 5-speaker smoke gate has run. Its corpus is larger (12.56
  usable hours vs. 1.93) and multi-speaker, so a full run would be a
  materially different — and not yet attempted — experiment from
  Section 6.
- **Translation is out of scope by design**, not merely unaddressed:
  `AGENTS.md` explicitly forbids adding translation, warning generation,
  medical advice, or paraphrasing to the TTS endpoint. Whoever integrates
  this model into an advisory system must solve translation separately
  (see the sibling Tiv project's report for why general-purpose MT was
  found unsuitable for this class of problem, and why a bounded,
  human-translated template set is recommended instead).
- **Remaining review flags** in the Adamawa dataset (43 high-silence
  clips, 33 repeated-apostrophe transcripts in the duration-gated subset)
  were excluded from the clean training pool but not manually reviewed
  for correctness.

## 10. Reproducibility

- Repository rules: `AGENTS.md`, `context.md`
- Cross-session handoff record: `sessions.md`
- Common Voice: `docs/cpu_sanity_protocol.md` / `_results.md`,
  `reports/generated/common_voice_fub_v26_0/audit.md`,
  `src/fub_tts/common_voice.py` (`inspect_audio`, `is_usable`,
  `--full-audio-scan`)
- Adamawa dataset: `docs/adamawa_dataset_protocol.md` / `_results.md`,
  `docs/adamawa_full_training_report.md`, `docs/adamawa_cpu_demo.md`,
  `scripts/audit_adamawa_dataset.py`,
  `scripts/prepare_adamawa_gpu_data.py`
- Shared multi-speaker training script:
  `scripts/train_vits_smoke_multispeaker.py`
- Environment verification: `reports/generated/environment/cuda_environment_report.md`
- Full-run artifacts: `checkpoints/adamawa-full/fub_adamawa_full-August-03-2026_12+59PM-e650b45/`
  (`metrics.json`, `validation.json`, `config.json`, `checkpoint_53000.pth`,
  `sample_0.wav`–`sample_2.wav`)

## 11. Next steps

1. Resolve the `fub`/`fuv` question with whoever owns the deployment
   target (Section 7) — before investing further compute either way.
2. Native-speaker listening review of both tracks' generated samples.
3. Record the exact licence identifier and evidence file for
   `Adamawa-Fulfulde-TTS-Dataset` before any release framing.
4. If Common Voice is worth scaling to full training, do so as its own
   deliberate experiment — its multi-speaker, larger-corpus profile is
   different enough from Section 6 that its results shouldn't be assumed.
5. Resolve the unmounted 419 GB NVMe device question (Section 6) before
   another full run risks root-volume exhaustion.
6. Decide the translation approach with the advisory-system owner
   (Section 9) — a solved problem in neither this nor the Tiv project.
