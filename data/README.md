# Data directories

- `raw/`: immutable source archives or extracted corpora; never commit.
- `interim/`: decoded or intermediate data; never commit audio.
- `processed/`: filtered training-ready audio; never commit audio.
- `manifests/`: reproducible generated metadata and split manifests.

Every generated manifest must retain the source dataset, dataset version,
language code, licence, anonymous speaker identifier, raw text, normalized
text, and split assignment.

