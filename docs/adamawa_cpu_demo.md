# Adamawa Fulfulde CPU demo

The completed Adamawa Fulfulde VITS model can synthesize on CPU. An NVIDIA GPU
is not required for inference. The demo supports either one-shot WAV generation
or a localhost browser interface and does not transmit text or audio elsewhere.

## Required files

Keep these two files together in durable storage before terminating EC2:

- `config.json`
- `checkpoint_53000.pth`

They are currently in:

```text
/home/ubuntu/fulfulde-tts/checkpoints/adamawa-full/
fub_adamawa_full-August-03-2026_12+59PM-e650b45/
```

Also preserve `metrics.json`, `validation.json`, and `sample_0.wav` through
`sample_2.wav` as experiment evidence. Preserve the repository itself because
it contains the custom conservative Fulfulde cleaner and demo script.

## Test on the current machine without using the GPU

From the repository root:

```bash
MPLCONFIGDIR=/tmp/fub-demo-matplotlib \
  .venv/bin/python scripts/demo_adamawa_cpu.py \
  --threads 4 \
  --text 'Hmm booɗɗum.' \
  --output demo_output.wav
```

The verified EC2 CPU test loaded the model in 1.60 seconds and synthesized a
1.888-second waveform in 0.652 seconds, a real-time factor of 0.345. CPU speed
will vary by machine.

## Run the private browser demo

```bash
MPLCONFIGDIR=/tmp/fub-demo-matplotlib \
  .venv/bin/python scripts/demo_adamawa_cpu.py \
  --threads 4 \
  --host 127.0.0.1 \
  --port 7860
```

Open `http://127.0.0.1:7860` when running locally. When running on a remote
machine, forward port 7860 over SSH or VS Code. Keep the server bound to
`127.0.0.1`; no public firewall rule is needed.

For a checkpoint stored elsewhere, provide explicit paths:

```bash
.venv/bin/python scripts/demo_adamawa_cpu.py \
  --config /path/to/config.json \
  --checkpoint /path/to/checkpoint_53000.pth
```

The server loads the model once, serializes concurrent inference requests, caps
input at 300 characters, and rejects unsupported characters rather than
silently discarding them.

## Move to another computer

Clone or copy this repository, create the Python environment with the
`cpu-sanity` dependency group, and copy the saved model files to the new
machine. For example, with `uv` installed:

```bash
uv sync --group cpu-sanity
```

Then start the demo with the explicit `--config` and `--checkpoint` paths shown
above. A CPU-only PyTorch installation can reduce the environment size, but it
should be validated separately before deleting the known-working environment.

## Static presentation fallback

For a zero-compute presentation, place `sample_0.wav`, `sample_1.wav`, and
`sample_2.wav` beside slides or a simple static webpage and label each with its
text from `validation.json`. This demonstrates fixed, authentic model outputs;
it does not accept new text.

## Before terminating EC2

1. Copy the repository and required model/result files to a laptop, S3, or an
   attached EBS volume that will survive termination.
2. Recompute the final checkpoint SHA-256 and confirm it equals
   `7dfd08b18b968f41ba4a5a09e7d61c197817647b7747aaf013ad18610b2940a8`.
3. Run one CPU synthesis on the destination machine.
4. Confirm the EC2 root volume's `DeleteOnTermination` setting.
5. Only then terminate the instance.

The source dataset is NOODL-1.0 and is not redistributed here. Release of the
derived checkpoint for the research demo relies on separate permission agreed
by email and confirmed by the user on 2026-08-07. The private correspondence is
retained outside the repository; see `model/README.md` for the public model-use
notice and artifact lineage.
