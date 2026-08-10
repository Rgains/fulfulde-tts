#!/usr/bin/env python3
"""Streamlit demo: type Adamawa Fulfulde text, hear it synthesized by the
full-corpus VITS model trained on Adamawa-Fulfulde-TTS-Dataset.

Run with: streamlit run scripts/demo_streamlit_adamawa.py
Unconditioned model (no speaker field in the source corpus) -- see
docs/paper.md for the fub/fuv variety caveat and other limitations before
treating this as more than a research baseline.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import torch

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text import cleaners as coqui_cleaners

REPO_ROOT = Path(__file__).resolve().parents[1]
EC2_RUN = Path(
    "/home/ubuntu/fulfulde-tts/checkpoints/adamawa-full/"
    "fub_adamawa_full-August-03-2026_12+59PM-e650b45"
)
HF_REPO_ID = os.environ.get("FUB_HF_REPO", "ejnuma/adamawa-fulfulde-tts")
CHECKPOINT_NAME = "checkpoint_53000.pth"
EXAMPLE_SENTENCES = (
    "Hmm booɗɗum.",
    "Ɓiira ɓinngel goo wi’ata baaba : hokkam limce;",
    "Ndiyam lorake pat, o wurtake, o hoo’i yeeraande o hokkiti foondu, ɓe kuuci.",
)


def fub_character_cleaner(text: str) -> str:
    return " ".join(text.split())


setattr(coqui_cleaners, "fub_character_cleaner", fub_character_cleaner)


LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def checkpoint_problem(checkpoint: Path) -> str | None:
    """Return a readable reason the checkpoint cannot be loaded, or None if it can."""
    if not checkpoint.exists():
        return (
            f"No checkpoint at {checkpoint}. Point FUB_RUN_DIR at a directory "
            "holding config.json and checkpoint_53000.pth."
        )
    with checkpoint.open("rb") as handle:
        head = handle.read(len(LFS_POINTER_MAGIC))
    if head == LFS_POINTER_MAGIC:
        return (
            f"{checkpoint} is an unresolved Git LFS pointer of "
            f"{checkpoint.stat().st_size} bytes, not the model itself. The host "
            "cloned this repository without Git LFS support, or the repository is "
            "over its LFS bandwidth quota. Fetch the checkpoint from storage the "
            "host can read and set FUB_RUN_DIR to it."
        )
    return None


def hub_token() -> str | None:
    """Read the Hub token from the environment, falling back to Streamlit secrets."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        # No secrets file locally, or the key is absent on the host.
        return None


@st.cache_resource(show_spinner="Fetching the model from the Hub...")
def fetch_from_hub() -> Path:
    """Download config and checkpoint from the Hub, returning their shared directory."""
    from huggingface_hub import hf_hub_download

    token = hub_token()
    config = hf_hub_download(HF_REPO_ID, "config.json", token=token)
    hf_hub_download(HF_REPO_ID, CHECKPOINT_NAME, token=token)
    return Path(config).parent


def resolve_run_dir() -> Path:
    """FUB_RUN_DIR, then the local model/ copy, then the EC2 run, then the Hub."""
    override = os.environ.get("FUB_RUN_DIR")
    if override:
        return Path(override)
    for candidate in (REPO_ROOT / "model", EC2_RUN):
        if (candidate / "config.json").exists():
            return candidate
    return fetch_from_hub()


@st.cache_resource
def load_model(run_dir_str: str) -> Vits:
    run_dir = Path(run_dir_str)
    torch.set_num_threads(int(os.environ.get("FUB_THREADS", "4")))
    config = VitsConfig()
    config.load_json(str(run_dir / "config.json"))
    model = Vits.init_from_config(config)
    model.load_checkpoint(config, str(run_dir / "checkpoint_53000.pth"), eval=True, strict=True)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model


def synthesize(model: Vits, text: str) -> tuple[int, np.ndarray]:
    device = next(model.parameters()).device
    normalized = fub_character_cleaner(text)
    if len(normalized) > 300:
        raise ValueError("Keep text at or below 300 characters for this demo.")
    model.tokenizer.not_found_characters = []
    token_ids = model.tokenizer.text_to_ids(normalized, language="fub")
    unknown = sorted(set(model.tokenizer.not_found_characters), key=ord)
    if unknown:
        rendered = ", ".join(f"{c!r} (U+{ord(c):04X})" for c in unknown)
        raise ValueError(f"Unsupported characters: {rendered}")
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        waveform = model.inference(tokens, aux_input={})["model_outputs"].squeeze().cpu().numpy()
    waveform = waveform.astype(np.float32)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0:
        waveform *= 0.9 / peak
    return model.config.audio.sample_rate, waveform


def main() -> None:
    st.set_page_config(page_title="Adamawa Fulfulde TTS demo", page_icon="🗣️")
    st.title("Adamawa Fulfulde TTS -- research baseline demo")
    st.caption(
        "VITS model trained on the Adamawa-Fulfulde-TTS-Dataset (53,000 steps). "
        "This is Adamawa Fulfulde (fub) from a Nigerian-collected corpus; fub is "
        "a distinct code from Nigerian Fulfulde (fuv) -- "
        "see docs/paper.md. No native speaker has evaluated this output yet; "
        "your listening feedback is exactly what this demo is for."
    )

    run_dir = resolve_run_dir()
    problem = checkpoint_problem(run_dir / "checkpoint_53000.pth")
    if problem:
        st.error(problem)
        st.stop()

    model = load_model(str(run_dir))

    if "text_input" not in st.session_state:
        st.session_state.text_input = EXAMPLE_SENTENCES[0]

    st.subheader("Try an example")
    cols = st.columns(len(EXAMPLE_SENTENCES))
    for col, sentence in zip(cols, EXAMPLE_SENTENCES):
        short = sentence if len(sentence) <= 20 else sentence[:17] + "..."
        if col.button(short, help=sentence, use_container_width=True):
            st.session_state.text_input = sentence

    text = st.text_area("Adamawa Fulfulde text", key="text_input", height=100)

    if st.button("Synthesize", type="primary"):
        if not text.strip():
            st.warning("Enter some Fulfulde text first.")
        else:
            try:
                with st.spinner("Synthesizing..."):
                    sample_rate, waveform = synthesize(model, text.strip())
                buffer = io.BytesIO()
                sf.write(buffer, waveform, sample_rate, format="WAV")
                st.audio(buffer.getvalue(), format="audio/wav")
                st.caption(f"{waveform.shape[-1] / sample_rate:.2f}s at {sample_rate} Hz")
            except ValueError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
