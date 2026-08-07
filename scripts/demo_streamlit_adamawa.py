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
EXAMPLE_SENTENCES = (
    "Hmm booɗɗum.",
    "Ɓiira ɓinngel goo wi’ata baaba : hokkam limce;",
    "Ndiyam lorake pat, o wurtake, o hoo’i yeeraande o hokkiti foondu, ɓe kuuci.",
)


def fub_character_cleaner(text: str) -> str:
    return " ".join(text.split())


setattr(coqui_cleaners, "fub_character_cleaner", fub_character_cleaner)


def resolve_run_dir() -> Path:
    """Prefer FUB_RUN_DIR, then the repository's local model/ copy, then the EC2 run."""
    override = os.environ.get("FUB_RUN_DIR")
    if override:
        return Path(override)
    local = REPO_ROOT / "model"
    return local if (local / "config.json").exists() else EC2_RUN


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
        "This is Cameroon Adamawa Fulfulde (fub), not Nigerian Fulfulde (fuv) -- "
        "see docs/paper.md. No native speaker has evaluated this output yet; "
        "your listening feedback is exactly what this demo is for."
    )

    run_dir = resolve_run_dir()
    checkpoint = run_dir / "checkpoint_53000.pth"
    if not checkpoint.exists():
        st.error(
            f"No checkpoint at {checkpoint}. Point FUB_RUN_DIR at a directory "
            "holding config.json and checkpoint_53000.pth."
        )
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
