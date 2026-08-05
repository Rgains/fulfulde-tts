#!/usr/bin/env python3
"""Run validated Adamawa Fulfulde VITS inference on CPU.

The script supports a one-shot command-line mode and a dependency-free local
web interface. The web server binds to localhost by default and is intended to
be reached through SSH or VS Code port forwarding.
"""

from __future__ import annotations

import argparse
import html
import json
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf
import torch

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text import cleaners as coqui_cleaners

DEFAULT_RUN = Path(
    "/home/ubuntu/fulfulde-tts/checkpoints/adamawa-full/"
    "fub_adamawa_full-August-03-2026_12+59PM-e650b45"
)
EXAMPLES = (
    "Hmm booɗɗum.",
    "Ɓiira ɓinngel goo wi’ata baaba : hokkam limce;",
    "Ndiyam lorake pat, o wurtake, o hoo’i yeeraande o hokkiti foondu, ɓe kuuci.",
)


def fub_character_cleaner(text: str) -> str:
    """Preserve Fulfulde graphemes and collapse file-format whitespace only."""
    return " ".join(text.split())


setattr(coqui_cleaners, "fub_character_cleaner", fub_character_cleaner)


def save_waveform(path: Path, waveform: torch.Tensor, sample_rate: int) -> float:
    audio = waveform.detach().float().cpu().squeeze().numpy().astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.98:
        audio *= 0.98 / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    return sf.info(path).duration


class CpuSynthesizer:
    def __init__(self, config_path: Path, checkpoint_path: Path, threads: int) -> None:
        torch.set_num_threads(threads)
        self.config = VitsConfig()
        self.config.load_json(str(config_path))
        if self.config.use_speaker_embedding:
            raise RuntimeError("This demo supports the unconditioned Adamawa model only")
        self.model = Vits.init_from_config(self.config)
        self.model.load_checkpoint(
            self.config, str(checkpoint_path), eval=True, strict=True
        )
        self.model = self.model.cpu().eval()
        self.lock = threading.Lock()

    def synthesize(self, text: str, output: Path) -> dict[str, float | str]:
        normalized = fub_character_cleaner(text)
        if not normalized:
            raise ValueError("Enter a non-empty Adamawa Fulfulde sentence")
        if len(normalized) > 300:
            raise ValueError("For a live demo, keep text at or below 300 characters")

        self.model.tokenizer.not_found_characters = []
        ids = self.model.tokenizer.text_to_ids(normalized, language="fub")
        unknown = sorted(set(self.model.tokenizer.not_found_characters), key=ord)
        if unknown:
            rendered = " ".join(f"{char!r} (U+{ord(char):04X})" for char in unknown)
            raise ValueError(f"Unsupported characters: {rendered}")

        tokens = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        started = time.perf_counter()
        with self.lock, torch.inference_mode():
            waveform = self.model.inference(tokens, aux_input={})["model_outputs"]
        inference_seconds = time.perf_counter() - started
        duration = save_waveform(output, waveform, self.config.audio.sample_rate)
        return {
            "text": normalized,
            "output": str(output),
            "audio_duration_seconds": duration,
            "inference_seconds": inference_seconds,
            "real_time_factor": inference_seconds / duration,
        }


def page(message: str = "", audio_url: str = "", text: str = EXAMPLES[0]) -> bytes:
    examples = "".join(
        f'<button type="button" onclick="document.querySelector(\'textarea\').value='
        f"{json.dumps(example, ensure_ascii=False)}>{html.escape(example)}</button>"
        for example in EXAMPLES
    )
    player = f'<audio controls autoplay src="{html.escape(audio_url)}"></audio>' if audio_url else ""
    document = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Adamawa Fulfulde TTS</title>
<style>
body {{ font: 18px system-ui, sans-serif; max-width: 760px; margin: 4rem auto; padding: 0 1rem; background:#faf8f2; color:#17221b }}
textarea {{ box-sizing:border-box; width:100%; min-height:8rem; padding:1rem; font:inherit }}
button {{ margin:.5rem .35rem .5rem 0; padding:.7rem 1rem; cursor:pointer }}
.generate {{ background:#176b45; color:white; border:0; border-radius:.35rem }}
audio {{ width:100%; margin-top:1rem }}
.status {{ min-height:1.6rem; color:#604000 }}
</style>
<h1>Adamawa Fulfulde speech demo</h1>
<p>Character-level VITS research baseline (<code>fub</code>). Type one short sentence.</p>
<form method="post" action="/synthesize">
<textarea name="text" maxlength="300" required>{html.escape(text)}</textarea><br>
<button class="generate" type="submit">Generate speech</button>
</form>
<div>{examples}</div>
<p class="status">{html.escape(message)}</p>
{player}
<p><small>Research demo. Generated audio has not yet received native-speaker quality evaluation.</small></p>
</html>"""
    return document.encode("utf-8")


def serve(synthesizer: CpuSynthesizer, output_dir: Path, host: str, port: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, body: bytes, status: HTTPStatus = HTTPStatus.OK, content_type: str = "text/html; charset=utf-8") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.respond(page())
                return
            if parsed.path.startswith("/audio/"):
                name = Path(parsed.path).name
                audio_path = output_dir / name
                if not name.endswith(".wav") or not audio_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self.respond(audio_path.read_bytes(), content_type="audio/wav")
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/synthesize":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 4096:
                    raise ValueError("Request is too large")
                values = parse_qs(self.rfile.read(length).decode("utf-8"))
                requested_text = values.get("text", [""])[0]
                filename = f"demo_{uuid.uuid4().hex}.wav"
                result = synthesizer.synthesize(requested_text, output_dir / filename)
                message = (
                    f"Generated {result['audio_duration_seconds']:.2f} s of audio "
                    f"in {result['inference_seconds']:.2f} s on CPU."
                )
                self.respond(page(message, f"/audio/{filename}", requested_text))
            except (UnicodeDecodeError, ValueError) as error:
                self.respond(page(str(error)), HTTPStatus.BAD_REQUEST)
            except Exception as error:
                self.respond(page(f"Synthesis failed: {error}"), HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}")

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Adamawa Fulfulde CPU demo: http://{host}:{port}")
    print("Use SSH/VS Code port forwarding when host is 127.0.0.1.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping CPU demo.")
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_RUN / "config.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN / "checkpoint_53000.pth")
    parser.add_argument("--threads", type=int, default=max(1, min(4, (torch.get_num_threads() or 1))))
    parser.add_argument("--text", help="Synthesize once instead of starting the web demo")
    parser.add_argument("--output", type=Path, default=Path("demo_output.wav"))
    parser.add_argument("--output-dir", type=Path, default=Path("demo_outputs"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    for path in (args.config, args.checkpoint):
        if not path.is_file():
            parser.error(f"file not found: {path}")

    started = time.perf_counter()
    synthesizer = CpuSynthesizer(args.config, args.checkpoint, args.threads)
    print(f"Loaded model on CPU in {time.perf_counter() - started:.2f} seconds")
    if args.text is not None:
        result = synthesizer.synthesize(args.text, args.output.resolve())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    serve(synthesizer, args.output_dir.resolve(), args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
