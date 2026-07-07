"""Shared helpers for the benchmark suite.

Everything here imports only the public ``qwen3_asr_causal`` package plus
stdlib / numpy / soundfile (and optionally librosa for resampling, jiwer +
whisper_normalizer in the callers), so the numbers describe the package as
users install it.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16_000


def load_wav(path: Path | str) -> np.ndarray:
    """Load any soundfile-readable audio as 16 kHz mono float32."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        try:
            import librosa
        except ImportError as exc:
            raise SystemExit(
                f"{path} is {sr} Hz; expected {SAMPLE_RATE}. "
                "Install librosa to resample, or convert the file."
            ) from exc
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.ascontiguousarray(audio, dtype=np.float32)


def transcribe_session(
    asr,
    audio: np.ndarray,
    *,
    feed_chunk_sec: float = 0.5,
    pace: float | None = None,
) -> tuple[list[tuple[str, float, float, float]], str]:
    """Feed one audio track through a fresh online processor session.

    Returns ``(emissions, settled_text)``:

    - ``emissions``: one ``(token_text, arrival_wall_sec, token_start,
      token_end)`` tuple per committed ASRToken, in emission order.
      ``arrival_wall_sec`` is wall-clock time relative to the start of the
      feed loop, taken right after the ``process_iter``/``finish`` call that
      produced the token (so it includes that decode's cost). Token
      start/end are the backend's own timestamp estimates.
    - ``settled_text``: the final transcript after ``finish()``, including
      any segment-rollover revisions the append-only emission stream
      dropped (falls back to the concatenated emitted tokens).

    ``pace=1.0`` sleeps so audio is never fed ahead of real time;
    ``pace=None`` feeds as fast as the decoder allows. Other values scale
    real time (``pace=2.0`` feeds at 2x).
    """
    from qwen3_asr_causal import Qwen3StreamingOnlineProcessor

    processor = Qwen3StreamingOnlineProcessor(asr)
    chunk = max(1, int(feed_chunk_sec * SAMPLE_RATE))
    emissions: list[tuple[str, float, float, float]] = []
    wall0 = time.perf_counter()
    for start in range(0, len(audio), chunk):
        piece = audio[start : start + chunk]
        end_time = (start + len(piece)) / SAMPLE_RATE
        if pace is not None:
            # Never feed audio before its wall time (scaled by pace).
            lead = end_time / pace - (time.perf_counter() - wall0)
            if lead > 0:
                time.sleep(lead)
        processor.insert_audio_chunk(piece, end_time)
        tokens, _ = processor.process_iter(is_last=False)
        arrival = time.perf_counter() - wall0
        emissions.extend((t.text or "", arrival, t.start, t.end) for t in tokens)
    tokens, _ = processor.finish()
    arrival = time.perf_counter() - wall0
    emissions.extend((t.text or "", arrival, t.start, t.end) for t in tokens)

    live_text = "".join(text for text, _, _, _ in emissions)
    settled_words = getattr(processor, "_emitted_words", None)
    settled_text = " ".join(settled_words) if settled_words else live_text
    return emissions, settled_text


def system_info() -> dict[str, Any]:
    """Platform / torch / accelerator snapshot for result provenance."""
    import torch

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
    }


def whisper_norm(lang: str | None):
    """Whisper text normalizer: English rules for 'en', basic otherwise."""
    if (lang or "").lower() in ("en", "eng", "english"):
        from whisper_normalizer.english import EnglishTextNormalizer

        return EnglishTextNormalizer()
    from whisper_normalizer.basic import BasicTextNormalizer

    return BasicTextNormalizer()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def manifest_rows(path: Path, audio_dir: Path | None = None) -> list[dict[str, Any]]:
    """Normalize manifest rows to {id, audio, text, language, word_alignments}.

    Accepts both the suite convention ({"id","audio","text"}) and the MCIF
    refs convention ({"audio_id","wav","human_text"}). Relative audio paths
    resolve against ``audio_dir`` when given, else the manifest's directory.
    """
    path = Path(path)
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(path):
        audio = row.get("audio") or row.get("wav")
        if not audio:
            raise SystemExit(f"{path}: manifest row without 'audio'/'wav': {row}")
        audio_path = Path(str(audio)).expanduser()
        if not audio_path.is_absolute():
            base = Path(audio_dir).expanduser() if audio_dir else path.parent
            audio_path = base / audio_path
        item_id = row.get("id") or row.get("audio_id") or audio_path.stem
        rows.append(
            {
                "id": str(item_id),
                "audio": audio_path,
                "text": row.get("text") or row.get("human_text") or None,
                "language": row.get("language"),
                "word_alignments": row.get("word_alignments"),
            }
        )
    return rows
