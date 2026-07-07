"""Opt-in end-to-end test with real weights and real audio.

Run with::

    QWEN3_ASR_CAUSAL_E2E=1 pytest tests/test_e2e.py

Downloads Qwen/Qwen3-ASR-0.6B and the published causal tower on first run
(~2.3 GB). The audio fixture is a 6 s macOS-TTS clip committed to the repo;
it exercises the full causal path: streaming mel extraction, append-only
causal encoder, rolling-KV decoder, stable commit, EOS flush.
"""

import os

import numpy as np
import pytest

E2E = os.environ.get("QWEN3_ASR_CAUSAL_E2E") == "1"

pytestmark = pytest.mark.skipif(
    not E2E,
    reason="set QWEN3_ASR_CAUSAL_E2E=1 to run the real-weights E2E test",
)

WAV = os.path.join(os.path.dirname(__file__), "data", "e2e_smoke.wav")
# TTS script: "The quick brown fox jumps over the lazy dog. Streaming
# transcription should recognize this final sentence."
EXPECTED_KEYWORDS = ("quick", "brown", "fox", "lazy", "dog")
EXPECTED_FINAL_WORD = "sentence"


@pytest.fixture(scope="module")
def transcription():
    sf = pytest.importorskip("soundfile")
    from qwen3_asr_causal import Qwen3StreamingASR, Qwen3StreamingOnlineProcessor

    audio, sample_rate = sf.read(WAV, dtype="float32", always_2d=False)
    assert sample_rate == 16_000

    asr = Qwen3StreamingASR(
        lan="en",
        qwen3_streaming_audio_backend="causal",
        # tower checkpoint intentionally omitted: the published default
        # must resolve on its own.
    )
    processor = Qwen3StreamingOnlineProcessor(asr)

    emitted: list[str] = []
    chunk = int(0.5 * sample_rate)
    for start in range(0, len(audio), chunk):
        end_time = min(len(audio), start + chunk) / sample_rate
        processor.insert_audio_chunk(audio[start : start + chunk], end_time)
        tokens, _ = processor.process_iter(is_last=False)
        for token in tokens:
            emitted.append((token.text or "").strip())
    final_tokens, _ = processor.finish()
    for token in final_tokens:
        emitted.append((token.text or "").strip())

    words = [word for word in emitted if word]
    return {"words": words, "text": " ".join(words), "asr": asr}


def test_transcript_contains_expected_content(transcription):
    text = transcription["text"].lower()
    assert transcription["words"], "no words emitted"
    hits = sum(1 for keyword in EXPECTED_KEYWORDS if keyword in text)
    assert hits >= 4, f"transcript missed the script: {transcription['text']!r}"


def test_eos_flush_emits_the_final_word(transcription):
    text = transcription["text"].lower()
    assert EXPECTED_FINAL_WORD in text, (
        "the held-back tail was not flushed at EOS: "
        f"{transcription['text']!r}"
    )


def test_default_tower_checkpoint_resolved(transcription):
    asr = transcription["asr"]
    assert asr.tower_checkpoint, "causal mode must resolve a tower checkpoint"


def test_second_session_is_independent(transcription):
    sf = pytest.importorskip("soundfile")
    from qwen3_asr_causal import Qwen3StreamingOnlineProcessor

    audio, sample_rate = sf.read(WAV, dtype="float32", always_2d=False)
    processor = Qwen3StreamingOnlineProcessor(transcription["asr"])
    processor.insert_audio_chunk(
        np.asarray(audio, dtype=np.float32), len(audio) / sample_rate
    )
    tokens, _ = processor.finish()
    text = " ".join((token.text or "").strip() for token in tokens).lower()
    assert "fox" in text or "dog" in text
