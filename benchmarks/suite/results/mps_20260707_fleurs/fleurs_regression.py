#!/usr/bin/env python3
"""FLEURS multilingual regression: causal tower vs windowed baseline.

The causal tower was distilled on English-only audio (LibriSpeech-960);
this measures what that did to other languages, deciding the HF-card
labeling (EN-only or not).

Usage: python fleurs_regression.py <lang_code> <backend> <n_utts> <out_json>
lang_code in {fr, de, zh}; backend in {causal, windowed}.
Expects <lang>.test.tsv and extracted audio under ./audio/<lang>/ next to
this script.
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "/Users/quentin/Documents/repos/WhisperLiveKit/third_party/qwen3-asr-causal/src")

from jiwer import cer, wer
from whisper_normalizer.basic import BasicTextNormalizer

from qwen3_asr_causal.asr import Qwen3StreamingASR
from qwen3_asr_causal.online import Qwen3StreamingOnlineProcessor

LANG = sys.argv[1]
BACKEND = sys.argv[2]
LIMIT = int(sys.argv[3])
OUT = Path(sys.argv[4])
HERE = Path(__file__).parent
FLEURS_DIR = {"fr": "fr_fr", "de": "de_de", "zh": "cmn_hans_cn"}[LANG]
norm = BasicTextNormalizer()


def utterances():
    count = 0
    with open(HERE / f"{FLEURS_DIR}.test.tsv") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            # id, filename, raw transcript, normalized transcript, ...
            filename, text = row[1], row[2]
            wav = HERE / "audio" / FLEURS_DIR / "test" / filename
            if not wav.exists():
                continue
            audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16_000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
            if len(audio) > 30 * 16_000:  # keep runs bounded
                continue
            yield audio.astype(np.float32), text
            count += 1
            if count >= LIMIT:
                return


def transcribe(asr, audio: np.ndarray) -> str:
    processor = Qwen3StreamingOnlineProcessor(asr)
    chunk = 8000
    words = []
    for start in range(0, len(audio), chunk):
        piece = audio[start : start + chunk]
        processor.insert_audio_chunk(piece, (start + len(piece)) / 16_000)
        tokens, _ = processor.process_iter(is_last=False)
        words.extend((t.text or "").strip() for t in tokens)
    tokens, _ = processor.finish()
    words.extend((t.text or "").strip() for t in tokens)
    joined = " ".join(w for w in words if w)
    return joined.replace(" ", "") if LANG == "zh" else joined


def main():
    kwargs = {"lan": LANG, "qwen3_streaming_audio_backend": BACKEND}
    asr = Qwen3StreamingASR(**kwargs)
    refs, hyps = [], []
    audio_sec = 0.0
    t0 = time.perf_counter()
    for index, (audio, text) in enumerate(utterances()):
        audio_sec += len(audio) / 16_000
        refs.append(norm(text))
        hyps.append(norm(transcribe(asr, audio)))
        if index % 25 == 0:
            print(f"{LANG}/{BACKEND}: {index}", flush=True)
    elapsed = time.perf_counter() - t0
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if LANG == "zh":
        score = cer([r for r, _ in pairs], [h for _, h in pairs])
        metric = "cer"
    else:
        score = wer([r for r, _ in pairs], [h for _, h in pairs])
        metric = "wer"
    result = {
        "language": LANG,
        "backend": BACKEND,
        "utts": len(pairs),
        "audio_min": round(audio_sec / 60, 1),
        "metric": metric,
        "score_basic_norm": round(score, 4),
        "rtf": round(elapsed / audio_sec, 3),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
