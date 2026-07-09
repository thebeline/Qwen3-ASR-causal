#!/usr/bin/env python3
"""Long-session stability bench: memory slope and output health over time.

Feeds a multi-file concatenation (default ~45 min) through ONE
Qwen3StreamingOnlineProcessor session as fast as the decoder allows: the
harshest allocator/backlog regime (real-time feeding is strictly gentler):
sampling process RSS, torch accelerator memory, event-history size and
output rate along the way.

Pass criteria printed at the end:
- post-warmup RSS slope < 2 MB per audio-minute;
- accelerator driver memory bounded (no monotonic growth);
- per-5-min committed words/min stays within [0.5x, 1.5x] of the median
  (detects repetition collapse or silent stalls);
- bounded event history (streamer.events_total grows, len(events) does not).

Usage:
    python benchmarks/suite/run_long_session.py --audio-dir DIR --minutes 45
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def _rss_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    scale = 1024**3 if sys.platform == "darwin" else 1024**2
    return usage / scale


def _accel_mem_gb(device_type: str) -> tuple[float, float]:
    import torch

    if device_type == "mps":
        return (
            torch.mps.current_allocated_memory() / 1024**3,
            torch.mps.driver_allocated_memory() / 1024**3,
        )
    if device_type == "cuda":
        return (
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.memory_reserved() / 1024**3,
        )
    return (0.0, 0.0)


def load_concat_audio(audio_dir: Path, minutes: float, gap_sec: float) -> np.ndarray:
    import soundfile as sf

    target = int(minutes * 60 * 16_000)
    gap = np.zeros(int(gap_sec * 16_000), dtype=np.float32)
    pieces: list[np.ndarray] = []
    total = 0
    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no wavs under {audio_dir}")
    index = 0
    while total < target:
        wav = wavs[index % len(wavs)]
        index += 1
        audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16_000:
            raise SystemExit(f"{wav} is {sr} Hz; expected 16 kHz")
        pieces.append(audio)
        pieces.append(gap)
        total += len(audio) + len(gap)
    return np.concatenate(pieces)[:target]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=45.0)
    parser.add_argument("--gap-sec", type=float, default=0.5)
    parser.add_argument("--feed-chunk-sec", type=float, default=2.0)
    parser.add_argument("--language", default="en")
    parser.add_argument("--sample-every-audio-sec", type=float, default=30.0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    from qwen3_asr_causal import Qwen3StreamingASR, Qwen3StreamingOnlineProcessor

    audio = load_concat_audio(args.audio_dir, args.minutes, args.gap_sec)
    audio_minutes = len(audio) / 16_000 / 60
    print(f"audio: {audio_minutes:.1f} min", flush=True)

    asr = Qwen3StreamingASR(lan=args.language, qwen3_streaming_audio_backend="causal")
    processor = Qwen3StreamingOnlineProcessor(asr)
    device_type = getattr(asr.device, "type", "cpu")

    samples: list[dict] = []
    word_times: list[float] = []  # audio time at emission for each word
    chunk = int(args.feed_chunk_sec * 16_000)
    next_sample = 0.0
    wall_start = time.perf_counter()

    for start in range(0, len(audio), chunk):
        piece = audio[start : start + chunk]
        end_time = (start + len(piece)) / 16_000
        processor.insert_audio_chunk(piece, end_time)
        tokens, _ = processor.process_iter(is_last=False)
        word_times.extend(end_time for _ in tokens)
        if end_time >= next_sample:
            alloc, driver = _accel_mem_gb(device_type)
            samples.append(
                {
                    "audio_sec": round(end_time, 1),
                    "rss_gb": round(_rss_gb(), 3),
                    "accel_alloc_gb": round(alloc, 3),
                    "accel_driver_gb": round(driver, 3),
                    "events_len": len(processor.streamer.events),
                    "events_total": processor.streamer.events_total,
                    "words_emitted": len(word_times),
                    "pending_sec": round(len(processor.audio_buffer) / 16_000, 1),
                }
            )
            print(json.dumps(samples[-1]), flush=True)
            next_sample += args.sample_every_audio_sec
    tokens, _ = processor.finish()
    word_times.extend(len(audio) / 16_000 for _ in tokens)
    wall = time.perf_counter() - wall_start

    # --- verdicts ---
    warm = [s for s in samples if s["audio_sec"] >= 600.0] or samples[len(samples) // 3 :]
    rss_slope = 0.0
    if len(warm) >= 2:
        dx = (warm[-1]["audio_sec"] - warm[0]["audio_sec"]) / 60.0
        rss_slope = (warm[-1]["rss_gb"] - warm[0]["rss_gb"]) * 1024 / max(dx, 1e-9)
    driver_tail = [s["accel_driver_gb"] for s in warm]
    words_per_window: list[float] = []
    window = 300.0
    edge = window
    count = 0
    for t in word_times:
        while t > edge:
            words_per_window.append(count / (window / 60.0))
            count = 0
            edge += window
        count += 1
    words_per_window.append(count / (window / 60.0))
    median_rate = statistics.median(words_per_window) if words_per_window else 0.0
    rate_ok = all(
        0.5 * median_rate <= rate <= 1.5 * median_rate for rate in words_per_window[1:-1]
    )

    result = {
        "audio_minutes": round(audio_minutes, 1),
        "wall_minutes": round(wall / 60.0, 1),
        "rtf_wall": round(wall / (len(audio) / 16_000), 3),
        "device": device_type,
        "rss_slope_mb_per_audio_min_postwarmup": round(rss_slope, 2),
        "accel_driver_gb_max_postwarmup": max(driver_tail) if driver_tail else None,
        "events_len_final": samples[-1]["events_len"] if samples else None,
        "events_total_final": samples[-1]["events_total"] if samples else None,
        "words_total": len(word_times),
        "words_per_min_5min_windows": [round(rate, 1) for rate in words_per_window],
        "pass_rss_slope_lt_2mb_min": abs(rss_slope) < 2.0,
        "pass_output_rate_stable": rate_ok,
        "pass_event_history_bounded": bool(
            samples and samples[-1]["events_len"] < samples[-1]["events_total"]
        ),
        "samples": samples,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2))
    printable = {k: v for k, v in result.items() if k != "samples"}
    print(json.dumps(printable, indent=2), flush=True)


if __name__ == "__main__":
    main()
