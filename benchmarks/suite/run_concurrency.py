#!/usr/bin/env python3
"""Concurrent-session bench for the HF backend (shared ASR, one decode lock).

Spawns N sessions on threads against ONE shared Qwen3StreamingASR, each
feeding its own audio at 1.0x wall-clock pace (0.5 s chunks, staggered
starts), and measures what serialization does to each session:

- backlog: pending un-decoded audio per session over time (median / p95 /
  final): the saturation signal; a healthy session keeps backlog near the
  decode cadence, a starved one drifts upward;
- decode busy ratio: summed decode-lock time / wall time (1.0 = the lock is
  the bottleneck);
- per-session committed words/min (starvation shows as output rate loss).

The HF path serializes all sessions behind asr.decode_lock by design; this
bench documents how many sessions a device actually sustains. Use the vLLM
backend for larger fleets.

Usage:
    python benchmarks/suite/run_concurrency.py --audio-dir DIR \
        --sessions 1,2,4,6 --seconds 120
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


def load_wavs(audio_dir: Path) -> list[np.ndarray]:
    import soundfile as sf

    tracks = []
    for wav in sorted(audio_dir.glob("*.wav")):
        audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            raise SystemExit(f"{wav} is {sr} Hz; expected 16 kHz")
        tracks.append(np.asarray(audio, dtype=np.float32))
    if not tracks:
        raise SystemExit(f"no wavs under {audio_dir}")
    return tracks


def run_session(
    asr,
    audio: np.ndarray,
    *,
    seconds: float,
    feed_chunk_sec: float,
    start_barrier: threading.Barrier,
    stagger_sec: float,
    out: dict,
) -> None:
    try:
        _run_session(
            asr,
            audio,
            seconds=seconds,
            feed_chunk_sec=feed_chunk_sec,
            start_barrier=start_barrier,
            stagger_sec=stagger_sec,
            out=out,
        )
    except Exception as exc:  # noqa: BLE001 - the bench must report failures
        out["error"] = f"{type(exc).__name__}: {exc}"


def _run_session(
    asr,
    audio: np.ndarray,
    *,
    seconds: float,
    feed_chunk_sec: float,
    start_barrier: threading.Barrier,
    stagger_sec: float,
    out: dict,
) -> None:
    from qwen3_asr_causal import Qwen3StreamingOnlineProcessor

    processor = Qwen3StreamingOnlineProcessor(asr)
    chunk = int(feed_chunk_sec * SAMPLE_RATE)
    total = min(len(audio), int(seconds * SAMPLE_RATE))
    backlog: list[float] = []
    decode_time = 0.0
    words = 0

    start_barrier.wait()
    time.sleep(stagger_sec)
    wall0 = time.perf_counter()
    for start in range(0, total, chunk):
        piece = audio[start : start + chunk]
        end_time = (start + len(piece)) / SAMPLE_RATE
        # Pace at 1.0x: never feed audio before its wall time.
        lead = end_time - (time.perf_counter() - wall0)
        if lead > 0:
            time.sleep(lead)
        processor.insert_audio_chunk(piece, end_time)
        backlog.append(len(processor.audio_buffer) / SAMPLE_RATE)
        t0 = time.perf_counter()
        tokens, _ = processor.process_iter(is_last=False)
        decode_time += time.perf_counter() - t0
        words += len(tokens)
    tokens, _ = processor.finish()
    words += len(tokens)
    wall = time.perf_counter() - wall0

    ordered = sorted(backlog)
    out.update(
        {
            "wall_sec": round(wall, 1),
            "audio_sec": round(total / SAMPLE_RATE, 1),
            "words": words,
            "words_per_min": round(words / (total / SAMPLE_RATE) * 60.0, 1),
            "decode_busy_sec": round(decode_time, 1),
            "backlog_median_sec": round(ordered[len(ordered) // 2], 2),
            "backlog_p95_sec": round(
                ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2
            ),
            "backlog_final_sec": round(backlog[-1], 2) if backlog else 0.0,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--sessions", default="1,2,4,6")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--feed-chunk-sec", type=float, default=0.5)
    parser.add_argument("--language", default="en")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    from qwen3_asr_causal import Qwen3StreamingASR

    tracks = load_wavs(args.audio_dir)
    asr = Qwen3StreamingASR(lan=args.language, qwen3_streaming_audio_backend="causal")

    results = []
    for n_sessions in [int(x) for x in args.sessions.split(",") if x.strip()]:
        barrier = threading.Barrier(n_sessions)
        outs: list[dict] = [{} for _ in range(n_sessions)]
        threads = [
            threading.Thread(
                target=run_session,
                args=(asr, tracks[i % len(tracks)]),
                kwargs={
                    "seconds": args.seconds,
                    "feed_chunk_sec": args.feed_chunk_sec,
                    "start_barrier": barrier,
                    "stagger_sec": (i * 0.7) % 2.0,
                    "out": outs[i],
                },
                daemon=True,
            )
            for i in range(n_sessions)
        ]
        wall0 = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall = time.perf_counter() - wall0
        healthy = [o for o in outs if "error" not in o and o]
        errors = [o["error"] for o in outs if "error" in o]
        row = {
            "sessions": n_sessions,
            "wall_sec": round(wall, 1),
            "sessions_failed": len(outs) - len(healthy),
            "errors": sorted(set(errors)),
        }
        if healthy:
            busy = sum(o["decode_busy_sec"] for o in healthy)
            row.update(
                {
                    "decode_busy_ratio": round(busy / wall, 3),
                    "backlog_median_sec_worst": max(
                        o["backlog_median_sec"] for o in healthy
                    ),
                    "backlog_p95_sec_worst": max(
                        o["backlog_p95_sec"] for o in healthy
                    ),
                    "backlog_final_sec_worst": max(
                        o["backlog_final_sec"] for o in healthy
                    ),
                    "words_per_min_min": min(o["words_per_min"] for o in healthy),
                    "words_per_min_median": statistics.median(
                        o["words_per_min"] for o in healthy
                    ),
                }
            )
        row["per_session"] = outs
        results.append(row)
        printable = {k: v for k, v in row.items() if k != "per_session"}
        print(json.dumps(printable), flush=True)
        if errors:
            print(
                f"sessions={n_sessions}: {len(errors)} session(s) failed; "
                "stopping the sweep at the saturation point",
                flush=True,
            )
            break

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"feed_chunk_sec": args.feed_chunk_sec, "runs": results}, indent=2)
        )


if __name__ == "__main__":
    main()
