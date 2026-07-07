#!/usr/bin/env python3
"""Live per-committed-word latency at 1.0x pace (wall clock, no replay).

Feeds each manifest file through ONE fresh Qwen3StreamingOnlineProcessor at
real-time pace (0.5 s chunks, never ahead of the wall clock) and records the
wall-clock arrival of every committed token relative to stream start. This
measures the same quantity as the offline event replay
(experiments/qwen3-causal/scripts/commit_latency_from_events.py) but LIVE:
real decode times, real pacing, no event log.

Two latency readings per committed word, clearly labeled:

- latency_token_sec = arrival_wall - token.end. Always available. The
  token's own end timestamp is BACKEND-ESTIMATED (linear interpolation
  back-dated by the expected commit lag), so this is a proxy; its p50 is
  also the sanity-guarded headline when no alignments exist.
- latency_aligned_sec = arrival_wall - aligned word end. Only when the
  manifest row carries "word_alignments" (forced-aligner words with
  start/end); emitted words are matched to aligned words with a
  SequenceMatcher over casefolded words (equal blocks only). The honest
  number when available.

Also reported: time-to-first-token per file (p50/p90/p95 over files).

Sanity guard: pooled latency_token p50 outside [0.5, 30] s exits nonzero
(a broken clock or a stalled session, not a real operating point).

Usage:
    python benchmarks/suite/run_latency.py --manifest-jsonl manifest.jsonl \
        --output-json latency.json
    # rows: {"id","audio","text"?,"language"?,"word_alignments"?:
    #        [{"word","start","end"}, ...]}
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from difflib import SequenceMatcher
from math import ceil
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    SAMPLE_RATE,
    load_wav,
    manifest_rows,
    system_info,
    transcribe_session,
)


def _match_word(word: str) -> str:
    """Casefolded word with surrounding punctuation stripped, for matching."""
    folded = word.casefold()
    while folded and unicodedata.category(folded[0])[0] in ("P", "S"):
        folded = folded[1:]
    while folded and unicodedata.category(folded[-1])[0] in ("P", "S"):
        folded = folded[:-1]
    return folded


def alignment_entries(word_alignments: list[Any]) -> list[tuple[str, float]]:
    """(word, end_sec) pairs from manifest word_alignments rows."""
    entries: list[tuple[str, float]] = []
    for item in word_alignments or []:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("word") or item.get("unit")
        end = item.get("end_sec", item.get("end_time", item.get("end")))
        if text is None or end is None:
            continue
        entries.append((str(text), float(end)))
    return entries


def aligned_end_times(
    emitted_words: list[str], entries: list[tuple[str, float]]
) -> list[float | None]:
    """Per-emitted-word aligned end time; None where no equal-block match."""
    ends: list[float | None] = [None] * len(emitted_words)
    if not entries:
        return ends
    hyp = [_match_word(word) for word in emitted_words]
    ref = [_match_word(word) for word, _ in entries]
    matcher = SequenceMatcher(None, ref, hyp, autojunk=False)
    for tag, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
        if tag != "equal":
            continue
        for ref_idx, hyp_idx in zip(range(ref_start, ref_end), range(hyp_start, hyp_end)):
            ends[hyp_idx] = entries[ref_idx][1]
    return ends


def percentiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p90": None, "p95": None}
    ordered = sorted(values)

    def rank(q: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, ceil(q * len(ordered)) - 1))]

    return {
        "count": len(ordered),
        "p50": round(rank(0.50), 3),
        "p90": round(rank(0.90), 3),
        "p95": round(rank(0.95), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="base dir for relative manifest audio paths (default: manifest dir)",
    )
    parser.add_argument(
        "--audio-backend", choices=("causal", "windowed"), default="causal"
    )
    parser.add_argument("--stable-iterations", type=int, default=None)
    parser.add_argument("--hold-back-words", type=int, default=None)
    parser.add_argument("--language", default="en", help="default when a row has none")
    parser.add_argument("--feed-chunk-sec", type=float, default=0.5)
    parser.add_argument(
        "--pace", type=float, default=1.0, help="1.0 = real time (the point of this bench)"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    rows = manifest_rows(args.manifest_jsonl, audio_dir=args.audio_dir)
    if args.limit:
        rows = rows[: args.limit]

    from qwen3_asr_causal import Qwen3StreamingASR

    asr_kwargs: dict[str, Any] = {
        "lan": args.language,
        "qwen3_streaming_audio_backend": args.audio_backend,
    }
    if args.stable_iterations is not None:
        asr_kwargs["qwen3_streaming_stable_iterations"] = args.stable_iterations
    if args.hold_back_words is not None:
        asr_kwargs["qwen3_streaming_hold_back_words"] = args.hold_back_words
    asr = Qwen3StreamingASR(**asr_kwargs)

    pooled_token: list[float] = []
    pooled_aligned: list[float] = []
    first_tokens: list[float] = []
    per_file: list[dict[str, Any]] = []
    for row in rows:
        asr.original_language = row["language"] or args.language
        audio = load_wav(row["audio"])
        audio_sec = len(audio) / SAMPLE_RATE
        emissions, settled_text = transcribe_session(
            asr, audio, feed_chunk_sec=args.feed_chunk_sec, pace=args.pace
        )
        emitted_words = [text.strip() for text, _, _, _ in emissions]
        token_lat = [arrival - end for _, arrival, _, end in emissions]
        entries = alignment_entries(row["word_alignments"])
        ends = aligned_end_times(emitted_words, entries)
        aligned_lat = [
            arrival - end
            for (_, arrival, _, _), end in zip(emissions, ends)
            if end is not None
        ]
        result = {
            "id": row["id"],
            "audio_sec": round(audio_sec, 2),
            "words_emitted": len(emissions),
            "time_to_first_token_sec": (
                round(emissions[0][1], 3) if emissions else None
            ),
            "latency_token_sec": percentiles(token_lat),
            "latency_aligned_sec": percentiles(aligned_lat),
            "aligned_match_ratio": (
                round(len(aligned_lat) / len(emissions), 3)
                if emissions and entries
                else None
            ),
            "settled_text": settled_text,
        }
        pooled_token.extend(token_lat)
        pooled_aligned.extend(aligned_lat)
        if emissions:
            first_tokens.append(emissions[0][1])
        per_file.append(result)
        printable = {k: v for k, v in result.items() if k != "settled_text"}
        print(json.dumps(printable, ensure_ascii=False), flush=True)

    summary = {
        "config": {
            "audio_backend": args.audio_backend,
            "hold_back_words": asr.hold_back_words,
            "stable_iterations": asr.stable_iterations,
            "chunk_sec": asr.chunk_sec,
            "feed_chunk_sec": args.feed_chunk_sec,
            "pace": args.pace,
            "model_id": asr.model_id,
        },
        "system": system_info(),
        "files": len(per_file),
        "files_with_alignments": sum(
            1 for row in rows if alignment_entries(row["word_alignments"])
        ),
        "latency_token_sec": percentiles(pooled_token),
        "latency_aligned_sec": percentiles(pooled_aligned),
        "time_to_first_token_sec": percentiles(first_tokens),
        "note": (
            "latency_token uses backend-estimated token end timestamps "
            "(proxy); latency_aligned uses manifest forced-aligner word ends "
            "(honest headline when present)."
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({**summary, "per_file": per_file}, ensure_ascii=False, indent=2)
        )

    p50 = summary["latency_token_sec"]["p50"]
    if p50 is None or not (0.5 <= p50 <= 30.0):
        print(
            f"SANITY FAIL: pooled latency_token p50 = {p50}; expected within "
            "[0.5, 30] s at 1.0x pace. Check pacing, clocks, and that the "
            "session actually committed words.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
