#!/usr/bin/env python3
"""WER bench for the live no-rewrite contract (manifest or LibriSpeech).

Feeds each file through ONE fresh Qwen3StreamingOnlineProcessor session
(0.5 s chunks, as fast as the decoder allows) and reports, per file and
pooled over the corpus:

- live WER: the concatenated append-only emission stream, exactly what a
  caption viewer saw (whisper-normalized per language);
- settled WER: the final transcript after finish(), including any
  segment-rollover revisions the live stream dropped;
- RTF: session wall time / audio duration;
- retractions: emissions that failed to extend the already-shown prefix
  (the no-rewrite contract; any nonzero count is a bug);
- revised_words_dropped: words where the settled transcript disagrees with
  what was emitted (revisions the contract silently dropped from output).

Usage:
    python benchmarks/suite/fetch_data.py --librispeech test-clean
    python benchmarks/suite/run_wer.py \
        --librispeech-dir ~/.cache/qwen3_asr_causal/bench/LibriSpeech/test-clean \
        --limit 50 --output-json out.json

    python benchmarks/suite/run_wer.py --manifest-jsonl manifest.jsonl
    # rows: {"id","audio","text" (or "human_text"),"language"?}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    SAMPLE_RATE,
    load_wav,
    manifest_rows,
    system_info,
    transcribe_session,
    whisper_norm,
)


def librispeech_rows(root: Path) -> list[dict[str, Any]]:
    """Walk an extracted LibriSpeech split: *.trans.txt lines + .flac files."""
    rows: list[dict[str, Any]] = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt_id, _, text = line.partition(" ")
            flac = trans.parent / f"{utt_id}.flac"
            if not flac.exists():
                raise SystemExit(f"missing audio for {utt_id}: {flac}")
            rows.append(
                {
                    "id": utt_id,
                    "audio": flac,
                    "text": text.strip(),
                    "language": "en",
                    "word_alignments": None,
                }
            )
    if not rows:
        raise SystemExit(f"no *.trans.txt under {root} (extracted LibriSpeech layout?)")
    return rows


def count_retractions(emissions: list[tuple[str, float, float, float]]) -> int:
    """Emissions that failed to extend the shown prefix (must be zero)."""
    shown = ""
    retractions = 0
    for text, _, _, _ in emissions:
        candidate = shown + text
        if not candidate.startswith(shown):  # pragma: no cover - contract
            retractions += 1
        shown = candidate
    return retractions


def count_revised_words(live_text: str, settled_text: str) -> int:
    live_words = live_text.split()
    settled_words = settled_text.split()
    return sum(1 for a, b in zip(live_words, settled_words) if a != b) + abs(
        len(live_words) - len(settled_words)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest-jsonl", type=Path, default=None)
    source.add_argument(
        "--librispeech-dir",
        type=Path,
        default=None,
        help="Extracted split root, e.g. .../LibriSpeech/test-clean",
    )
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.manifest_jsonl is not None:
        rows = manifest_rows(args.manifest_jsonl, audio_dir=args.audio_dir)
    else:
        rows = librispeech_rows(args.librispeech_dir.expanduser())
    missing_text = [row["id"] for row in rows if not row["text"]]
    if missing_text:
        raise SystemExit(f"rows without reference text: {missing_text[:5]} ...")
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

    import jiwer

    normalizers: dict[str, Any] = {}
    per_file: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        lang = row["language"] or args.language
        # Sequential bench: switching the session language on the shared ASR
        # is safe here (one session at a time; servers use SessionASRProxy).
        asr.original_language = lang
        audio = load_wav(row["audio"])
        audio_sec = len(audio) / SAMPLE_RATE
        wall0 = time.perf_counter()
        emissions, settled_text = transcribe_session(
            asr, audio, feed_chunk_sec=args.feed_chunk_sec, pace=None
        )
        wall = time.perf_counter() - wall0
        live_text = "".join(text for text, _, _, _ in emissions)

        norm = normalizers.setdefault(lang, whisper_norm(lang))
        ref_norm = norm(row["text"])
        live_norm = norm(live_text)
        settled_norm = norm(settled_text)
        result = {
            "id": row["id"],
            "language": lang,
            "audio_sec": round(audio_sec, 2),
            "wall_sec": round(wall, 2),
            "rtf": round(wall / audio_sec, 4) if audio_sec else None,
            "words_emitted": len(emissions),
            "wer_live": round(jiwer.wer(ref_norm, live_norm), 4) if ref_norm else None,
            "wer_settled": (
                round(jiwer.wer(ref_norm, settled_norm), 4) if ref_norm else None
            ),
            "retractions": count_retractions(emissions),
            "revised_words_dropped": count_revised_words(live_text, settled_text),
            "live_text": live_text,
            "settled_text": settled_text,
        }
        bucket = pooled.setdefault(lang, {"ref": [], "live": [], "settled": []})
        if ref_norm:
            bucket["ref"].append(ref_norm)
            bucket["live"].append(live_norm)
            bucket["settled"].append(settled_norm)
        per_file.append(result)
        printable = {k: v for k, v in result.items() if not k.endswith("_text")}
        print(json.dumps(printable, ensure_ascii=False), flush=True)

    corpus: dict[str, Any] = {}
    for lang, bucket in pooled.items():
        corpus[lang] = {
            "files": len(bucket["ref"]),
            "wer_live": round(jiwer.wer(bucket["ref"], bucket["live"]), 4),
            "wer_settled": round(jiwer.wer(bucket["ref"], bucket["settled"]), 4),
        }
    audio_total = sum(item["audio_sec"] for item in per_file)
    wall_total = sum(item["wall_sec"] for item in per_file)
    summary = {
        "config": {
            "audio_backend": args.audio_backend,
            "hold_back_words": asr.hold_back_words,
            "stable_iterations": asr.stable_iterations,
            "chunk_sec": asr.chunk_sec,
            "feed_chunk_sec": args.feed_chunk_sec,
            "model_id": asr.model_id,
            "normalization": "whisper (English rules for 'en', basic otherwise)",
        },
        "system": system_info(),
        "files": len(per_file),
        "audio_sec_total": round(audio_total, 1),
        "rtf_overall": round(wall_total / audio_total, 4) if audio_total else None,
        "corpus_wer": corpus,
        "retractions_total": sum(item["retractions"] for item in per_file),
        "revised_words_dropped_total": sum(
            item["revised_words_dropped"] for item in per_file
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({**summary, "per_file": per_file}, ensure_ascii=False, indent=2)
        )


if __name__ == "__main__":
    main()
