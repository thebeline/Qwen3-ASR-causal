#!/usr/bin/env python3
"""Replay saved streaming events and measure per-committed-word latency.

Replays the stable text commit policy over event logs saved by
``eval_cached_full_hypothesis.py --events-dir`` (no model re-run) and answers
the question the aggregate WER/RTF benches never did: *how long after a word
is spoken does the live contract commit it?*

Latency model (paced 1.0x session, decode-per-chunk):

    fed_sec(k)  = cumulative input_mel_frames * mel_hop_ms / 1000
    wall(k)     = max(fed_sec(k), audio_sec(k)) + generate_ms(k) / 1000

For each word the append-only emission stream commits, three measurements:

- ``latency_aligned_sec``   = wall(commit) - forced-aligner end time of the
  matched settled-transcript word (needs ``word_alignments`` in the manifest
  or predictions rows; the honest headline number).
- ``latency_proxy_sec``     = wall(commit) - audio_sec at the event where the
  settled word first became hypothesis-stable (alignment-free, biased low by
  up to one chunk: the word had been spoken by then).
- ``delay_vs_first_stable`` = wall(commit) - wall(first hypothesis-stable
  event): the pure policy-induced delay an oracle committer would not pay.

The replay is exact for the recorded decode cadence for any
(hold_back_words, stable_iterations, min_commit_audio_sec) policy: segment
rollovers are triggered by cached-step caps / punctuation (policy-independent)
and segment finals use finalize_mode="latest" (the last hypothesis, also
policy-independent). Changing chunk_sec/block_frames still requires re-running
the model.

The emission stream mirrors the production online processor
(qwen3_asr_causal.online.Qwen3StreamingOnlineProcessor._emit_committed):
append-only, shrunk or revised prefixes are dropped from output (counted here
as would-be retractions) while the revised text becomes the new diff baseline.

Note one production semantic this replay reproduces on purpose: with the
default ``normalize_commit_match=False``, commit units keep their trailing
whitespace, so the newest hypothesis word ("two") never raw-matches its later
form ("two "). The LCP therefore always stops one word short of the frontier:
hold_back_words=N behaves like N+1. Interpret sweep results accordingly.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from math import ceil
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen3_streaming.metrics import (  # noqa: E402
    hypothesis_word_end_times,
    word_error_rate,
)
from qwen3_streaming.stable_commit import (  # noqa: E402
    StableTextCommitState,
    normalize_text_unit_for_match,
    split_text_units,
    update_stable_text_commit,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_int_grid(value: str) -> list[int]:
    values: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = [part.strip() for part in item.split(":")]
            if len(parts) not in (2, 3):
                raise ValueError(f"invalid range item: {item!r}")
            start, stop = int(parts[0]), int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            if step <= 0:
                raise ValueError("range step must be > 0")
            values.extend(range(start, stop + 1, step))
        else:
            values.append(int(item))
    if not values:
        raise ValueError("grid cannot be empty")
    deduped = sorted(set(values))
    if deduped[0] < 0:
        raise ValueError("grid values must be >= 0")
    return deduped


def parse_float_grid(value: str) -> list[float]:
    values: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = [part.strip() for part in item.split(":")]
            if len(parts) not in (2, 3):
                raise ValueError(f"invalid range item: {item!r}")
            start, stop = float(parts[0]), float(parts[1])
            step = float(parts[2]) if len(parts) == 3 else 1.0
            if step <= 0.0:
                raise ValueError("range step must be > 0")
            current = start
            while current <= stop + 1e-9:
                values.append(round(current, 6))
                current += step
        else:
            values.append(float(item))
    if not values:
        raise ValueError("grid cannot be empty")
    deduped = sorted(set(values))
    if deduped[0] < 0.0:
        raise ValueError("grid values must be >= 0")
    return deduped


def _norm_word(word: str) -> str:
    return normalize_text_unit_for_match(word, case_sensitive=False)


def _words(text: str) -> list[str]:
    return [unit.strip() for unit in split_text_units(text or "")]


@dataclass
class EventTimeline:
    """Per-event wall clock and global hypothesis, config-independent."""

    wall_sec: list[float]
    audio_sec: list[float]
    fed_sec: list[float]
    generate_ms_total: float
    audio_total_sec: float

    @classmethod
    def from_events(
        cls,
        events: list[dict[str, Any]],
        *,
        mel_hop_ms: float,
    ) -> "EventTimeline":
        wall: list[float] = []
        audio: list[float] = []
        fed: list[float] = []
        fed_frames = 0
        generate_total = 0.0
        for event in events:
            fed_frames += int(event.get("input_mel_frames") or 0)
            fed_sec = fed_frames * mel_hop_ms / 1000.0
            audio_sec = float(event.get("audio_sec") or 0.0)
            generate_ms = float(event.get("generate_ms") or 0.0)
            generate_total += generate_ms
            base = max(fed_sec, audio_sec)
            wall.append(base + generate_ms / 1000.0)
            audio.append(audio_sec)
            fed.append(fed_sec)
        audio_total = max(fed[-1] if fed else 0.0, audio[-1] if audio else 0.0)
        return cls(
            wall_sec=wall,
            audio_sec=audio,
            fed_sec=fed,
            generate_ms_total=generate_total,
            audio_total_sec=audio_total,
        )


def global_hypothesis_words(event: dict[str, Any]) -> list[str]:
    """Completed segments plus the active segment hypothesis, as words."""
    if "segment_hypothesis" in event:
        return _words(str(event.get("completed_text") or "")) + _words(
            str(event.get("segment_hypothesis") or "")
        )
    return _words(str(event.get("hypothesis") or ""))


def settled_first_stable_walls(
    events: list[dict[str, Any]],
    timeline: EventTimeline,
    settled_words: list[str],
) -> tuple[list[float | None], list[float | None]]:
    """Earliest wall/audio time when each settled-word prefix became visible.

    Position j is filled at the first event whose global hypothesis starts
    with settled_words[0..j] (normalized, case-insensitive). This is the
    oracle committer's floor on the recorded hypothesis stream.
    """
    target = [_norm_word(word) for word in settled_words]
    first_wall: list[float | None] = [None] * len(target)
    first_audio: list[float | None] = [None] * len(target)
    best = 0
    for idx, event in enumerate(events):
        hyp = [_norm_word(word) for word in global_hypothesis_words(event)]
        limit = min(len(hyp), len(target))
        depth = 0
        while depth < limit and hyp[depth] == target[depth]:
            depth += 1
        for position in range(best, depth):
            first_wall[position] = timeline.wall_sec[idx]
            first_audio[position] = timeline.audio_sec[idx]
        best = max(best, depth)
        if best >= len(target):
            break
    return first_wall, first_audio


@dataclass
class EmittedWord:
    word: str
    event_index: int
    wall_sec: float
    source: str  # "policy" | "rollover" | "eos_flush"


@dataclass
class ReplayCounters:
    shrink_events: int = 0
    revision_events: int = 0
    revision_words: int = 0
    skipped_events: int = 0


@dataclass
class ReplayResult:
    emitted: list[EmittedWord]
    counters: ReplayCounters
    events_processed: int = 0
    eos_flush_words: int = 0

    @property
    def emitted_text(self) -> str:
        return " ".join(word.word for word in self.emitted)


def replay_commit_stream(
    events: list[dict[str, Any]],
    timeline: EventTimeline,
    *,
    hold_back_words: int,
    stable_iterations: int,
    min_commit_audio_sec: float = 0.0,
    normalize_commit_match: bool = False,
    stability_requires_new_audio: bool = False,
    final_flush: bool = True,
) -> ReplayResult:
    """Exact replay of the production append-only emission stream."""
    if hold_back_words < 0:
        raise ValueError("hold_back_words must be >= 0")
    if stable_iterations <= 0:
        raise ValueError("stable_iterations must be > 0")

    state = StableTextCommitState()
    counters = ReplayCounters()
    emitted: list[EmittedWord] = []
    current_segment: int | None = None
    last_segment_text = ""
    last_completed_words: list[str] = []
    processed = 0

    def emit_diff(
        target_words: list[str],
        completed_len: int,
        event_index: int,
        *,
        source_for_new: str,
    ) -> None:
        n_emitted = len(emitted)
        if len(target_words) < n_emitted:
            counters.shrink_events += 1
            return
        revised = sum(
            1
            for prev, new in zip(
                (item.word for item in emitted), target_words[:n_emitted]
            )
            if prev != new
        )
        if revised:
            counters.revision_events += 1
            counters.revision_words += revised
        for offset, word in enumerate(target_words[n_emitted:]):
            index = n_emitted + offset
            source = "rollover" if index < completed_len else source_for_new
            emitted.append(
                EmittedWord(
                    word=word,
                    event_index=event_index,
                    wall_sec=timeline.wall_sec[event_index],
                    source=source,
                )
            )

    for idx, event in enumerate(events):
        segment_index = event.get("segment_index")
        completed_words = (
            _words(str(event.get("completed_text") or ""))
            if "segment_hypothesis" in event
            else []
        )
        boundary = segment_index != current_segment or (
            len(completed_words) != len(last_completed_words)
        )
        if stability_requires_new_audio and not boundary:
            new_steps = int(event.get("new_cached_steps") or 0)
            if new_steps <= 0 and not bool(event.get("is_flush")):
                counters.skipped_events += 1
                continue
        if segment_index != current_segment:
            state = StableTextCommitState()
            current_segment = segment_index

        segment_text = str(
            event.get(
                "segment_hypothesis",
                event.get("hypothesis", ""),
            )
            or ""
        )
        update = update_stable_text_commit(
            state,
            segment_text,
            hold_back_units=hold_back_words,
            stable_iterations=stable_iterations,
            normalize_for_match=normalize_commit_match,
            allow_commit=float(event.get("audio_sec") or 0.0)
            >= min_commit_audio_sec,
        )
        last_segment_text = segment_text
        last_completed_words = completed_words
        target = completed_words + _words(update.committed_text)
        emit_diff(target, len(completed_words), idx, source_for_new="policy")
        processed += 1

    eos_before = len(emitted)
    if final_flush and processed and events:
        final_update = update_stable_text_commit(
            state,
            last_segment_text,
            hold_back_units=hold_back_words,
            stable_iterations=stable_iterations,
            normalize_for_match=normalize_commit_match,
            final=True,
            final_revises_committed=True,
        )
        target = last_completed_words + _words(final_update.committed_text)
        emit_diff(
            target,
            len(last_completed_words),
            len(events) - 1,
            source_for_new="eos_flush",
        )

    return ReplayResult(
        emitted=emitted,
        counters=counters,
        events_processed=processed,
        eos_flush_words=len(emitted) - eos_before,
    )


def match_emitted_to_settled(
    emitted_words: list[str],
    settled_words: list[str],
) -> list[int | None]:
    """Map each emitted word to a settled-transcript position (or None)."""
    emitted_norm = [_norm_word(word) for word in emitted_words]
    settled_norm = [_norm_word(word) for word in settled_words]
    mapping: list[int | None] = [None] * len(emitted_words)
    matcher = SequenceMatcher(None, settled_norm, emitted_norm, autojunk=False)
    for tag, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
        if tag != "equal":
            continue
        for ref_idx, hyp_idx in zip(range(ref_start, ref_end), range(hyp_start, hyp_end)):
            mapping[hyp_idx] = ref_idx
    return mapping


def _percentiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None}
    ordered = sorted(values)

    def rank(q: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, ceil(q * len(ordered)) - 1))]

    return {
        "count": len(ordered),
        "mean": statistics.mean(ordered),
        "p50": rank(0.50),
        "p90": rank(0.90),
        "p95": rank(0.95),
    }


def _mean(values: Iterable[float | int | None]) -> float | None:
    kept = [float(value) for value in values if value is not None]
    return statistics.mean(kept) if kept else None


@dataclass
class FileArtifacts:
    """Config-independent per-file data, computed once."""

    item_id: str
    events: list[dict[str, Any]]
    timeline: EventTimeline
    reference: str | None
    settled_words: list[str]
    settled_text: str
    first_stable_wall: list[float | None]
    first_stable_audio: list[float | None]
    aligned_end_sec: list[float | None] | None
    wer_settled: float | None


def load_file_artifacts(
    row: dict[str, Any],
    *,
    events_dir: Path,
    manifest_rows: dict[str, dict[str, Any]],
    mel_hop_ms: float,
) -> FileArtifacts:
    item_id = str(row["id"])
    event_path = events_dir / f"{item_id}.jsonl"
    if not event_path.exists():
        raise FileNotFoundError(event_path)
    events = load_jsonl(event_path)
    timeline = EventTimeline.from_events(events, mel_hop_ms=mel_hop_ms)
    manifest_row = manifest_rows.get(item_id, {})
    reference = row.get("reference") or manifest_row.get("text") or None
    settled_text = str(
        row.get("last_hypothesis_text") or row.get("final_text") or ""
    )
    settled_words = _words(settled_text)
    first_wall, first_audio = settled_first_stable_walls(
        events, timeline, settled_words
    )
    word_alignments = manifest_row.get("word_alignments") or row.get(
        "word_alignments"
    )
    aligned_end: list[float | None] | None = None
    if word_alignments:
        aligned_end = hypothesis_word_end_times(
            split_text_units(settled_text),
            word_alignments,
        )
    return FileArtifacts(
        item_id=item_id,
        events=events,
        timeline=timeline,
        reference=str(reference) if reference else None,
        settled_words=settled_words,
        settled_text=settled_text,
        first_stable_wall=first_wall,
        first_stable_audio=first_audio,
        aligned_end_sec=aligned_end,
        wer_settled=word_error_rate(str(reference), settled_text)
        if reference
        else None,
    )


@dataclass
class WordMeasurement:
    file_id: str
    word: str
    emitted_index: int
    settled_index: int | None
    source: str
    event_index: int
    wall_sec: float
    first_stable_wall_sec: float | None = None
    first_stable_audio_sec: float | None = None
    aligned_end_sec: float | None = None

    @property
    def delay_vs_first_stable_sec(self) -> float | None:
        if self.first_stable_wall_sec is None:
            return None
        return self.wall_sec - self.first_stable_wall_sec

    @property
    def latency_proxy_sec(self) -> float | None:
        if self.first_stable_audio_sec is None:
            return None
        return self.wall_sec - self.first_stable_audio_sec

    @property
    def latency_aligned_sec(self) -> float | None:
        if self.aligned_end_sec is None:
            return None
        return self.wall_sec - self.aligned_end_sec


def measure_file(
    artifacts: FileArtifacts,
    *,
    hold_back_words: int,
    stable_iterations: int,
    min_commit_audio_sec: float,
    normalize_commit_match: bool,
    stability_requires_new_audio: bool,
) -> tuple[ReplayResult, list[WordMeasurement]]:
    replay = replay_commit_stream(
        artifacts.events,
        artifacts.timeline,
        hold_back_words=hold_back_words,
        stable_iterations=stable_iterations,
        min_commit_audio_sec=min_commit_audio_sec,
        normalize_commit_match=normalize_commit_match,
        stability_requires_new_audio=stability_requires_new_audio,
    )
    emitted_words = [item.word for item in replay.emitted]
    mapping = match_emitted_to_settled(emitted_words, artifacts.settled_words)
    measurements: list[WordMeasurement] = []
    for emitted_index, (item, settled_index) in enumerate(
        zip(replay.emitted, mapping)
    ):
        measurement = WordMeasurement(
            file_id=artifacts.item_id,
            word=item.word,
            emitted_index=emitted_index,
            settled_index=settled_index,
            source=item.source,
            event_index=item.event_index,
            wall_sec=item.wall_sec,
        )
        if settled_index is not None:
            measurement.first_stable_wall_sec = artifacts.first_stable_wall[
                settled_index
            ]
            measurement.first_stable_audio_sec = artifacts.first_stable_audio[
                settled_index
            ]
            if artifacts.aligned_end_sec is not None:
                measurement.aligned_end_sec = artifacts.aligned_end_sec[
                    settled_index
                ]
        measurements.append(measurement)
    return replay, measurements


def evaluate_config(
    files: list[FileArtifacts],
    *,
    hold_back_words: int,
    stable_iterations: int,
    min_commit_audio_sec: float,
    normalize_commit_match: bool,
    stability_requires_new_audio: bool,
    per_word_rows: list[WordMeasurement] | None = None,
) -> dict[str, Any]:
    pooled_aligned: list[float] = []
    pooled_proxy: list[float] = []
    pooled_delay: list[float] = []
    per_file_p50_aligned: list[float | None] = []
    wer_live: list[float | None] = []
    wer_settled: list[float | None] = []
    first_commit: list[float | None] = []
    eos_words: list[int] = []
    matched = 0
    total_words = 0
    rollover_words = 0
    shrink_total = 0
    revision_events_total = 0
    revision_words_total = 0
    decode_cost: list[float] = []

    for artifacts in files:
        replay, measurements = measure_file(
            artifacts,
            hold_back_words=hold_back_words,
            stable_iterations=stable_iterations,
            min_commit_audio_sec=min_commit_audio_sec,
            normalize_commit_match=normalize_commit_match,
            stability_requires_new_audio=stability_requires_new_audio,
        )
        if per_word_rows is not None:
            per_word_rows.extend(measurements)
        aligned = [
            value
            for item in measurements
            if (value := item.latency_aligned_sec) is not None
        ]
        proxy = [
            value
            for item in measurements
            if (value := item.latency_proxy_sec) is not None
        ]
        delay = [
            value
            for item in measurements
            if (value := item.delay_vs_first_stable_sec) is not None
        ]
        pooled_aligned.extend(aligned)
        pooled_proxy.extend(proxy)
        pooled_delay.extend(delay)
        per_file_p50_aligned.append(_percentiles(aligned)["p50"])
        wer_live.append(
            word_error_rate(artifacts.reference, replay.emitted_text)
            if artifacts.reference
            else None
        )
        wer_settled.append(artifacts.wer_settled)
        first_commit.append(
            replay.emitted[0].wall_sec if replay.emitted else None
        )
        eos_words.append(replay.eos_flush_words)
        matched += sum(
            1 for item in measurements if item.settled_index is not None
        )
        total_words += len(measurements)
        rollover_words += sum(
            1 for item in measurements if item.source == "rollover"
        )
        shrink_total += replay.counters.shrink_events
        revision_events_total += replay.counters.revision_events
        revision_words_total += replay.counters.revision_words
        if artifacts.timeline.audio_total_sec > 0:
            decode_cost.append(
                artifacts.timeline.generate_ms_total
                / artifacts.timeline.audio_total_sec
            )

    return {
        "hold_back_words": hold_back_words,
        "stable_iterations": stable_iterations,
        "min_commit_audio_sec": min_commit_audio_sec,
        "normalize_commit_match": normalize_commit_match,
        "stability_requires_new_audio": stability_requires_new_audio,
        "count_files": len(files),
        "words_emitted_total": total_words,
        "settled_match_ratio": (matched / total_words) if total_words else None,
        "rollover_words_ratio": (
            rollover_words / total_words if total_words else None
        ),
        "eos_flush_words_mean": _mean(eos_words),
        "latency_aligned_sec": _percentiles(pooled_aligned),
        "latency_proxy_sec": _percentiles(pooled_proxy),
        "delay_vs_first_stable_sec": _percentiles(pooled_delay),
        "latency_aligned_p50_per_file_mean": _mean(per_file_p50_aligned),
        "first_commit_wall_sec_mean": _mean(first_commit),
        "wer_live_mean": _mean(wer_live),
        "wer_settled_mean": _mean(wer_settled),
        "would_be_shrink_events_total": shrink_total,
        "would_be_revision_events_total": revision_events_total,
        "would_be_revision_words_total": revision_words_total,
        "decode_ms_per_audio_sec_mean": _mean(decode_cost),
    }


def _headline_latency(row: dict[str, Any]) -> float:
    aligned = row["latency_aligned_sec"]
    proxy = row["latency_proxy_sec"]
    if aligned.get("count"):
        return float(aligned["p50"])
    if proxy.get("count"):
        return float(proxy["p50"])
    return float("inf")


def mark_pareto(rows: list[dict[str, Any]]) -> None:
    """Pareto front over (wer_live_mean, headline latency p50), lower=better."""
    for row in rows:
        row["pareto"] = False
    candidates = [
        row for row in rows if row.get("wer_live_mean") is not None
    ]
    for row in candidates:
        latency = _headline_latency(row)
        wer = float(row["wer_live_mean"])
        dominated = any(
            other is not row
            and float(other["wer_live_mean"]) <= wer
            and _headline_latency(other) <= latency
            and (
                float(other["wer_live_mean"]) < wer
                or _headline_latency(other) < latency
            )
            for other in candidates
        )
        row["pareto"] = not dominated


def write_per_word_csv(path: Path, rows: list[WordMeasurement]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file_id",
                "emitted_index",
                "settled_index",
                "word",
                "source",
                "event_index",
                "wall_sec",
                "first_stable_wall_sec",
                "first_stable_audio_sec",
                "aligned_end_sec",
                "delay_vs_first_stable_sec",
                "latency_proxy_sec",
                "latency_aligned_sec",
            ]
        )
        for item in rows:
            writer.writerow(
                [
                    item.file_id,
                    item.emitted_index,
                    item.settled_index,
                    item.word,
                    item.source,
                    item.event_index,
                    f"{item.wall_sec:.3f}",
                    _fmt(item.first_stable_wall_sec),
                    _fmt(item.first_stable_audio_sec),
                    _fmt(item.aligned_end_sec),
                    _fmt(item.delay_vs_first_stable_sec),
                    _fmt(item.latency_proxy_sec),
                    _fmt(item.latency_aligned_sec),
                ]
            )


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else ""


def _manifest_by_id(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = load_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("id") or row.get("audio_id")
        if key is None and row.get("audio"):
            key = Path(str(row["audio"])).stem
        if key is None and row.get("wav"):
            key = Path(str(row["wav"])).stem
        if key is not None:
            by_id[str(key)] = row
    return by_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure per-committed-word latency by replaying saved streaming "
            "events under a (grid of) stable-commit policies, without "
            "rerunning ASR inference."
        )
    )
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--manifest-jsonl",
        type=Path,
        default=None,
        help="Optional manifest with word_alignments (and reference text).",
    )
    parser.add_argument("--hold-back-words", default="6")
    parser.add_argument("--stable-iterations", default="2")
    parser.add_argument("--min-commit-audio-sec", default="0")
    parser.add_argument("--normalize-commit-match", action="store_true")
    parser.add_argument(
        "--stability-requires-new-audio",
        action="store_true",
        help=(
            "Skip decode events that added no encoder steps (models a "
            "block-aligned decode cadence; without this, sub-block chunks "
            "produce duplicate hypotheses that count as free stability "
            "iterations)."
        ),
    )
    parser.add_argument("--mel-hop-ms", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--per-word-csv",
        type=Path,
        default=None,
        help="Dump per-word measurements (single-config grids only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_rows = [
        row for row in load_jsonl(args.predictions_jsonl) if row.get("error") is None
    ]
    manifest_rows = _manifest_by_id(args.manifest_jsonl)
    hold_back_grid = parse_int_grid(args.hold_back_words)
    stable_grid = parse_int_grid(args.stable_iterations)
    min_commit_grid = parse_float_grid(args.min_commit_audio_sec)

    n_configs = len(hold_back_grid) * len(stable_grid) * len(min_commit_grid)
    if args.per_word_csv is not None and n_configs != 1:
        raise SystemExit("--per-word-csv requires a single-config grid")

    files = [
        load_file_artifacts(
            row,
            events_dir=args.events_dir,
            manifest_rows=manifest_rows,
            mel_hop_ms=args.mel_hop_ms,
        )
        for row in prediction_rows
    ]
    aligned_files = sum(1 for item in files if item.aligned_end_sec is not None)

    results: list[dict[str, Any]] = []
    per_word_rows: list[WordMeasurement] | None = (
        [] if args.per_word_csv is not None else None
    )
    for hold_back_words in hold_back_grid:
        for stable_iterations in stable_grid:
            for min_commit_audio_sec in min_commit_grid:
                results.append(
                    evaluate_config(
                        files,
                        hold_back_words=hold_back_words,
                        stable_iterations=stable_iterations,
                        min_commit_audio_sec=min_commit_audio_sec,
                        normalize_commit_match=args.normalize_commit_match,
                        stability_requires_new_audio=args.stability_requires_new_audio,
                        per_word_rows=per_word_rows,
                    )
                )

    mark_pareto(results)
    results.sort(key=_headline_latency)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.per_word_csv is not None and per_word_rows is not None:
        write_per_word_csv(args.per_word_csv, per_word_rows)

    summary = {
        "files": len(files),
        "files_with_alignments": aligned_files,
        "configs": len(results),
        "pareto": [
            {
                "hold_back_words": row["hold_back_words"],
                "stable_iterations": row["stable_iterations"],
                "min_commit_audio_sec": row["min_commit_audio_sec"],
                "latency_aligned_p50": row["latency_aligned_sec"].get("p50"),
                "latency_proxy_p50": row["latency_proxy_sec"].get("p50"),
                "delay_vs_first_stable_p50": row[
                    "delay_vs_first_stable_sec"
                ].get("p50"),
                "wer_live_mean": row["wer_live_mean"],
                "would_be_revision_words_total": row[
                    "would_be_revision_words_total"
                ],
            }
            for row in results
            if row.get("pareto")
        ][: args.top_k],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
