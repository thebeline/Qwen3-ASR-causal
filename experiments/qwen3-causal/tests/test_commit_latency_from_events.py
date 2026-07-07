import importlib.util
import sys
from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "commit_latency_from_events.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "commit_latency_from_events",
    _SCRIPT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
latency = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = latency
_SPEC.loader.exec_module(latency)


def _event(
    hypothesis: str,
    *,
    audio_sec: float,
    input_mel_frames: int = 200,
    generate_ms: float = 500.0,
    segment_index: int = 0,
    completed_text: str = "",
    **extra,
):
    event = {
        "segment_index": segment_index,
        "segment_hypothesis": hypothesis,
        "completed_text": completed_text,
        "hypothesis": (completed_text + " " + hypothesis).strip(),
        "audio_sec": audio_sec,
        "input_mel_frames": input_mel_frames,
        "generate_ms": generate_ms,
        "new_cached_steps": 25,
    }
    event.update(extra)
    return event


GROWING_EVENTS = [
    _event("one two", audio_sec=2.0),
    _event("one two three four", audio_sec=4.0),
    _event("one two three four five", audio_sec=6.0),
    _event("one two three four five six", audio_sec=8.0),
]


def _timeline(events):
    return latency.EventTimeline.from_events(events, mel_hop_ms=10.0)


def test_timeline_wall_model_adds_generate_time_to_fed_audio():
    timeline = _timeline(GROWING_EVENTS)
    assert timeline.fed_sec == [2.0, 4.0, 6.0, 8.0]
    assert timeline.wall_sec == [2.5, 4.5, 6.5, 8.5]
    assert timeline.audio_total_sec == 8.0
    assert timeline.generate_ms_total == 2000.0


def test_replay_commits_lcp_and_flushes_tail_at_eos():
    timeline = _timeline(GROWING_EVENTS)
    replay = latency.replay_commit_stream(
        GROWING_EVENTS,
        timeline,
        hold_back_words=0,
        stable_iterations=1,
    )
    assert replay.emitted_text == "one two three four five six"
    # Raw-unit LCP stops one word short of the frontier (trailing-space
    # semantics): each event commits up to lcp-1 words even at hold_back=0.
    walls = [item.wall_sec for item in replay.emitted]
    assert walls == [4.5, 6.5, 6.5, 8.5, 8.5, 8.5]
    sources = [item.source for item in replay.emitted]
    assert sources == ["policy"] * 4 + ["eos_flush"] * 2
    assert replay.eos_flush_words == 2
    assert replay.counters.revision_events == 0


def test_hold_back_words_delays_commit():
    timeline = _timeline(GROWING_EVENTS)
    replay = latency.replay_commit_stream(
        GROWING_EVENTS,
        timeline,
        hold_back_words=2,
        stable_iterations=1,
    )
    # Effective holdback is hold_back+1 with raw-unit matching: e2 commits
    # lcp(3)-2=1 word, e3 commits lcp(4)-2=2 -> one more word.
    policy_words = [item for item in replay.emitted if item.source == "policy"]
    assert [item.word for item in policy_words] == ["one", "two"]
    assert [item.wall_sec for item in policy_words] == [6.5, 8.5]
    # EOS flush still recovers the tail.
    assert replay.emitted_text == "one two three four five six"


def test_first_stable_walls_track_prefix_growth():
    timeline = _timeline(GROWING_EVENTS)
    settled = ["one", "two", "three", "four", "five", "six"]
    first_wall, first_audio = latency.settled_first_stable_walls(
        GROWING_EVENTS, timeline, settled
    )
    assert first_wall == [2.5, 2.5, 4.5, 4.5, 6.5, 8.5]
    assert first_audio == [2.0, 2.0, 4.0, 4.0, 6.0, 8.0]


def test_revised_prefix_is_dropped_from_output_and_counted():
    events = [
        _event("a b c d", audio_sec=2.0),
        _event("a b c d", audio_sec=4.0),
        _event("a X c d e", audio_sec=6.0),
        _event("a X c d e", audio_sec=8.0),
    ]
    timeline = _timeline(events)
    replay = latency.replay_commit_stream(
        events,
        timeline,
        hold_back_words=0,
        stable_iterations=1,
    )
    # "a b c d" was committed, the later revision to "a X ..." must not
    # retract it; the EOS flush appends only the new tail.
    assert replay.emitted_text == "a b c d e"
    assert replay.counters.revision_events == 1
    assert replay.counters.revision_words == 1
    assert replay.emitted[-1].source == "eos_flush"


def test_segment_rollover_emits_finalized_tail_at_next_event():
    events = [
        _event("alpha beta gamma", audio_sec=2.0),
        _event(
            "alpha beta gamma delta",
            audio_sec=4.0,
            segment_rollover=True,
            segment_final_text="alpha beta gamma delta",
        ),
        _event(
            "epsilon",
            audio_sec=6.0,
            segment_index=1,
            completed_text="alpha beta gamma delta",
        ),
        _event(
            "epsilon zeta",
            audio_sec=8.0,
            segment_index=1,
            completed_text="alpha beta gamma delta",
        ),
    ]
    timeline = _timeline(events)
    replay = latency.replay_commit_stream(
        events,
        timeline,
        hold_back_words=0,
        stable_iterations=1,
    )
    assert replay.emitted_text == "alpha beta gamma delta epsilon zeta"
    by_word = {item.word: item for item in replay.emitted}
    # "gamma"/"delta" were never policy-committed inside segment 0 (raw LCP
    # stops before the frontier); they appear when the finalized segment
    # shows up in completed_text at event index 2.
    assert by_word["gamma"].source == "rollover"
    assert by_word["delta"].source == "rollover"
    assert by_word["delta"].wall_sec == 6.5
    # Segment 1 never repeats "epsilon zeta" before EOS, so both words ride
    # the final flush.
    assert by_word["epsilon"].source == "eos_flush"
    assert by_word["zeta"].source == "eos_flush"
    assert by_word["zeta"].wall_sec == 8.5


def test_stability_requires_new_audio_skips_duplicate_decodes():
    events = [
        _event("one two", audio_sec=2.0),
        _event("one two", audio_sec=4.0),
        _event(
            "one two",
            audio_sec=4.0,
            input_mel_frames=0,
            new_cached_steps=0,
        ),
        _event("one two", audio_sec=6.0),
    ]
    timeline = _timeline(events)
    eager = latency.replay_commit_stream(
        events,
        timeline,
        hold_back_words=0,
        stable_iterations=2,
    )
    strict = latency.replay_commit_stream(
        events,
        timeline,
        hold_back_words=0,
        stable_iterations=2,
        stability_requires_new_audio=True,
    )
    assert strict.counters.skipped_events == 1
    first_eager = next(item for item in eager.emitted if item.source == "policy")
    first_strict = next(item for item in strict.emitted if item.source == "policy")
    # The duplicate hypothesis counts as a free stability iteration in the
    # eager replay, committing one event earlier than the strict one.
    assert first_eager.wall_sec == 4.5
    assert first_strict.wall_sec == 6.5


def test_min_commit_audio_sec_blocks_early_commits():
    timeline = _timeline(GROWING_EVENTS)
    replay = latency.replay_commit_stream(
        GROWING_EVENTS,
        timeline,
        hold_back_words=0,
        stable_iterations=1,
        min_commit_audio_sec=5.0,
    )
    policy_walls = [
        item.wall_sec for item in replay.emitted if item.source == "policy"
    ]
    assert policy_walls and min(policy_walls) >= 6.5


def test_match_emitted_to_settled_handles_insertions():
    mapping = latency.match_emitted_to_settled(
        ["one", "two", "EXTRA", "three"],
        ["one", "two", "three"],
    )
    assert mapping == [0, 1, None, 2]


def _write_artifacts(tmp_path, events, *, reference, settled, alignments=None):
    events_dir = tmp_path / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    import json

    with (events_dir / "item1.jsonl").open("w") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    row = {
        "id": "item1",
        "reference": reference,
        "last_hypothesis_text": settled,
    }
    if alignments is not None:
        row["word_alignments"] = alignments
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps(row) + "\n")
    return predictions, events_dir


def test_measure_file_produces_aligned_latencies(tmp_path):
    alignments = [
        {"word": "one", "start": 0.0, "end": 0.5},
        {"word": "two", "start": 0.6, "end": 1.0},
        {"word": "three", "start": 2.1, "end": 2.5},
        {"word": "four", "start": 2.6, "end": 3.0},
        {"word": "five", "start": 4.1, "end": 4.5},
        {"word": "six", "start": 6.1, "end": 6.5},
    ]
    predictions, events_dir = _write_artifacts(
        tmp_path,
        GROWING_EVENTS,
        reference="one two three four five six",
        settled="one two three four five six",
        alignments=alignments,
    )
    rows = latency.load_jsonl(predictions)
    artifacts = latency.load_file_artifacts(
        rows[0],
        events_dir=events_dir,
        manifest_rows={},
        mel_hop_ms=10.0,
    )
    assert artifacts.aligned_end_sec == [0.5, 1.0, 2.5, 3.0, 4.5, 6.5]
    replay, measurements = latency.measure_file(
        artifacts,
        hold_back_words=0,
        stable_iterations=1,
        min_commit_audio_sec=0.0,
        normalize_commit_match=False,
        stability_requires_new_audio=False,
    )
    assert replay.emitted_text == "one two three four five six"
    aligned = [item.latency_aligned_sec for item in measurements]
    # commit walls [4.5,6.5,6.5,8.5,8.5,8.5] minus word end times.
    assert aligned == [4.0, 5.5, 4.0, 5.5, 4.0, 2.0]
    proxies = [item.latency_proxy_sec for item in measurements]
    # walls minus first-stable audio positions [2,2,4,4,6,8].
    assert proxies == [2.5, 4.5, 2.5, 4.5, 2.5, 0.5]
    delays = [item.delay_vs_first_stable_sec for item in measurements]
    # walls minus first-stable walls [2.5,2.5,4.5,4.5,6.5,8.5].
    assert delays == [2.0, 4.0, 2.0, 4.0, 2.0, 0.0]


def test_evaluate_config_aggregates_and_pareto(tmp_path):
    predictions, events_dir = _write_artifacts(
        tmp_path,
        GROWING_EVENTS,
        reference="one two three four five six",
        settled="one two three four five six",
    )
    rows = latency.load_jsonl(predictions)
    artifacts = latency.load_file_artifacts(
        rows[0],
        events_dir=events_dir,
        manifest_rows={},
        mel_hop_ms=10.0,
    )
    fast = latency.evaluate_config(
        [artifacts],
        hold_back_words=0,
        stable_iterations=1,
        min_commit_audio_sec=0.0,
        normalize_commit_match=False,
        stability_requires_new_audio=False,
    )
    slow = latency.evaluate_config(
        [artifacts],
        hold_back_words=4,
        stable_iterations=2,
        min_commit_audio_sec=0.0,
        normalize_commit_match=False,
        stability_requires_new_audio=False,
    )
    assert fast["wer_live_mean"] == 0.0
    assert fast["words_emitted_total"] == 6
    assert fast["settled_match_ratio"] == 1.0
    assert (
        fast["latency_proxy_sec"]["p50"] <= slow["latency_proxy_sec"]["p50"]
    )
    assert fast["decode_ms_per_audio_sec_mean"] == 250.0
    rows = [fast, slow]
    latency.mark_pareto(rows)
    assert fast["pareto"] is True


def test_grid_parsers_accept_ranges():
    assert latency.parse_int_grid("0,2,4:6") == [0, 2, 4, 5, 6]
    assert latency.parse_float_grid("0,1.5:2.5:0.5") == [0.0, 1.5, 2.0, 2.5]
