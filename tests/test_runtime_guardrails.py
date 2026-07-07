"""Serving guardrails: bounded event history, EOS flush loss bound, defaults.

These pin the fixes that make the backend safe for long-lived sessions:
- the per-session decode-event history must not grow for the session lifetime;
- the causal encoder's EOS flush must encode every whole 80 ms conv block and
  can only drop the sub-block remainder;
- the causal tower checkpoint defaults to the published HF repo everywhere.
"""

import pytest

from qwen3_asr_causal.model_paths import (
    DEFAULT_CAUSAL_TOWER_CHECKPOINT,
    DEFAULT_QWEN3_STREAMING_MODEL,
)
from qwen3_asr_causal.streamer import (
    CachedFullHypothesisConfig,
    CachedFullHypothesisStreamer,
)


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(f"w{int(token_id)}" for token_id in token_ids)

    def encode(self, text, add_special_tokens=False):
        return [10, 7, 11]


class FakeModel:
    def init_cached_audio_decode_state(self):
        return object()


def make_streamer(**config_kwargs):
    return CachedFullHypothesisStreamer(
        FakeModel(),
        FakeTokenizer(),
        CachedFullHypothesisConfig(
            wait_token_id=99,
            word_start_token_id=98,
            **config_kwargs,
        ),
    )


def test_event_history_is_bounded_by_default():
    streamer = make_streamer()
    assert streamer.config.event_history_limit == 64
    for step in range(80):
        streamer.update_from_hypothesis([1, 2], audio_sec=float(step))
    assert len(streamer.events) == 64
    assert streamer.events_total == 80
    # The retained window is the most recent one.
    assert streamer.events[-1]["audio_sec"] == 79.0
    assert streamer.events[0]["audio_sec"] == 16.0
    # online.py only relies on truthiness ("has any decode happened").
    assert bool(streamer.events)


def test_event_history_small_limit_keeps_tail():
    streamer = make_streamer(event_history_limit=4)
    for step in range(10):
        streamer.update_from_hypothesis([1], audio_sec=float(step))
    assert [event["audio_sec"] for event in streamer.events] == [6.0, 7.0, 8.0, 9.0]
    assert streamer.events_total == 10


def test_event_history_unbounded_is_opt_in():
    streamer = make_streamer(event_history_limit=0)
    for step in range(80):
        streamer.update_from_hypothesis([1], audio_sec=float(step))
    assert len(streamer.events) == 80


def test_commit_lag_estimate_matches_measured_p50():
    from qwen3_asr_causal.online import estimate_commit_lag_seconds

    # Measured per-word p50 on the 21-file MCIF replay (chunk 1.92 s):
    # 6w/2it 5.89 s, 6w/1it 3.98 s, 2w/1it 2.13 s.
    for hold, stable, measured in ((6, 2, 5.89), (6, 1, 3.98), (2, 1, 2.13)):
        estimate = estimate_commit_lag_seconds(
            chunk_sec=1.92,
            hold_back_words=hold,
            stable_iterations=stable,
        )
        assert abs(estimate - measured) < 0.35, (hold, stable, estimate)


def test_default_checkpoints_are_single_sourced():
    from qwen3_asr_causal import asr as asr_module
    from qwen3_asr_causal import cli as cli_module

    assert asr_module.DEFAULT_CAUSAL_TOWER_CHECKPOINT == DEFAULT_CAUSAL_TOWER_CHECKPOINT
    parser = cli_module.build_parser()
    args = parser.parse_args(["transcribe", "audio.wav"])
    assert args.tower == DEFAULT_CAUSAL_TOWER_CHECKPOINT
    assert args.model == DEFAULT_QWEN3_STREAMING_MODEL


def test_flush_pending_encodes_whole_blocks_and_bounds_loss():
    torch = pytest.importorskip("torch")
    from qwen3_asr_causal.causal import QwenAudioCausalKVEncoder

    from qwen3_streaming_fakes import N_MELS, TinyQwenAudioTower, tiny_config

    torch.manual_seed(0)
    tower = TinyQwenAudioTower().eval()
    config = tiny_config(qwen_audio_block_bidirectional=True)
    encoder = QwenAudioCausalKVEncoder(tower, config, block_frames=32).eval()

    state = encoder.init_state()
    # 77 frames: 2 whole attention blocks (64) consumed by forward_chunk,
    # 13 frames left pending = 1 whole conv block (8) + 5-frame remainder.
    mels = torch.randn(1, 77, N_MELS)
    hidden, state = encoder.forward_chunk(mels, state)
    assert hidden.shape[1] == 64 // encoder.chunk_frames
    assert state.mel_buffer is not None and state.mel_buffer.shape[1] == 13

    flushed, state = encoder.flush_pending(state)
    # The whole 80 ms conv block is encoded ...
    assert flushed.shape[1] == 8 // encoder.chunk_frames
    # ... and only the sub-block remainder (< chunk_frames = 80 ms) is gone.
    assert 13 - 8 < encoder.chunk_frames
    assert state.mel_buffer is None
    assert state.pending_frames == 0


def test_flush_pending_on_empty_buffer_is_a_noop():
    torch = pytest.importorskip("torch")
    from qwen3_asr_causal.causal import QwenAudioCausalKVEncoder

    from qwen3_streaming_fakes import N_MELS, TinyQwenAudioTower, tiny_config

    torch.manual_seed(0)
    tower = TinyQwenAudioTower().eval()
    encoder = QwenAudioCausalKVEncoder(
        tower,
        tiny_config(qwen_audio_block_bidirectional=True),
        block_frames=32,
    ).eval()

    state = encoder.init_state()
    mels = torch.randn(1, 64, N_MELS)
    _, state = encoder.forward_chunk(mels, state)

    flushed, state = encoder.flush_pending(state)
    assert flushed.shape[1] == 0
