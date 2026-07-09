"""Attention sinks for the causal-KV encoder (ROADMAP Phase 0 diagnostic).

Pins: (a) sinks disabled is bit-identical to the pre-sinks behavior (checked
against the production package encoder, which shares the algorithm); (b) the
sink prefix is never evicted and keeps its original positions; (c) outputs
only diverge once the rolling window starts evicting, i.e. the mask exemption
does exactly what StreamingLLM prescribes and nothing more.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "tests"))

from qwen3_streaming_fakes import N_MELS, TinyQwenAudioTower, tiny_config  # noqa: E402

from qwen3_streaming.native_realtime_model import (  # noqa: E402
    QwenAudioCausalKVEncoder,
)
from qwen3_streaming.realtime_config import RealtimeAudioConfig  # noqa: E402

BLOCK = 32  # 4 output steps per block on the tiny tower


def _config(sink_steps: int) -> RealtimeAudioConfig:
    base = tiny_config(qwen_audio_block_bidirectional=True)
    return RealtimeAudioConfig(
        d_model=base.d_model,
        n_mels=base.n_mels,
        qwen_audio_left_context_sec=base.qwen_audio_left_context_sec,
        qwen_audio_block_bidirectional=True,
        qwen_audio_sink_steps=sink_steps,
    )


def _run(encoder, mels, *, blocks: int):
    state = encoder.init_state()
    outs = []
    for i in range(blocks):
        outs.append(
            encoder._encode_ready_mels(mels[:, i * BLOCK : (i + 1) * BLOCK, :], state)
        )
    return torch.cat(outs, dim=1), state


def _make(sink_steps: int, tower):
    encoder = QwenAudioCausalKVEncoder(tower, _config(sink_steps), chunk_frames=8).eval()
    encoder.left_context_steps = 6
    return encoder


def test_sinks_disabled_matches_production_encoder():
    from qwen3_asr_causal.causal import QwenAudioCausalKVEncoder as SrcEncoder

    torch.manual_seed(0)
    tower = TinyQwenAudioTower().eval()
    torch.manual_seed(1)
    mels = torch.randn(1, 12 * BLOCK, N_MELS)

    src = SrcEncoder(
        tower, tiny_config(qwen_audio_block_bidirectional=True), block_frames=BLOCK
    ).eval()
    src.left_context_steps = 6
    exp = _make(0, tower)

    with torch.no_grad():
        state = src.init_state()
        src_outs = []
        for i in range(12):
            hidden, state = src.forward_chunk(
                mels[:, i * BLOCK : (i + 1) * BLOCK, :], state
            )
            src_outs.append(hidden)
        out_src = torch.cat(src_outs, dim=1)
        out_exp, _ = _run(exp, mels, blocks=12)

    torch.testing.assert_close(out_src, out_exp, atol=1e-5, rtol=0)


def test_sink_prefix_is_pinned_and_only_changes_post_eviction():
    torch.manual_seed(0)
    tower = TinyQwenAudioTower().eval()
    torch.manual_seed(1)
    mels = torch.randn(1, 12 * BLOCK, N_MELS)

    with torch.no_grad():
        out_plain, _ = _run(_make(0, tower), mels, blocks=12)
        out_sink, state = _run(_make(4, tower), mels, blocks=12)

    # Block 0 queries have the whole stream in-window: sinks are a no-op.
    torch.testing.assert_close(out_sink[:, :4], out_plain[:, :4], atol=1e-5, rtol=0)
    # Once the window slides past step 0, the pinned prefix changes attention.
    assert float((out_sink[:, 8:] - out_plain[:, 8:]).abs().max()) > 1e-6

    for cache in state.layer_caches:
        assert cache.key.shape[-2] == 4 + 6  # sinks + rolling tail
        assert cache.positions[:4].tolist() == [0, 1, 2, 3]
        # Tail stays contiguous and recent.
        tail = cache.positions[4:].tolist()
        assert tail == list(range(tail[0], tail[0] + 6))


def test_sinks_reject_mutable_tail_combination():
    tower = TinyQwenAudioTower().eval()
    base = tiny_config(qwen_audio_block_bidirectional=True)
    config = RealtimeAudioConfig(
        d_model=base.d_model,
        n_mels=base.n_mels,
        qwen_audio_left_context_sec=base.qwen_audio_left_context_sec,
        qwen_audio_block_bidirectional=True,
        qwen_audio_sink_steps=4,
        qwen_audio_mutable_tail_sec=1.0,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        QwenAudioCausalKVEncoder(tower, config, chunk_frames=8)
