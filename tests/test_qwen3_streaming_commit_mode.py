"""Qwen3StreamingASR commit-mode validation and streamer dispatch (no model load)."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from qwen3_asr_causal.asr import Qwen3StreamingASR  # noqa: E402
from qwen3_asr_causal.sentence_streamer import SentenceSegmentedStreamer  # noqa: E402
from qwen3_asr_causal.streamer import SegmentedCachedFullHypothesisStreamer  # noqa: E402


def test_unknown_commit_mode_is_rejected_before_loading_the_model():
    with pytest.raises(ValueError, match="commit_mode"):
        Qwen3StreamingASR(lan="en", qwen3_streaming_commit_mode="word")


def test_sentence_soft_limit_at_or_above_the_cap_is_rejected():
    with pytest.raises(ValueError, match="soft limit"):
        Qwen3StreamingASR(
            lan="en",
            qwen3_streaming_commit_mode="sentence",
            qwen3_streaming_segment_max_steps=200,
            qwen3_streaming_sentence_soft_steps=200,
        )


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [10, 7, 11]


class _Model:
    def init_cached_audio_decode_state(self):
        return object()


def _unloaded_asr(**overrides):
    asr = object.__new__(Qwen3StreamingASR)
    asr.__dict__.update(
        original_language="en",
        base_context="",
        qwen_tokenizer=_Tokenizer(),
        model=_Model(),
        wait_token_id=99,
        word_start_token_id=98,
        eos_token_id=None,
        max_new_tokens=256,
        hold_back_words=6,
        stable_iterations=2,
        suppress_token_ids=(),
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        audio_placeholder_token_id=7,
        decoder_rolling_kv=False,
        speculative_draft=False,
        segment_max_steps=200,
        segment_keep_tail_steps=25,
        prompt_context_words=0,
        segment_punct_rollover=False,
        segment_punct_min_steps=150,
        segment_roll_before_generate=False,
        reset_encoder_on_rollover=False,
        commit_mode="prefix",
        sentence_soft_steps=150,
        sentence_rollover_margin_sec=0.5,
    )
    asr.__dict__.update(overrides)
    return asr


def test_prefix_mode_builds_the_unchanged_segmented_streamer():
    streamer = _unloaded_asr().build_streamer("en")

    assert type(streamer) is SegmentedCachedFullHypothesisStreamer
    assert streamer.segment_max_cached_steps == 200
    assert streamer.segment_keep_tail_steps == 25


def test_sentence_mode_builds_the_sentence_streamer_with_step_margin():
    streamer = _unloaded_asr(commit_mode="sentence", sentence_rollover_margin_sec=0.4).build_streamer("en")

    assert isinstance(streamer, SentenceSegmentedStreamer)
    assert streamer.sentence_soft_steps == 150
    assert streamer.sentence_rollover_margin_steps == 5
    assert streamer.segment_max_cached_steps == 200
    assert streamer.segment_keep_tail_steps == 25
