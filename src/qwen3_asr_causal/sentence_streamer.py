"""Sentence-commit variant of the segmented full-hypothesis streamer.

``commit_mode="sentence"`` replaces the word-prefix commit policy and the
wholesale time-based segment rollover:

* Commit: whole sentences, once complete and unchanged (word identity) for
  ``stable_iterations`` decodes (see ``sentence_commit.update_sentence_commit``).
  ``hold_back_words`` does not apply: the unfinished last sentence is the
  hold-back.
* Soft rollover: once the active segment reaches ``sentence_soft_steps``, the
  segment is cut at the end of its last committed sentence. Committed
  sentences leave the decoded audio (they stop being re-decoded every chunk,
  which is what bounds decode cost), and all audio from the cut on is carried
  into the next segment so the in-progress sentence keeps its start.
* Seam: the cut is placed ``sentence_rollover_margin_steps`` before the
  estimated sentence end, so the next segment re-transcribes a few committed
  words; ``sentence_commit.seam_overlap`` drops them.
* Safety valve: a segment past ``segment_max_cached_steps`` with no usable
  sentence end is finalized wholesale (committed + latest remainder), as in
  the prefix mode, carrying ``segment_keep_tail_steps`` and de-duplicating
  that tail too.

Committed text only ever grows by appending, including across rollovers.
``final_tokens`` carries the latest raw hypothesis tokens only: in this mode
the text is the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .sentence_commit import (
    SEAM_DEDUP_MAX_WORDS,
    SentenceCommitState,
    SentenceCommitUpdate,
    WordEmergenceTracker,
    seam_overlap,
    update_sentence_commit,
    words_after_committed,
)
from .streamer import (
    CachedFullHypothesisFinal,
    SegmentedCachedFullHypothesisStreamer,
    decode_clean_token_ids,
    join_text_segments,
    trailing_text_words,
)

COMMIT_MODES = ("prefix", "sentence")
# 150 steps x 80 ms = 12 s: keeps the per-chunk re-decode real-time safe on
# the 1.7B model (the 400-step segments fell behind at 1 s chunks).
DEFAULT_SENTENCE_SOFT_STEPS = 150
DEFAULT_SENTENCE_ROLLOVER_MARGIN_SEC = 0.5
_DEFAULT_DECODER_STEP_MS = 80


def validate_sentence_rollover(*, soft_steps: int, max_steps: int, margin_steps: int) -> None:
    if soft_steps < 0:
        raise ValueError("sentence soft limit must be >= 0 steps (0 disables the soft rollover)")
    if margin_steps < 0:
        raise ValueError("sentence rollover margin must be >= 0")
    if max_steps > 0 and soft_steps >= max_steps:
        raise ValueError(
            f"sentence soft limit ({soft_steps} steps) must be below the segment "
            f"hard limit ({max_steps} steps)"
        )


@dataclass
class SentenceSegmentedStreamer(SegmentedCachedFullHypothesisStreamer):
    sentence_soft_steps: int = DEFAULT_SENTENCE_SOFT_STEPS
    sentence_rollover_margin_steps: int = 6
    seam_dedup_max_words: int = SEAM_DEDUP_MAX_WORDS
    sentence_state: SentenceCommitState = field(default_factory=SentenceCommitState)
    emergence: WordEmergenceTracker = field(default_factory=WordEmergenceTracker)
    # Tail of the completed text that the active segment's audio re-transcribes.
    seam_tail_words: list[str] = field(default_factory=list)
    # Latest hypothesis of the active segment with the seam overlap removed.
    segment_words: list[str] = field(default_factory=list)
    # Uncommitted words carried by a soft rollover, until the next decode.
    carried_words: list[str] = field(default_factory=list)
    # Stream-relative audio time (s) where the committed text ends.
    committed_end_sec: float = 0.0
    _rollover_step: int = field(default=0, repr=False)
    _committed_end_index: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_sentence_rollover(
            soft_steps=int(self.sentence_soft_steps),
            max_steps=int(self.segment_max_cached_steps),
            margin_steps=int(self.sentence_rollover_margin_steps),
        )
        if self.seam_dedup_max_words < 0:
            raise ValueError("seam_dedup_max_words must be >= 0")

    @property
    def step_sec(self) -> float:
        config = getattr(self.model, "config", None)
        return float(getattr(config, "decoder_step_ms", _DEFAULT_DECODER_STEP_MS)) / 1000.0

    def update_from_hypothesis(
        self,
        hypothesis_tokens: Sequence[int],
        *,
        audio_sec: float,
        is_flush: bool = False,
        cached_steps: int = 0,
        **diagnostics: int,
    ) -> dict[str, Any]:
        tokens = [int(token_id) for token_id in hypothesis_tokens]
        text = decode_clean_token_ids(
            self.tokenizer,
            tokens,
            wait_token_id=self.config.wait_token_id,
            word_start_token_id=self.config.word_start_token_id,
        )
        self.last_hypothesis_tokens = tokens
        self.last_hypothesis_text = text
        raw_words = text.split()
        self.emergence.update(raw_words, int(cached_steps))
        drop = seam_overlap(self.seam_tail_words, raw_words, self.seam_dedup_max_words)
        self.segment_words = raw_words[drop:]
        self.carried_words = []
        update = update_sentence_commit(
            self.sentence_state,
            self.segment_words,
            stable_iterations=self.config.stable_iterations,
            allow_commit=float(audio_sec) >= self.config.min_commit_audio_sec,
        )
        self._committed_end_index = (
            None if update.committed_end is None else drop + update.committed_end
        )
        if update.delta_words and self._committed_end_index is not None:
            self._record_commit_boundary(self._committed_end_index - 1)

        event = self._build_event(
            update,
            tokens,
            drop=drop,
            audio_sec=audio_sec,
            is_flush=is_flush,
            cached_steps=int(cached_steps),
            diagnostics=diagnostics,
        )
        self._maybe_roll(event, cached_steps=int(cached_steps), is_flush=is_flush)
        self.events.append(event)
        self.events_total += 1
        return event

    def _record_commit_boundary(self, word_index: int) -> None:
        """Remember where the last committed word ends in the active segment."""
        lower, upper = self.emergence.end_bounds(word_index)
        self._rollover_step = max(0, lower - int(self.sentence_rollover_margin_steps))
        estimate = float(upper) if lower == 0 else (lower + upper) / 2.0
        self.committed_end_sec = max(
            self.committed_end_sec,
            (self.dropped_cached_steps_total + estimate) * self.step_sec,
        )

    def _build_event(
        self,
        update: SentenceCommitUpdate,
        tokens: list[int],
        *,
        drop: int,
        audio_sec: float,
        is_flush: bool,
        cached_steps: int,
        diagnostics: dict[str, int],
    ) -> dict[str, Any]:
        segment_committed = " ".join(self.sentence_state.committed_words)
        segment_unstable = " ".join(update.unstable_words)
        segment_hypothesis = " ".join(self.segment_words)
        segment_display = join_text_segments(segment_committed, segment_unstable)
        self.last_display_text = segment_display
        self.last_committed_text = segment_committed
        self.last_global_hypothesis_text = join_text_segments(self.completed_text, segment_hypothesis)
        self.last_global_committed_text = join_text_segments(self.completed_text, segment_committed)
        self.last_global_display_text = join_text_segments(self.completed_text, segment_display)
        event: dict[str, Any] = {key: int(value) for key, value in diagnostics.items()}
        event.update(
            {
                "commit_mode": "sentence",
                "is_flush": is_flush,
                "audio_sec": float(audio_sec),
                "cached_steps": cached_steps,
                "hypothesis_tokens": len(tokens),
                "committed_units": len(self.sentence_state.committed_words),
                "delta_units": len(update.delta_words),
                "pending_sentences": update.pending_sentences,
                "seam_overlap_words": drop,
                "sentence_rollover_step": self._rollover_step,
                "committed_end_sec": self.committed_end_sec,
                "segment_index": int(self.segments_finalized),
                "segments_finalized": int(self.segments_finalized),
                "dropped_cached_steps_total": int(self.dropped_cached_steps_total),
                "segment_hypothesis": segment_hypothesis,
                "segment_committed": segment_committed,
                "segment_display": segment_display,
                "segment_unstable": segment_unstable,
                "segment_delta": " ".join(update.delta_words),
                "completed_text": self.completed_text,
                "hypothesis": self.last_global_hypothesis_text,
                "committed": self.last_global_committed_text,
                "display": self.last_global_display_text,
                "unstable": segment_unstable,
                "delta": " ".join(update.delta_words),
                "candidate": self.last_global_committed_text,
                "segment_rollover": False,
            }
        )
        if self._rolled_before_generate:
            event["segment_rolled_before_generate"] = True
            self._rolled_before_generate = False
        return event

    def _maybe_roll(self, event: dict[str, Any], *, cached_steps: int, is_flush: bool) -> None:
        """Sentence-aligned soft rollover first; wholesale cap rollover as the fallback."""
        if (
            not is_flush
            and self.sentence_soft_steps > 0
            and cached_steps >= self.sentence_soft_steps
            and self._rollover_step > 0
        ):
            reason = "sentence"
            segment_final = self._roll_at_sentence_end()
        elif self.segment_max_cached_steps > 0 and cached_steps > self.segment_max_cached_steps:
            reason = "cap"
            segment_final = self.roll_segment()
        else:
            return
        # A cap rollover commits the whole remainder; report it now.
        self.last_global_committed_text = self.completed_text
        event.update(
            {
                "segment_rollover": True,
                "segment_rollover_reason": reason,
                "segment_final_text": segment_final.final_text,
                "segments_finalized": int(self.segments_finalized),
                "dropped_cached_steps_total": int(self.dropped_cached_steps_total),
                "completed_text_after_roll": self.completed_text,
                "active_cached_steps_after_roll": self._active_cached_steps(),
                "committed": self.completed_text,
                "committed_end_sec": self.committed_end_sec,
            }
        )

    def _begin_next_segment(self, segment_text: str) -> None:
        self.completed_text = join_text_segments(self.completed_text, segment_text)
        self.segments_finalized += 1
        self._segment_prompt_template = None
        # Prompt head and audio positions both change: the rolling decoder
        # KV cache is unusable after any rollover.
        if getattr(self.state, "decoder", None) is not None:
            self.state.decoder = None

    def _roll_at_sentence_end(self) -> CachedFullHypothesisFinal:
        """Close the segment after its last committed sentence, carrying the rest."""
        keep_from = min(self._rollover_step, self._active_cached_steps())
        committed_words = list(self.sentence_state.committed_words)
        carried_words = words_after_committed(self.sentence_state, self.segment_words)
        carried_tracker = (
            self.emergence.carried(self._committed_end_index, keep_from)
            if self._committed_end_index is not None
            else WordEmergenceTracker()
        )
        segment_text = " ".join(committed_words)
        self._begin_next_segment(segment_text)
        frame_hidden = self.state.frame_hidden
        self.state.frame_hidden = frame_hidden[:, keep_from:, :]
        self.dropped_cached_steps_total += int(keep_from)
        self._reset_encoder_for_rollover()
        self._reset_active_segment_state()
        self.emergence = carried_tracker
        self.carried_words = carried_words
        self.seam_tail_words = self._completed_tail_words()
        return self._segment_final(segment_text)

    def roll_segment(self) -> CachedFullHypothesisFinal:
        """Wholesale rollover (hard cap / roll-before-generate safety valve)."""
        if self.segment_finalize_mode == "latest":
            segment_words = self._segment_final_words()
        else:
            segment_words = list(self.sentence_state.committed_words)
        segment_text = " ".join(segment_words)
        segment_end_step = self.dropped_cached_steps_total + self._active_cached_steps()
        self._begin_next_segment(segment_text)
        self._trim_cached_audio_window()
        self._reset_encoder_for_rollover()
        self._reset_active_segment_state()
        if segment_words:
            self.committed_end_sec = max(self.committed_end_sec, segment_end_step * self.step_sec)
        if self._active_cached_steps() > 0:
            self.seam_tail_words = self._completed_tail_words()
        return self._segment_final(segment_text)

    def _completed_tail_words(self) -> list[str]:
        return trailing_text_words(self.completed_text, self.seam_dedup_max_words).split()

    def _segment_final(self, segment_text: str) -> CachedFullHypothesisFinal:
        return CachedFullHypothesisFinal(
            final_tokens=list(self.last_hypothesis_tokens),
            final_text=segment_text,
            final_display_text=segment_text,
            stable_committed_text=segment_text,
            last_hypothesis_text=segment_text,
            final_committed_units=len(segment_text.split()),
        )

    def _segment_final_words(self) -> list[str]:
        committed = list(self.sentence_state.committed_words)
        if self.carried_words:
            return committed + list(self.carried_words)
        return committed + words_after_committed(self.sentence_state, self.segment_words)

    def _reset_active_segment_state(self) -> None:
        super()._reset_active_segment_state()
        self.sentence_state = SentenceCommitState()
        self.emergence = WordEmergenceTracker()
        self.segment_words = []
        self.carried_words = []
        self.seam_tail_words = []
        self._rollover_step = 0
        self._committed_end_index = None

    def finalize(self, *, finalize_mode: str = "latest") -> CachedFullHypothesisFinal:
        """Global transcript: completed segments + committed + (latest) remainder.

        Never revises committed text and does not mutate streamer state.
        """
        if finalize_mode not in {"latest", "stable"}:
            raise ValueError("finalize_mode must be 'latest' or 'stable'")
        committed_text = join_text_segments(
            self.completed_text, " ".join(self.sentence_state.committed_words)
        )
        if finalize_mode == "latest":
            final_text = join_text_segments(
                self.completed_text, " ".join(self._segment_final_words())
            )
        else:
            final_text = committed_text
        return CachedFullHypothesisFinal(
            final_tokens=list(self.last_hypothesis_tokens),
            final_text=final_text,
            final_display_text=final_text,
            stable_committed_text=committed_text,
            last_hypothesis_text=join_text_segments(
                self.completed_text, " ".join(self.segment_words or self.carried_words)
            ),
            final_committed_units=len(final_text.split()),
        )
