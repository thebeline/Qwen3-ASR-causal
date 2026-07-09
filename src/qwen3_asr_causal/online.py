"""Per-session online processor for the qwen3-streaming backend.

State machine per WebSocket session::

    insert_audio_chunk -> pending sample buffer (cheap, no GPU work)
    process_iter       -> when enough audio is pending, featurize incrementally,
                          append mel to the streamer (one bounded full-hypothesis
                          decode), then emit newly committed words as ASRTokens
    get_buffer         -> the unstable hypothesis tail (display-only)
    start_silence/finish -> flush mel tail + right context, finalize, emit rest

Word timestamps are linear interpolations across each newly committed span
(the streamer is text-only). Typical error is on the order of a second:
fine for line/diarization alignment, not for precise word timing; use the
``qwen3-vllm`` backend (ForcedAligner) when exact timestamps matter.

Decode pacing is self-adjusting: each decode must "pay for itself", so the
next one waits for at least ``_PACING x`` the previous decode duration in new
audio. On hardware slower than the audio rate the chunks grow until the
per-audio-second cost amortizes, instead of lag-spiraling.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import List, Tuple

import numpy as np

from .types import ASRToken, Transcript

logger = logging.getLogger(__name__)

_AVG_WORDS_PER_SECOND = 2.4
_DECODE_OVERHEAD_SECONDS = 0.3


def estimate_commit_lag_seconds(
    *,
    chunk_sec: float,
    hold_back_words: int,
    stable_iterations: int,
    right_context_sec: float = 0.0,
) -> float:
    """Expected distance between the audio head and a word being committed.

    Structural model validated against the 21-file MCIF per-word replay
    (2026-07-07): holdback words at ~2.4 words/s, plus one extra decode
    cadence per additional stability iteration, plus half a cadence of
    phase, plus decode time and any right context. Predicted/measured p50:
    6w/2it 5.68/5.89 s, 6w/1it 3.76/3.98 s, 2w/1it 2.09/2.13 s.
    """
    return (
        hold_back_words / _AVG_WORDS_PER_SECOND
        + chunk_sec * max(0, stable_iterations - 1)
        + chunk_sec / 2.0
        + _DECODE_OVERHEAD_SECONDS
        + right_context_sec
    )


class Qwen3StreamingOnlineProcessor:
    SAMPLING_RATE = 16_000
    _PACING = 1.2
    _MIN_WORD_SECONDS = 0.05

    def __init__(self, asr, logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        # Used to back-date committed-word timestamps; must track the commit
        # policy or word timing skews with it. The windowed backend keeps its
        # historical constant (its hypotheses churn differently).
        if getattr(asr, "audio_backend", "windowed") == "causal":
            self._commit_lag_seconds = estimate_commit_lag_seconds(
                chunk_sec=float(getattr(asr, "chunk_sec", 2.0)),
                hold_back_words=int(getattr(asr, "hold_back_words", 6)),
                stable_iterations=int(getattr(asr, "stable_iterations", 1)),
            )
        else:
            self._commit_lag_seconds = 2.5
        session_language = getattr(asr, "_session_language", None)
        self._language = session_language or asr.original_language
        self._detected_language = (
            self._language if self._language != "auto" else None
        )

        self.streamer = asr.build_streamer(self._language)
        self.mel = asr.new_mel_extractor()

        self.end = 0.0
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer = []

        self._emitted_words: list[str] = []
        self._any_word_emitted = False
        self._last_commit_time = 0.0
        self._last_event: dict | None = None
        self._last_decode_duration = 0.0

    # ------------------------------------------------------------------
    # audio_processor contract
    # ------------------------------------------------------------------

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: float):
        self.end = audio_stream_end_time
        self.audio_buffer = np.append(self.audio_buffer, audio.astype(np.float32))

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        try:
            if is_last:
                return self._flush(), self.end
            pending_sec = len(self.audio_buffer) / self.SAMPLING_RATE
            due_after = max(self.asr.chunk_sec, self._PACING * self._last_decode_duration)
            if pending_sec < due_after:
                return [], self.end
            event = self._decode_pending()
            if event is None:
                return [], self.end
            return self._emit_committed(event["committed"], self.end), self.end
        except Exception as exc:
            logger.warning("[qwen3-streaming] process_iter error: %s", exc, exc_info=True)
            return [], self.end

    def get_buffer(self) -> Transcript:
        unstable = (self._last_event or {}).get("unstable", "")
        if not unstable:
            return Transcript(start=None, end=None, text="")
        return Transcript(start=self._last_commit_time, end=self.end, text=unstable)

    def start_silence(self) -> Tuple[List[ASRToken], float]:
        tokens = self._flush()
        logger.info("[qwen3-streaming] start_silence: flushed %d words", len(tokens))
        self._reset_for_next_utterance()
        return tokens, self.end

    def end_silence(self, silence_duration: float, offset: float):
        self.end += silence_duration
        self._last_commit_time += silence_duration

    def new_speaker(self, change_speaker):
        self.start_silence()

    def warmup(self, audio, init_prompt=""):
        return None

    def finish(self) -> Tuple[List[ASRToken], float]:
        tokens = self._flush()
        logger.info("[qwen3-streaming] finish: flushed %d words", len(tokens))
        return tokens, self.end

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    # Largest audio span appended in one decode. Catch-up after a stall (file
    # transcription, REST endpoint) must not grow appends without bound: a
    # single 70 s append prefills ~875 decoder steps: far past the ~200-step
    # segment cap the rollover assumes: with transient allocations that can
    # pin the accelerator at its memory ceiling and turn decodes into a
    # tens-of-seconds death spiral (observed on MPS at 42 GB). Six chunks
    # (12 s at the 2 s default) stays under the segment cap while still
    # amortizing catch-up decodes.
    _MAX_APPEND_CHUNKS = 6

    def _decode_pending(self) -> dict | None:
        """Featurize pending audio and run bounded streamer updates."""
        audio = self.audio_buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        max_slice = max(
            1,
            int(self._MAX_APPEND_CHUNKS * self.asr.chunk_sec * self.SAMPLING_RATE),
        )
        event = None
        started = time.perf_counter()
        for start in range(0, len(audio), max_slice):
            frames = self.mel.append(audio[start : start + max_slice])
            if frames is None or frames.shape[1] == 0:
                continue
            with self.asr.decode_lock:
                event = self.streamer.append_mel_chunk(frames.to(self.asr.device))
                self._release_mps_allocator_cache()
        self._last_decode_duration = time.perf_counter() - started
        if event is None:
            return None
        self._last_event = event
        return event

    def _release_mps_allocator_cache(self) -> None:
        """Return MPS cached blocks to the driver when they pile up.

        Decode pacing coalesces pending audio into variable-size appends, so
        consecutive prefills rarely share tensor shapes. The MPS caching
        allocator buckets by shape and never releases memory on its own:
        faster-than-realtime feeding (file transcription, catch-up) grows
        driver memory by GBs per audio-minute while live tensors stay ~2 GB.
        """
        if getattr(self.asr.device, "type", "") != "mps":
            return
        import torch

        driver = torch.mps.driver_allocated_memory()
        allocated = torch.mps.current_allocated_memory()
        try:
            ceiling = float(torch.mps.recommended_max_memory())
        except Exception:
            ceiling = 0.0
        threshold = max(2 * (1024**3), 0.08 * ceiling)
        if driver > threshold and driver > 1.5 * allocated:
            torch.mps.empty_cache()

    def _flush(self) -> List[ASRToken]:
        """Flush mel tail and right context, finalize the active segment."""
        audio = self.audio_buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        with self.asr.decode_lock:
            frames = self.mel.append(audio)
            if frames is not None and frames.shape[1] > 0:
                self.streamer.append_mel_chunk(frames.to(self.asr.device))
            tail = self.mel.flush()
            if tail is not None and tail.shape[1] > 0:
                self.streamer.append_mel_chunk(tail.to(self.asr.device))
            # Causal backend: encode the partial attention block still
            # buffered in the encoder (no-op for the windowed backend, which
            # is flushed by the right-context zeros below instead).
            flush_pending = getattr(self.streamer, "flush_pending_audio", None)
            if flush_pending is not None:
                flush_pending()
            right_context_frames = self.asr.right_context_frames
            if right_context_frames > 0 and self.streamer.events:
                import torch

                zeros = torch.zeros(
                    1, right_context_frames, self.asr.n_mels, device=self.asr.device
                )
                self.streamer.append_mel_chunk(zeros, is_flush=True)
            final = self.streamer.finalize(finalize_mode="latest")
            self._release_mps_allocator_cache()
        self._last_event = None
        return self._emit_committed(final.final_text, self.end, flush=True)

    def _reset_for_next_utterance(self):
        self.streamer = self.asr.build_streamer(self._language)
        self.mel.reset()
        self._emitted_words = []
        self._last_event = None
        self._last_decode_duration = 0.0
        self._last_commit_time = self.end

    def _emit_committed(
        self, committed_text: str, event_time: float, flush: bool = False
    ) -> List[ASRToken]:
        """Diff global committed text against already-emitted words.

        Output is append-only: if a segment-rollover finalization revised
        already-emitted words, the revision is dropped (logged) and the new
        text becomes the diff baseline.
        """
        target = committed_text.split()
        n_emitted = len(self._emitted_words)
        if len(target) < n_emitted:
            logger.debug(
                "[qwen3-streaming] committed text shrank (%d -> %d words); keeping baseline",
                n_emitted,
                len(target),
            )
            return []
        if target[:n_emitted] != self._emitted_words:
            logger.debug(
                "[qwen3-streaming] %d already-emitted words were revised at a segment "
                "boundary; revision dropped from output",
                n_emitted,
            )
        new_words = target[n_emitted:]
        self._emitted_words = target
        if not new_words:
            return []

        t0 = self._last_commit_time
        t1 = event_time if flush else event_time - self._commit_lag_seconds
        t1 = min(max(t1, t0 + self._MIN_WORD_SECONDS * len(new_words)), self.end)
        if t1 <= t0:
            t1 = min(t0 + self._MIN_WORD_SECONDS * len(new_words), self.end)
            t1 = max(t1, t0)

        tokens: list[ASRToken] = []
        span = t1 - t0
        for idx, word in enumerate(new_words):
            start = t0 + span * idx / len(new_words)
            end = t0 + span * (idx + 1) / len(new_words)
            text = word if not self._any_word_emitted else " " + word
            self._any_word_emitted = True
            tokens.append(
                ASRToken(
                    start=start,
                    end=end,
                    text=text,
                    detected_language=self._detected_language,
                )
            )
        self._last_commit_time = t1
        return tokens
