"""SentenceSegmentedStreamer / online processor behavior on an audio-conditioned fake.

The fake model encodes a word timeline into the cached audio itself: cached
step ``s`` holds the id of the word whose audio ends at stream step ``s``
(0 = no word end). Its decoder transcribes exactly the words whose end step
is in the cache, minus the newest ``hesitancy_steps`` (the real decoder holds
back words at the audio edge). So trimming or carrying the wrong audio at a
rollover really loses or duplicates words, and word timing is ground truth.
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qwen3_asr_causal.online import Qwen3StreamingOnlineProcessor  # noqa: E402
from qwen3_asr_causal.sentence_commit import normalize_word  # noqa: E402
from qwen3_asr_causal.sentence_streamer import SentenceSegmentedStreamer  # noqa: E402
from qwen3_asr_causal.streamer import CachedFullHypothesisConfig  # noqa: E402

SR = 16_000
FRAMES_PER_STEP = 8
STEP_SEC = 0.08

SENTENCES = [
    "Hello there, my friend.",
    "Dr. Smith arrived at 3.5 p.m. today.",
    "We talked about the plan for a while.",
    "Then we left.",
    "It was late, and everyone was tired.",
    "Tomorrow we will continue the work.",
    "Goodbye for now",
]


class Timeline:
    def __init__(self, sentences, word_steps=3, gap_steps=5, tail_steps=12):
        self.vocab = {}
        self.steps = []
        self.word_end_step = {}
        self.sentence_last_ids = []
        for sentence in sentences:
            for word in sentence.split():
                token = len(self.vocab) + 1
                self.vocab[token] = word
                self.steps.extend([0] * (word_steps - 1) + [token])
                self.word_end_step[token] = len(self.steps) - 1
            self.sentence_last_ids.append(len(self.vocab))
            self.steps.extend([0] * gap_steps)
        self.steps.extend([0] * tail_steps)
        self.words = [self.vocab[token] for token in sorted(self.vocab)]
        # Casing/punctuation churn: odd decodes render comma words without
        # the comma and capitalized words in lowercase.
        self.churn = {}
        for token, word in list(self.vocab.items()):
            variant = word.replace(",", "") if "," in word else None
            if variant is None and word[:1].isupper() and not word.endswith("."):
                variant = word.lower()
            if variant is not None:
                variant_id = 1000 + token
                self.vocab[variant_id] = variant
                self.churn[token] = variant_id


class TimelineTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(self.vocab[int(token_id)] for token_id in token_ids)


class TimelineModel:
    def __init__(self, timeline, hesitancy_steps=0, churn=False):
        self.timeline = timeline
        self.hesitancy_steps = hesitancy_steps
        self.churn = churn
        self.cursor = 0
        self.pending_frames = 0
        self.decodes = 0
        self.config = SimpleNamespace(
            mel_hop_ms=10,
            decoder_step_ms=80,
            frames_per_decoder_step=FRAMES_PER_STEP,
            n_mels=128,
        )

    def init_cached_audio_decode_state(self):
        return SimpleNamespace(
            audio=SimpleNamespace(frames_seen=0), frame_hidden=None, decoder=None
        )

    def append_audio_to_cache(self, mels, state):
        frames = self.pending_frames + int(mels.shape[1])
        steps, self.pending_frames = divmod(frames, FRAMES_PER_STEP)
        state.audio.frames_seen += int(mels.shape[1])
        ids = [
            self.timeline.steps[i] if i < len(self.timeline.steps) else 0
            for i in range(self.cursor, self.cursor + steps)
        ]
        self.cursor += steps
        delta = torch.tensor(ids, dtype=torch.float32).reshape(1, steps, 1)
        if state.frame_hidden is None:
            state.frame_hidden = delta
        else:
            state.frame_hidden = torch.cat([state.frame_hidden, delta], dim=1)
        return state.frame_hidden, delta, state

    def generate_full_hypothesis_from_cached_audio(self, frame_hidden, **_kwargs):
        self.decodes += 1
        visible = max(0, int(frame_hidden.shape[1]) - self.hesitancy_steps)
        ids = [int(value) for value in frame_hidden[0, :visible, 0].tolist() if value]
        if self.churn and self.decodes % 2:
            ids = [self.timeline.churn.get(token, token) for token in ids]
        return torch.tensor(ids, dtype=torch.long)


def make_streamer(timeline, *, hesitancy=2, churn=False, soft=40, cap=100,
                  keep_tail=0, margin=6, stable_iterations=2):
    return SentenceSegmentedStreamer(
        TimelineModel(timeline, hesitancy_steps=hesitancy, churn=churn),
        TimelineTokenizer(timeline.vocab),
        CachedFullHypothesisConfig(
            wait_token_id=-1,
            word_start_token_id=-2,
            stable_iterations=stable_iterations,
        ),
        segment_max_cached_steps=cap,
        segment_keep_tail_steps=keep_tail,
        sentence_soft_steps=soft,
        sentence_rollover_margin_steps=margin,
    )


def feed(streamer, steps_per_chunk, total_steps):
    events = []
    fed = 0
    while fed < total_steps:
        steps = min(steps_per_chunk, total_steps - fed)
        events.append(streamer.append_mel_chunk(torch.zeros(1, steps * FRAMES_PER_STEP, 128)))
        fed += steps
    return events


def norm_words(text):
    return [normalize_word(word) for word in text.split()]


def test_every_word_is_transcribed_once_across_sentence_rollovers():
    timeline = Timeline(SENTENCES)
    streamer = make_streamer(timeline, churn=True)

    events = feed(streamer, 12, len(timeline.steps))
    final = streamer.finalize()

    reasons = [event.get("segment_rollover_reason") for event in events if event["segment_rollover"]]
    assert reasons.count("sentence") >= 3
    assert "cap" not in reasons
    assert norm_words(final.final_text) == norm_words(" ".join(timeline.words))
    assert any(event["seam_overlap_words"] > 0 for event in events)


def test_commits_are_whole_sentences_and_append_only():
    timeline = Timeline(SENTENCES)
    streamer = make_streamer(timeline, churn=True)
    boundaries = set()
    position = 0
    for sentence in SENTENCES:
        position += len(sentence.split())
        boundaries.add(position)

    events = feed(streamer, 12, len(timeline.steps))

    previous = []
    for event in events:
        committed = event["committed"].split()
        assert committed[: len(previous)] == previous
        if len(committed) > len(previous):
            assert len(committed) in boundaries
        previous = committed
    assert previous, "sentence mode committed nothing before the flush"


def test_rollover_carries_audio_from_before_the_committed_sentence_end():
    timeline = Timeline(SENTENCES)
    streamer = make_streamer(timeline, churn=False)
    end_step = {word_index: timeline.word_end_step[word_index + 1] for word_index in range(len(timeline.words))}

    rolls = 0
    fed = 0
    while fed < len(timeline.steps):
        steps = min(12, len(timeline.steps) - fed)
        committed_before = len(streamer.last_global_committed_text.split())
        event = streamer.append_mel_chunk(torch.zeros(1, steps * FRAMES_PER_STEP, 128))
        fed += steps
        if event.get("segment_rollover_reason") != "sentence":
            continue
        rolls += 1
        carried_from = streamer.dropped_cached_steps_total
        committed = len(event["committed"].split())
        assert committed >= committed_before
        # The cut sits before the last committed word's end (re-decoded,
        # then de-duplicated) and therefore before every uncommitted word.
        assert carried_from <= end_step[committed - 1]
        for word_index in range(committed, len(timeline.words)):
            if end_step[word_index] < fed:
                assert end_step[word_index] >= carried_from
    assert rolls >= 3


def test_cap_fallback_without_punctuation_dedups_the_kept_tail():
    timeline = Timeline(["one two three four five six seven eight nine ten eleven twelve "
                         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"])
    streamer = make_streamer(timeline, hesitancy=0, soft=20, cap=30, keep_tail=6)

    events = feed(streamer, 6, len(timeline.steps))
    final = streamer.finalize()

    reasons = [event.get("segment_rollover_reason") for event in events if event["segment_rollover"]]
    assert reasons and set(reasons) == {"cap"}
    assert final.final_text.split() == timeline.words


def test_finalize_right_after_a_soft_roll_keeps_the_carried_sentence():
    timeline = Timeline(SENTENCES)
    streamer = make_streamer(timeline)

    fed = 0
    while True:
        event = streamer.append_mel_chunk(torch.zeros(1, 12 * FRAMES_PER_STEP, 128))
        fed += 12
        if event.get("segment_rollover_reason") == "sentence":
            break
    final = streamer.finalize()

    visible = [
        word for token, word in sorted(timeline.vocab.items())
        if token < 1000 and timeline.word_end_step[token] < fed - 2
    ]
    assert final.final_text.split() == visible


def test_soft_limit_must_stay_below_the_hard_cap():
    timeline = Timeline(SENTENCES)

    with pytest.raises(ValueError, match="soft limit"):
        make_streamer(timeline, soft=100, cap=100)


# --- online processor -------------------------------------------------------


class FakeMelExtractor:
    """One mel frame per 160 samples; keeps the sub-frame remainder."""

    def __init__(self):
        self.pending = 0

    def append(self, audio):
        samples = self.pending + len(audio)
        frames, self.pending = divmod(samples, 160)
        return torch.zeros(1, frames, 128) if frames else None

    def flush(self):
        return None

    def reset(self):
        self.pending = 0


class TimelineASR:
    sep = ""
    SAMPLING_RATE = SR

    def __init__(self, timeline):
        self.timeline = timeline
        self.original_language = "en"
        self.chunk_sec = 1.0
        self.device = torch.device("cpu")
        self.right_context_frames = 0
        self.n_mels = 128
        self.decode_lock = threading.Lock()

    def build_streamer(self, language=None):
        return make_streamer(self.timeline, churn=True)

    def new_mel_extractor(self):
        return FakeMelExtractor()


def test_online_sentence_mode_emits_sentences_with_audio_anchored_monotonic_times():
    timeline = Timeline(SENTENCES)
    processor = Qwen3StreamingOnlineProcessor(TimelineASR(timeline))
    total_sec = len(timeline.steps) * STEP_SEC
    packet_sec = 0.25

    batches = []
    sent = 0.0
    while sent < total_sec - 1e-9:
        seconds = min(packet_sec, total_sec - sent)
        sent += seconds
        processor.insert_audio_chunk(np.zeros(int(round(seconds * SR)), dtype=np.float32), sent)
        tokens, _ = processor.process_iter()
        if tokens:
            batches.append(tokens)
    final_tokens, _ = processor.finish()

    streamed = [token for batch in batches for token in batch]
    all_tokens = streamed + final_tokens
    assert norm_words("".join(token.text for token in all_tokens)) == norm_words(" ".join(timeline.words))
    for batch in batches:
        assert batch[-1].text.rstrip().endswith((".", "?", "!"))
    previous_end = 0.0
    for token in all_tokens:
        assert previous_end - 1e-9 <= token.start <= token.end
        previous_end = token.end
    # The end of each streamed sentence lands within a second of its audio.
    emitted = 0
    for batch in batches:
        emitted += len(batch)
        true_end = (timeline.word_end_step[emitted] + 1) * STEP_SEC
        assert abs(batch[-1].end - true_end) <= 1.0
