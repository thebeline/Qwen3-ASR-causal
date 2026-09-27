"""Sentence-granular commit policy for the qwen3-streaming backend.

Pure text/bookkeeping logic (no torch) used by ``SentenceSegmentedStreamer``:

* ``split_sentences``: split a word list at terminal punctuation with a small
  abbreviation/initialism heuristic.
* ``update_sentence_commit``: commit whole sentences once they are complete
  (followed by at least one more word) and unchanged across the last
  ``stable_iterations`` decodes.
* ``seam_overlap``: count the leading words of a carried-over segment's
  hypothesis that re-transcribe the tail of already-committed text.
* ``WordEmergenceTracker``: per-word audio-step bounds derived from the decode
  at which each word first appeared, used to place a rollover at a sentence
  end without cutting the next sentence.

Words are whitespace-separated units; languages written without spaces are
not segmented by this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Sequence

from .stable_commit import normalize_text_unit_for_match

_TERMINAL_CHARS = ".?!…"
_CLOSING_CHARS = "\"'»”’)]}"
_OPENING_CHARS = "\"'«“‘([{"
# Titles and short forms that end in a period but almost never end a
# sentence in transcribed speech. Kept small on purpose: a missed boundary
# only delays a commit, a false boundary commits half a sentence.
ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs",
        "cf", "approx", "gen", "gov", "sen", "rep", "capt", "lt", "col",
        "sgt", "fig",
    }
)
# "U.S.", "e.g.", "a.m.": two or more letter-period pairs.
_INITIALISM_RE = re.compile(r"^(?:[A-Za-z]\.){2,}$")
# "J." in "J. Smith"; "I." and "A." are real sentence endings.
_SINGLE_INITIAL_RE = re.compile(r"^[B-HJ-Z]\.$")

# Longest seam overlap searched for. The cut sits up to one decode interval
# plus the margin before the committed end: ~1.5 s at 1 s chunks, ~3 s when
# decode pacing stretches the interval. At ~4 words/s that is 6-12 words.
SEAM_DEDUP_MAX_WORDS = 16
# Committed words that may go unmatched at the end of the committed span when
# re-locating it inside a revised hypothesis.
_MAX_UNMATCHED_COMMITTED_TAIL = 2


def normalize_word(word: str) -> str:
    """Lowercase and strip edge punctuation: the unit compared across decodes."""
    return normalize_text_unit_for_match(word, case_sensitive=False)


def sentence_key(words: Sequence[str]) -> str:
    """Comparison key for sentence stability.

    Word identity only: casing and punctuation inside the sentence are
    ignored because the decoder re-punctuates earlier text on almost every
    pass, which is what kept the word-prefix policy from committing. The
    sentence boundary itself is structural (it decides which words belong to
    the sentence), so a moved boundary still changes the key.
    """
    return " ".join(normalize_word(word) for word in words)


def _starts_lowercase(word: str) -> bool:
    stripped = word.lstrip(_OPENING_CHARS)
    return bool(stripped) and stripped[0].isalpha() and stripped[0].islower()


def is_sentence_end(word: str, next_word: str | None) -> bool:
    """Whether ``word`` closes a sentence, given the word that follows it.

    Heuristic:
    * the word, minus closing quotes/brackets, ends in ``. ? ! …``;
    * an ellipsis followed by a lowercase word is a pause, not an end
      ("was... thinking"). Only ellipses: the decoder flips the casing of
      the next word between passes, and a boundary that depends on it would
      never stay stable long enough to commit;
    * a single period after a known abbreviation ("Dr."), an initialism
      ("U.S.", "e.g.") or a single capital initial ("J.") is not an end;
    * decimals ("3.5") never qualify: the period is not at the word's end.
    """
    core = word.rstrip(_CLOSING_CHARS)
    if not core or core[-1] not in _TERMINAL_CHARS:
        return False
    if core.endswith(("...", "…")):
        return next_word is None or not _starts_lowercase(next_word)
    if core[-1] != ".":
        return True
    stem = core[:-1].lstrip(_OPENING_CHARS).lower()
    if stem in ABBREVIATIONS:
        return False
    bare = core.lstrip(_OPENING_CHARS)
    return not (_INITIALISM_RE.match(bare) or _SINGLE_INITIAL_RE.match(bare))


def split_sentences(words: Sequence[str]) -> list[list[str]]:
    """Group words into sentences; the last group may be unterminated."""
    sentences: list[list[str]] = []
    current: list[str] = []
    for idx, word in enumerate(words):
        current.append(word)
        next_word = words[idx + 1] if idx + 1 < len(words) else None
        if is_sentence_end(word, next_word):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def locate_committed_end(
    committed: Sequence[str],
    hypothesis: Sequence[str],
) -> int | None:
    """Index in ``hypothesis`` just past the already-committed words.

    Both inputs are normalized words. The fast path is an exact prefix match.
    When the decoder revised committed words (a substitution, a merge such as
    "can not" -> "cannot"), the committed span is re-located by sequence
    alignment. Returns None when the alignment is too weak to trust: fewer
    than half the committed words match, or more than
    ``_MAX_UNMATCHED_COMMITTED_TAIL`` trailing committed words are unmatched.
    """
    count = len(committed)
    if count == 0:
        return 0
    if list(hypothesis[:count]) == list(committed):
        return count
    blocks = [
        block
        for block in SequenceMatcher(
            None, list(committed), list(hypothesis), autojunk=False
        ).get_matching_blocks()
        if block.size
    ]
    if not blocks or 2 * sum(block.size for block in blocks) < count:
        return None
    last = blocks[-1]
    unmatched_tail = count - (last.a + last.size)
    if unmatched_tail > _MAX_UNMATCHED_COMMITTED_TAIL:
        return None
    end = last.b + last.size + unmatched_tail
    return end if end <= len(hypothesis) else None


def seam_overlap(
    committed_tail: Sequence[str],
    hypothesis: Sequence[str],
    max_words: int = SEAM_DEDUP_MAX_WORDS,
) -> int:
    """Leading hypothesis words that repeat the tail of committed text.

    A carried-over segment starts slightly before the committed sentence end,
    so its hypothesis opens with a few already-committed words. Rule: the
    largest k <= ``max_words`` such that the first k hypothesis words equal
    the last k committed words (normalized). If no exact overlap exists, the
    first hypothesis word may be a mis-heard piece of the committed word whose
    audio the cut went through: accepted when the remaining k-1 words match
    exactly and either the piece is a 2+ character suffix of that committed
    word ("tence." for "sentence.") or at least 2 words back it up.
    Exact overlaps win over cut-word overlaps.
    """
    tail = [normalize_word(word) for word in committed_tail[-max_words:]]
    head = [normalize_word(word) for word in hypothesis[:max_words]]
    limit = min(len(tail), len(head))
    for k in range(limit, 0, -1):
        if head[:k] == tail[-k:]:
            return k
    for k in range(limit, 0, -1):
        if head[1:k] != tail[len(tail) - k + 1 :]:
            continue
        fragment, anchor = head[0], tail[-k]
        is_suffix = 2 <= len(fragment) < len(anchor) and anchor.endswith(fragment)
        if is_suffix or k - 1 >= 2:
            return k
    return 0


@dataclass
class SentenceCommitState:
    committed_words: list[str] = field(default_factory=list)
    # Per decode: keys of the complete, uncommitted sentences after the
    # committed boundary, newest last. Bounded to ``stable_iterations``.
    history: list[list[str]] = field(default_factory=list)


@dataclass(frozen=True)
class SentenceCommitUpdate:
    delta_words: list[str]
    # Index in the hypothesis just past the committed words, or None when the
    # committed span could not be located in this hypothesis.
    committed_end: int | None
    unstable_words: list[str]
    pending_sentences: int


def _common_prefix_length(lists: Sequence[Sequence[str]]) -> int:
    limit = min(len(items) for items in lists)
    newest = lists[-1]
    idx = 0
    while idx < limit and all(items[idx] == newest[idx] for items in lists):
        idx += 1
    return idx


def update_sentence_commit(
    state: SentenceCommitState,
    hypothesis_words: Sequence[str],
    *,
    stable_iterations: int,
    allow_commit: bool = True,
) -> SentenceCommitUpdate:
    """Commit complete sentences that stayed identical for ``stable_iterations`` decodes.

    A sentence is complete when terminal punctuation closes it and at least
    one more word follows, so the last sentence of a hypothesis is never
    committed here (the segment flush commits it). Committed words are never
    revised: later hypotheses are aligned past them and only the remainder is
    examined. When the committed span cannot be located, nothing commits and
    the stability history restarts.
    """
    if stable_iterations <= 0:
        raise ValueError("stable_iterations must be > 0")
    words = list(hypothesis_words)
    committed = state.committed_words
    offset = locate_committed_end(
        [normalize_word(word) for word in committed],
        [normalize_word(word) for word in words],
    )
    if offset is None:
        state.history.clear()
        return SentenceCommitUpdate(
            delta_words=[],
            committed_end=None,
            unstable_words=words[len(committed) :],
            pending_sentences=0,
        )

    complete = split_sentences(words[offset:])[:-1]
    state.history.append([sentence_key(sentence) for sentence in complete])
    del state.history[:-stable_iterations]

    stable = 0
    if allow_commit and len(state.history) >= stable_iterations:
        stable = _common_prefix_length(state.history)
    delta = [word for sentence in complete[:stable] for word in sentence]
    if stable:
        state.committed_words = committed + delta
        state.history = [keys[stable:] for keys in state.history]
    end = offset + len(delta)
    return SentenceCommitUpdate(
        delta_words=delta,
        committed_end=end,
        unstable_words=words[end:],
        pending_sentences=len(complete) - stable,
    )


def words_after_committed(
    state: SentenceCommitState,
    hypothesis_words: Sequence[str],
) -> list[str]:
    """Uncommitted remainder of a hypothesis (positional if alignment fails)."""
    words = list(hypothesis_words)
    offset = locate_committed_end(
        [normalize_word(word) for word in state.committed_words],
        [normalize_word(word) for word in words],
    )
    if offset is None:
        offset = min(len(state.committed_words), len(words))
    return words[offset:]


@dataclass
class WordEmergenceTracker:
    """Audio-step bounds for each word of the latest hypothesis.

    For every word the tracker keeps the active-segment step count of the
    decode where the word first appeared (``first_seen``) and of the decode
    just before it (``last_absent``). Words are carried across decodes by
    sequence alignment of normalized words, so a revision elsewhere in the
    hypothesis keeps the history of unchanged words.

    Interpretation: a word cannot be transcribed before its audio is cached,
    so it ends at or before ``first_seen`` (and before any later word's
    ``first_seen``, words being in audio order). A word that was absent with
    ``last_absent`` steps cached had usually not finished by then: the decoder
    emits a fully heard word, but may hold back the last few hundred ms,
    which the rollover margin absorbs.
    """

    words: list[str] = field(default_factory=list)
    first_seen: list[int] = field(default_factory=list)
    last_absent: list[int] = field(default_factory=list)
    previous_steps: int = 0

    def update(self, words: Sequence[str], steps: int) -> None:
        normalized = [normalize_word(word) for word in words]
        first_seen = [int(steps)] * len(normalized)
        last_absent = [int(self.previous_steps)] * len(normalized)
        if self.words and normalized:
            matcher = SequenceMatcher(None, self.words, normalized, autojunk=False)
            for block in matcher.get_matching_blocks():
                for k in range(block.size):
                    first_seen[block.b + k] = self.first_seen[block.a + k]
                    last_absent[block.b + k] = self.last_absent[block.a + k]
        self.words = normalized
        self.first_seen = first_seen
        self.last_absent = last_absent
        self.previous_steps = int(steps)

    def end_bounds(self, index: int) -> tuple[int, int]:
        """(lower, upper) step estimate for where word ``index`` ends."""
        if not 0 <= index < len(self.words):
            raise IndexError(f"word index {index} outside hypothesis of {len(self.words)}")
        upper = min(self.first_seen[index:])
        lower = min(self.last_absent[index], upper)
        return lower, upper

    def carried(self, start_index: int, dropped_steps: int) -> "WordEmergenceTracker":
        """Tracker for words ``start_index:`` after ``dropped_steps`` leave the cache."""
        start = max(0, min(int(start_index), len(self.words)))
        shift = int(dropped_steps)
        return WordEmergenceTracker(
            words=self.words[start:],
            first_seen=[max(0, step - shift) for step in self.first_seen[start:]],
            last_absent=[max(0, step - shift) for step in self.last_absent[start:]],
            previous_steps=max(0, self.previous_steps - shift),
        )
