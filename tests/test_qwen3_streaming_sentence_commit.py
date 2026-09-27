"""Sentence splitting, sentence-stability commits, seam de-duplication."""

import pytest

from qwen3_asr_causal.sentence_commit import (
    SentenceCommitState,
    WordEmergenceTracker,
    is_sentence_end,
    locate_committed_end,
    seam_overlap,
    split_sentences,
    update_sentence_commit,
    words_after_committed,
)


def sentences(text):
    return [" ".join(group) for group in split_sentences(text.split())]


# --- splitter ---------------------------------------------------------------


def test_split_keeps_abbreviations_and_initialisms_inside_sentences():
    assert sentences("Dr. Smith met Mr. Jones in the U.S. today. Then J. Doe left.") == [
        "Dr. Smith met Mr. Jones in the U.S. today.",
        "Then J. Doe left.",
    ]


def test_split_ignores_decimals_and_lowercase_continuations():
    assert sentences("It costs 3.5 dollars, e.g. the small one. Fine") == [
        "It costs 3.5 dollars, e.g. the small one.",
        "Fine",
    ]


def test_split_on_question_exclamation_and_ellipsis():
    assert sentences("Why? Because! I was… Okay... Next") == [
        "Why?",
        "Because!",
        "I was…",
        "Okay...",
        "Next",
    ]


def test_trailing_ellipsis_before_lowercase_does_not_split():
    assert sentences("I was… thinking about it... and then.") == [
        "I was… thinking about it... and then."
    ]


def test_period_boundary_does_not_depend_on_next_word_casing():
    # Casing of the next word churns between decodes; the boundary must not.
    assert sentences("We left. we came back") == ["We left.", "we came back"]
    assert sentences("We left. We came back") == ["We left.", "We came back"]


def test_split_handles_closing_quotes_and_brackets():
    assert sentences('He said "stop." She left (quickly.) Then silence') == [
        'He said "stop."',
        "She left (quickly.)",
        "Then silence",
    ]


def test_single_letter_pronoun_and_article_do_end_sentences():
    assert sentences("So do I. Plan A. Go") == ["So do I.", "Plan A.", "Go"]


def test_missing_final_punctuation_leaves_an_open_last_group():
    assert sentences("One two. Three four") == ["One two.", "Three four"]
    assert sentences("no punctuation at all") == ["no punctuation at all"]
    assert split_sentences([]) == []


def test_is_sentence_end_at_hypothesis_end_uses_punctuation_only():
    assert is_sentence_end("done.", None)
    assert not is_sentence_end("Dr.", None)
    assert not is_sentence_end("done", None)


# --- sentence-stability commit ----------------------------------------------


def run(state, hypothesis, stable_iterations=2, allow_commit=True):
    return update_sentence_commit(
        state,
        hypothesis.split(),
        stable_iterations=stable_iterations,
        allow_commit=allow_commit,
    )


def test_complete_sentence_commits_after_it_repeats():
    state = SentenceCommitState()

    first = run(state, "Hello there. How are")
    second = run(state, "Hello there. How are you")

    assert first.delta_words == []
    assert second.delta_words == ["Hello", "there."]
    assert second.unstable_words == ["How", "are", "you"]
    assert second.committed_end == 2


def test_punctuation_and_casing_churn_inside_a_sentence_does_not_block_commit():
    state = SentenceCommitState()

    run(state, "Hello, there my friend. We")
    update = run(state, "hello there, my friend. We went")

    # Word identity is stable; the latest rendering is what gets committed.
    assert update.delta_words == ["hello", "there,", "my", "friend."]


def test_changed_words_restart_stability():
    state = SentenceCommitState()

    run(state, "I saw the cat. It")
    changed = run(state, "I saw the hat. It")
    stable = run(state, "I saw the hat. It ran")

    assert changed.delta_words == []
    assert stable.delta_words == ["I", "saw", "the", "hat."]


def test_moved_sentence_boundary_is_a_change():
    state = SentenceCommitState()

    run(state, "Wait. For me please")
    moved = run(state, "Wait for me. Please")

    assert moved.delta_words == []


def test_last_sentence_is_never_committed_even_when_terminated():
    state = SentenceCommitState()

    for _ in range(4):
        update = run(state, "One sentence. Two sentences.")

    assert state.committed_words == ["One", "sentence."]
    assert update.unstable_words == ["Two", "sentences."]
    assert update.pending_sentences == 0


def test_several_sentences_commit_together_and_history_rebases():
    state = SentenceCommitState()

    run(state, "A b. C d. E")
    both = run(state, "A b. C d. E f")
    next_one = run(state, "A b. C d. E f. G")
    after = run(state, "A b. C d. E f. G h")

    assert both.delta_words == ["A", "b.", "C", "d."]
    assert next_one.delta_words == []
    assert after.delta_words == ["E", "f."]
    assert state.committed_words == ["A", "b.", "C", "d.", "E", "f."]


def test_stable_iterations_one_commits_immediately():
    state = SentenceCommitState()

    update = run(state, "Done. Next", stable_iterations=1)

    assert update.delta_words == ["Done."]


def test_allow_commit_false_defers_without_losing_stability():
    state = SentenceCommitState()

    run(state, "Hi there. More", allow_commit=False)
    held = run(state, "Hi there. More", allow_commit=False)
    released = run(state, "Hi there. More words")

    assert held.delta_words == []
    assert released.delta_words == ["Hi", "there."]


def test_committed_words_are_never_revised_by_a_later_hypothesis():
    state = SentenceCommitState()
    run(state, "I can not go. We", stable_iterations=1)

    update = run(state, "I cannot go. We left. Then", stable_iterations=1)

    assert state.committed_words == ["I", "can", "not", "go.", "We", "left."]
    assert update.delta_words == ["We", "left."]


def test_unlocatable_committed_span_holds_commits():
    state = SentenceCommitState()
    run(state, "Alpha beta gamma. Delta", stable_iterations=1)

    update = run(state, "Totally different words here. And more", stable_iterations=1)

    assert update.committed_end is None
    assert update.delta_words == []
    assert state.committed_words == ["Alpha", "beta", "gamma."]


def test_words_after_committed_realigns_and_falls_back_positionally():
    state = SentenceCommitState(committed_words=["I", "can", "not", "go."])

    assert words_after_committed(state, "I cannot go. We left".split()) == ["We", "left"]
    assert words_after_committed(state, "x y z w v".split()) == ["v"]


def test_locate_committed_end_rejects_weak_alignment():
    assert locate_committed_end(["a", "b"], ["a", "b", "c"]) == 2
    assert locate_committed_end([], ["a"]) == 0
    assert locate_committed_end(["a", "b", "c", "d"], ["z", "y", "d"]) is None
    assert locate_committed_end(["a", "b", "c", "d", "e"], ["a", "b", "q"]) is None


# --- seam de-duplication ----------------------------------------------------


def test_seam_drops_the_longest_exact_overlap():
    committed = "we went home today.".split()

    assert seam_overlap(committed, "home today. And then".split()) == 2
    assert seam_overlap(committed, "Today and then".split()) == 1
    assert seam_overlap(committed, "And then".split()) == 0


def test_seam_repeated_word_only_drops_the_overlap():
    assert seam_overlap("I said no.".split(), "no. No. No.".split()) == 1


def test_seam_drops_a_cut_word_fragment():
    committed = "that was the sentence.".split()

    assert seam_overlap(committed, "tence. And so".split()) == 1
    assert seam_overlap(committed, "the sentence. And".split()) == 2
    assert seam_overlap(committed, "e. And".split()) == 0


def test_seam_tolerates_a_misheard_cut_word_backed_by_two_matches():
    committed = "we went home today.".split()

    assert seam_overlap(committed, "uh went home today. And".split()) == 4
    # One matching word is not enough evidence for a non-suffix piece.
    assert seam_overlap(committed, "uh today. And".split()) == 0


def test_seam_window_is_bounded():
    committed = [f"w{i}" for i in range(12)]

    assert seam_overlap(committed, committed[2:] + ["next"], max_words=8) == 0
    assert seam_overlap(committed, committed[4:] + ["next"], max_words=8) == 8


def test_seam_without_committed_tail_drops_nothing():
    assert seam_overlap([], "anything here".split()) == 0


# --- word emergence ---------------------------------------------------------


def test_emergence_bounds_follow_first_appearance():
    tracker = WordEmergenceTracker()
    tracker.update("one two".split(), 10)
    tracker.update("one two three".split(), 20)
    tracker.update("One, two three four.".split(), 30)

    assert tracker.end_bounds(0) == (0, 10)
    assert tracker.end_bounds(2) == (10, 20)
    assert tracker.end_bounds(3) == (20, 30)


def test_emergence_revised_word_is_bounded_by_later_words():
    tracker = WordEmergenceTracker()
    tracker.update("the cat sat".split(), 10)
    tracker.update("the hat sat down".split(), 20)

    # "hat" is new, but "sat" after it was already heard at step 10.
    assert tracker.end_bounds(1) == (10, 10)


def test_emergence_carried_shifts_steps_and_keeps_later_words():
    tracker = WordEmergenceTracker()
    tracker.update("a b".split(), 10)
    tracker.update("a b c".split(), 20)

    carried = tracker.carried(start_index=2, dropped_steps=12)
    carried.update("b c d".split(), 15)

    assert carried.words == ["b", "c", "d"]
    assert carried.end_bounds(1) == (0, 8)
    assert carried.end_bounds(2) == (8, 15)


def test_emergence_rejects_out_of_range_index():
    tracker = WordEmergenceTracker()
    tracker.update(["a"], 5)

    with pytest.raises(IndexError):
        tracker.end_bounds(1)
