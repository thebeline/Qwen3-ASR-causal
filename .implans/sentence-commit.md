# Sentence commit mode: notes and upstream PR plan

Fork-local notes (github.com/thebeline/Qwen3-ASR-causal, branch `local`).
This directory must never appear in an upstream PR.

**Status:** Upstream PR parked 2026-10-08 (Mike); prefix-mode comparison used
Chapterhouse settings, not upstream's hold 6 / stable 1 defaults — measure
those before any PR.

## What it does

Feature commit: `7f7d5f3 feat: sentence commit mode for qwen3-streaming`
(on top of upstream main `8975258`). Selected with
`--qwen3-streaming-commit-mode sentence`; the default stays `prefix`.

- `sentence_commit.py` holds the pure text and bookkeeping logic (no torch):
  `split_sentences` splits at terminal punctuation with a small
  abbreviation/initialism heuristic; `update_sentence_commit` decides commits;
  `seam_overlap` counts the leading words of a carried-over segment that
  re-transcribe committed text; `WordEmergenceTracker` gives per-word
  audio-step bounds from the decode at which each word first appeared (the
  backend has no word timestamps), so a rollover can sit at a sentence end.
- `sentence_streamer.py` (`SentenceSegmentedStreamer`) replaces the
  word-prefix commit policy and the wholesale time-based rollover:
  - Commit whole sentences only. `hold_back_words` does not apply; the
    unfinished last sentence is the hold-back.
  - Soft rollover: once the segment reaches `sentence_soft_steps` (150 steps,
    12 s), cut at the end of the last committed sentence. Committed sentences
    leave the decoded audio, which bounds per-chunk decode cost; the
    in-progress sentence carries its audio into the next segment.
  - Seam: the cut sits `sentence_rollover_margin_steps` (0.5 s) before the
    estimated sentence end; the re-transcribed committed words are dropped
    with `seam_overlap`.
  - Safety valve: past `segment_max_cached_steps` with no usable sentence end,
    the segment is finalized wholesale as in prefix mode.
  - Committed text only grows by appending, across rollovers too.

**Current commit rule:** a sentence is committed once it is complete (followed
by at least one more word) and its words are identical across the last
`stable_iterations` decodes. Chapterhouse runs `stable_iterations=2` with
1.0 s chunks. (Upstream's shipped prefix-mode default is hold 6 / stable 1.)

## Measured result (2026-10-08)

30-minute window, 3300-5100 s of the 2026-10-04 assembly recording.
Reference is Mike's offline Qwen3-ASR pipeline, so these figures are agreement
with that pipeline, not error against human ground truth. 4027 reference words.
RTX 4070 Ti.

Every arm ran Chapterhouse's `run.sh` settings: `stable_iterations` 2, segment
max 200 steps, keep-tail 25. The prefix arm ran upstream's code with those
settings, not upstream's shipped defaults (hold 6 / stable 1).

| arm | WER | S / D / I | commit p50 | commit p95 | revised words | RTF | peak VRAM |
|---|---:|---|---:|---:|---:|---:|---:|
| prefix (upstream code, Chapterhouse settings), 1 s chunks | 9.86% | 108 / 78 / 211 | 5.93 s | 14.51 s | 3 | 0.276 | 4358 MiB |
| **sentence, 1 s chunks** | **7.30%** | 104 / 87 / 103 | **4.35 s** | **11.60 s** | 0 | 0.253 | 4358 MiB |
| sentence, 2 s chunks | 7.95% | 99 / 96 / 125 | 5.59 s | 12.45 s | 0 | 0.152 | 4334 MiB |
| sentence, 1 s decode, 2 s packets | 8.12% | 100 / 95 / 132 | 5.61 s | 12.39 s | 0 | 0.156 | 4334 MiB |

- The WER gain is almost all insertions (211 to 103): fewer duplicated words
  at segment seams.
- 2 s cadence: sentence mode with 2 s chunks gives 7.95% WER and p50 5.59 s;
  1 s decode fed 2 s packets gives 8.12% and 5.61 s. Against 1 s chunks that
  is about +0.7 WER points and +1.2 s p50, for about 40% less compute
  (RTF 0.253 to 0.152).
- Both 1 s arms have the same 4 dropped runs (34 words, longest 3.6 s at
  3672.5 s). The drops are not caused by the commit mode; Chapterhouse's
  `docs/HANDOFF.md` traces most of them to the VAD gating soft speech.
- "0 revised words" holds by design (commits are append-only), so it says
  nothing about corrections lost by committing early; see idea A.

Source: `whisperlivekit/.scratch/assembly-eval/scores/window-3300-5100.json`
in the Chapterhouse superproject (not committed).

### Full file (2026-10-08)

The whole recording (0-8963 s, 19640 reference words), sentence mode with
1 s chunks, same settings and reference: **WER 4.18%** (S / D / I 338 / 226 /
256), commit p50 **4.44 s**, p95 **11.4 s**, RTF 0.316, peak VRAM 4358 MiB,
0 revised words. 6 dropped runs (45 words), longest 3.6 s at 3672.5 s. There
is no full-file prefix run. Source: `scores/full-asr.json`, same directory.

## Open ideas

**A. Missed-fixes counter.** A committed sentence stays in the re-decoded audio
until the next sentence-end rollover. Log how often a later decode's words for
an already-committed sentence differ from the committed text. It costs nothing
at runtime and measures whether committing earlier loses corrections. Revised
word counts can't show this because commits are append-only.

**B. Settle knobs in `update_sentence_commit`.** Options:
- More stable iterations (about +1 s each at 1 s chunks).
- Require k words after the sentence end, not just one.
- Require the following sentence to be complete (one more sentence, about
  5-10 s more latency).

Use counter A to choose between them.

## Upstream PR plan (parked 2026-10-08, see Status)

1. Simplify first. `SentenceSegmentedStreamer` overrides `update_from_hypothesis`,
   `roll_segment`, `_reset_active_segment_state` and `finalize` from
   `SegmentedCachedFullHypothesisStreamer` in `streamer.py`. Look for
   near-duplication and pull shared pieces into base-class hooks.
2. Add A (counter, exposed in events/logs) and B (knobs) as config.
3. Cut a clean branch from upstream `main` with only the feature: no
   `.implans/`, no Chapterhouse-specific defaults. Rebase onto upstream main
   at that point.
4. QuentinFuxa also maintains WhisperLiveKit. Chapterhouse commit `ea887e0`
   (`feat(qwen3): wire sentence commit mode; add replay comparison script`)
   adds the WLK-side flags. Open that as a companion PR after this one lands.

## Upstream context (checked 2026-10-08)

- Qwen3-ASR-causal: <https://github.com/QuentinFuxa/Qwen3-ASR-causal>. Single
  contributor (QuentinFuxa, 43 commits). No issues, PRs or discussions ever
  (discussions are off), only a `main` branch. Last push 2026-07-09
  (`8975258 ship the v0.2 tower numbers`). Forks: thebeline (this),
  okietrained, siance-assistant, qf-sia, none with new pushes. There is no
  precedent for merging outside PRs here, and the repo has been quiet for
  three months.
- README "Commit latency" section (upstream main): stable 2 to stable 1 cut
  p50 from 5.9 s to 4.1 s; "93% of words were committed by the ~12-16 s
  punctuation rollover rather than the stability policy". That is the problem
  sentence mode targets, so the PR can cite it. Upstream measured on a 21-talk
  long-form set (single-speaker talks), not multi-speaker meeting audio.
  Nothing upstream mentions sentence commits.
- WhisperLiveKit: <https://github.com/QuentinFuxa/WhisperLiveKit>. Active
  (last push 2026-10-01). External PRs do get merged: #385, #387, #389, #391
  and #422 were merged within 1-3 weeks. #375 (qwen3-vllm fixes) was closed
  without merging.
  - Qwen3 streaming landed in #381 (v0.2.23, 2026-07-09):
    <https://github.com/QuentinFuxa/WhisperLiveKit/pull/381>.
  - #389 fixed a qwen3 crash on diarized lines (merged):
    <https://github.com/QuentinFuxa/WhisperLiveKit/pull/389>.
  - Open #425, native Qwen3 MLX streaming adapter (clkao):
    <https://github.com/QuentinFuxa/WhisperLiveKit/pull/425>.
  - Open #449, caption display semantics (clkao). It derives captions per
    sentence downstream ("a finished sentence is held… final text never
    visibly shrinks"). This is adjacent to sentence commit; mention it in the
    companion PR: <https://github.com/QuentinFuxa/WhisperLiveKit/pull/449>.
  - No issue or PR about qwen3 commit latency, stable iterations, hold-back or
    rollover behaviour.
