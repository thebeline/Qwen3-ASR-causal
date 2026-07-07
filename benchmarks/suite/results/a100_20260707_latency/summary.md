# A100 cross-check + cadence stage 2 — MCIF-21, causal backend

JarvisLabs A100-40GB (cu130, python 3.12), same corpus/config as the
MPS run in `../mps_20260707_latency/`. Events regenerated on-device;
policies replayed offline with `commit_latency_from_events.py`.

## The latency frontier is hardware-independent

Per-word proxy latency (seconds), A100 vs (MPS):

| policy | p50 | p95 | live WER (legacy norm) |
|---|---:|---:|---:|
| hold 6 / stable 2 | 5.93 (5.89) | 13.64 (13.6) | 0.2639 |
| hold 6 / stable 1 (default) | 4.02 (3.98) | 7.90 (7.9) | 0.2696 |
| hold 2 / stable 1 | 2.18 (2.13) | 7.84 (7.8) | 0.2726 |

Settled corpus WER 0.2639 (human refs, legacy norm) vs 0.2621 on MPS —
run noise. HF/CUDA RTF 0.159 (matches the published H100 HF 0.160).
Commit latency is policy-bound, not hardware-bound: the numbers in the
README hold across devices.

## Cadence stage 2: 960 ms blocks are not worth it

Same corpus at `--chunk-ms 960` (96-frame blocks, a trained configuration):

| point | settled WER | live p50 (h6/s1) | live p50 (h2/s1) | decode ms per audio-s |
|---|---:|---:|---:|---:|
| 1920 ms (shipped) | 0.2639 | 4.02 | 2.18 | 148 |
| 960 ms | 0.2811 | 3.02 | 2.05 | 205 |

Half-size blocks cost +1.7 pt settled WER and +39% decode compute, and
barely move the low-latency floor (2.05 vs 2.18 s): the floor is decode
cadence + generation time, and at hold 2 the 1920 ms point already sits
on it. Verdict: stay at 1920 ms. The route below ~2 s p50 is an
early-commit policy (e.g. punctuation-gated, replayable offline from
these same events), not a faster cadence.

## Concurrency (HF path, one shared ASR, paced 1.0x sessions)

| sessions | worst backlog med / p95 (s) | words/min min / median |
|---:|---|---|
| 1 | 1.5 / 2.0 | 150 / 150 |
| 2 | 1.5 / 2.0 | 128 / 139 |
| 4 | 1.5 / 2.0 | 122 / 138 |
| 6 | 1.5 / 3.5 | 119 / 139 |
| 8 | 2.5 / 6.5 | 120 / 138 |

The decode lock serializes, but decodes are short: an A100 sustains 6-8
real-time sessions on the plain HF path with modest backlog growth at 8.
(On MPS the same bench saturates at 2 sessions — allocator ceiling — see
../mps_20260707_latency/concurrency_mps.json.) Larger fleets belong to
the vLLM backend.

## LibriSpeech gates at the new default (full test sets, A100)

| split | stable_iterations | utts | WER (whisper norm) | published (stable 2) |
|---|---:|---:|---:|---:|
| test-clean | 1 | 2620 | **3.67** | 3.64 |
| test-other | 1 | 2939 | **7.22** | 7.16 |

The stable_iterations=2 -> 1 flip costs +0.03 / +0.06 pt on short-form —
run noise. Short utterances commit mostly through the EOS flush, so the
long-form latency gain is free here.

## LibriSpeech gates at the new default (stable_iterations=1)

Full test sets through the streaming stack (A100, `ls_gates/`):

| set | s=1 (this run) | s=2 (published) | Δ |
|---|---:|---:|---:|
| test-clean (2620 utts) | **3.67** | 3.64 | +0.03 |
| test-other (2939 utts) | **7.22** | 7.16 | +0.06 |

Same-stack control on the first 500 utterances: clean s1/s2 and other
s1 11.18 / s2 11.14 — deltas within run noise. Short utterances commit
mostly through the EOS flush, so the stability-iteration flip does not
affect short-form quality; the default change is validated on both axes
(long-form: +0.5 pt for p50 5.9→4.0 s; short-form: unchanged).
