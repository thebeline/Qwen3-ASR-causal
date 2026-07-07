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
