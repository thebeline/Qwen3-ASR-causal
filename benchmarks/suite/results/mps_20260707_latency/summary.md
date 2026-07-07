# Per-word commit latency — MCIF-21, causal backend, policy sweep

First measurement of the metric that matters for live captions: how long
after a word is spoken does the live no-rewrite contract commit it.

## Method

- Events: `experiments/qwen3-causal/scripts/eval_cached_full_hypothesis.py`
  at the production causal operating point (chunk 1920 ms, block 192,
  left 15 s, block-bidirectional, repetition 1.15 / no-repeat-3-gram,
  seg cap 200 + reset, punct rollover min 150 + roll-before-generate,
  `--language English`), Apple M-series MPS fp16.
  Corpus check vs published H100 numbers: settled whisper-norm WER
  0.1826 vs 0.1810 published; per-file smoke 0.3440 vs 0.343.
- Replay: `experiments/qwen3-causal/scripts/commit_latency_from_events.py`
  (exact per-segment replay of the append-only emission contract; wall
  clock = cumulative fed audio + recorded decode time; per-word latency
  proxy = commit wall − audio position where the settled word first became
  hypothesis-stable, biased low by ≤ one chunk).
- 21 talks, ~2 h audio, ~13.7 k committed words per config.

## Frontier (proxy latency, seconds; WER = live, whisper-normalized)

| hold_back / stable_iter | p50 | p90 | p95 | WER live | Δ vs settled 0.1826 | rollover-committed |
|---|---:|---:|---:|---:|---:|---:|
| 6 / 2 (previous default) | 5.89 | 13.5 | 13.6 | 0.1830 | +0.0004 | 93% |
| 8 / 1 | 4.00 | 7.8 | 9.7 | 0.1873 | +0.0047 | 55% |
| **6 / 1 (new causal default)** | **3.98** | **7.8** | **7.9** | **0.1878** | **+0.0052** | 50% |
| 4 / 1 | 3.96 | 5.9 | 7.8 | 0.1882 | +0.0056 | 47% |
| 2 / 1 (low-latency suggestion) | 2.13 | 5.9 | 7.8 | 0.1893 | +0.0067 | 44% |

Full 54-config grid in `sweep_stage1.jsonl` (legacy-norm WER inside;
whisper-norm re-scored for the rows above).

## Readings

- At stable_iterations=2 the policy barely commits: 93% of words wait for
  the ~12-16 s punctuation rollover — that cadence, not the holdback, set
  the old latency (p95 = words waiting out their segment).
- One knob (stable_iterations 2→1) buys ~1.5-1.7x at every percentile for
  +0.5 pt; hold_back only matters below 3 words, where p50 hits the decode
  cadence floor (~2.1 s at 1.92 s chunks).
- First-commit drops 14.4 s → 5.8-7.1 s; EOS flush shrinks 18 → 5-9 words.
- Wall clocks include M-series decode times (~0.2-0.5 s); H100 shaves
  ~0.1-0.3 s. Policy delay dominates on both.
- Pending validation on the cloud day: LibriSpeech short-form gates at the
  new default, and the (chunk 0.96 s, block 96) cadence point which is the
  only way below ~2 s p50.
