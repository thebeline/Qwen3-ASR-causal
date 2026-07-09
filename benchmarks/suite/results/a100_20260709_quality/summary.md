# Quality diagnostics: attention-sinks probe + C1 distillation pilot

JarvisLabs A100-40GB, 2026-07-09, ~6 GPU-hours total. Two experiments from
the quality roadmap, both gated ahead of any larger training spend.

## Attention-sinks probe: NEGATIVE (hypothesis falsified)

StreamingLLM-style sinks (`--qwen-audio-sink-steps`, branch
`quality-diagnostics`) pin the first N encoder steps in the causal KV cache.
Tested on the 5 longest MCIF talks with the segment reset DISABLED — the
regime that historically collapses:

| config | WER (human refs, legacy norm) |
|---|---:|
| no reset, sinks 0 | 0.9545 (collapse reproduced) |
| no reset, sinks 16 | 0.9636 |
| no reset, sinks 32 | 0.9642 |
| reset control (shipped behavior) | 0.2559 |

Pinned sinks recover nothing: the long-form drift is NOT initial-key
eviction (the Voxtral -3.25 WER sinks result does not transfer to this
tower). The per-segment encoder reset stays; the remaining lead for
no-reset long-form is state-exposure training (roadmap Phase 1).

## C1 pilot: PROMOTED (data diversity works)

20k-step audio-only self-distillation, warm-started from the published
tower; teacher = the original offline tower. Data: People's Speech (clean)
interleaved with LibriSpeech-960 (p=0.3 replay), mixed 96/192 blocks,
position offsets p=0.5, lr 5e-6 cosine. ~1k equivalent audio-hours, ~$3.

Full MCIF-21 under the production contract (chunk 1920 ms, reset,
finalize latest):

| tower | legacy norm | whisper norm |
|---|---:|---:|
| published (`qfuxa/qwen3-asr-0.6b-streaming`) | 0.2639 | 0.183 |
| pilot (`tower_c1_pilot_ps1k_step20k.pt`, local) | **0.2517** | **0.172** |

About -1 point whisper-norm for one cheap pilot; 4 of the 5 gate files
improved. Promotion criterion (>= 1 legacy point on the corpus) met.

Notes: the in-training gate ran the 5-minute gate files without the
segment reset, so it sat at the ~0.95 collapse and was not informative —
score checkpoints with the production contract instead (as done here).
The pilot checkpoint lives outside the repo
(`~/Downloads/qwen3_checkpoints/`); it is NOT published.

## Next (dedicated session, gated)

Scale C1: 2-5k audio-hours (add GigaSpeech/YODAS-EN or accept the
People's Speech mix), same recipe, ~$10-15 on A100. Kill criterion from
the plan: stop if the corpus does not reach <= 0.16 whisper-norm after
+2k h. Optional: multilingual mix (MLS/CommonVoice) to lift the
English-only limitation in the same run.

## Validation and publication (v0.2 tower)

Full validation of the pilot tower before shipping (A100, `ls_*_pilot.json`,
`latency_pilot_*.jsonl`):

| metric | v0.1 (published) | pilot | delta |
|---|---:|---:|---:|
| MCIF-21 settled (whisper norm) | 18.3 | **17.2** | -1.1 |
| MCIF-21 live, default policy | 18.8 | **17.6** | -1.2 |
| MCIF-21 live, hold_back 2 | 18.9 | 17.8 | -1.1 |
| LibriSpeech test-clean (s=1, full) | 3.67 | 3.73 | +0.06 |
| LibriSpeech test-other (s=1, full) | 7.22 | 7.34 | +0.12 |
| commit latency p50 / p95 (default) | 4.0 / 7.9 s | 4.1 / 8.0 s | unchanged |

Long-form gains dominate a marginal short-form cost; latency is
policy-bound and unchanged. Shipped as the v0.2 weights of
`qfuxa/qwen3-asr-0.6b-streaming` (weights commit 8aaa766; the v0.1
weights remain available at revision 012f13dc). Still English-only.
