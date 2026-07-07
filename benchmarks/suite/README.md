# Benchmark suite

Reproducible benches for the published causal runtime. Everything here
imports only `qwen3_asr_causal` (never whisperlivekit) so the numbers
describe the package as users install it.

| script | measures | status |
|---|---|---|
| `run_flops.py` | measured GFLOPs per audio-second, causal vs windowed (FlopCounterMode, real tower weights) | ready |
| `run_wer.py` | MCIF-21 + LibriSpeech WER under the live no-rewrite contract | planned |
| `run_latency.py` | per-committed-word latency p50/p95 + time-to-first-word at 1.0x | planned |
| `run_long_session.py` | 45-min session: memory slope, windowed-WER stability | planned |
| `run_concurrency.py` | N concurrent sessions vs per-session latency/RTF | planned |

Until the consolidation lands, the offline latency/policy metrology lives in
`experiments/qwen3-causal/scripts/commit_latency_from_events.py` (replays
saved decode events; see its docstring for the exact latency model) and the
WER/RTF harness in `benchmarks/macos_qwen3/run_wer_rtf.py`.

```bash
python benchmarks/suite/run_flops.py   # tower defaults to the published HF repo
```

Reference numbers (H100, 2026-06): causal 41.5 GFLOPs/s constant vs
windowed 125.9 avg / 172.2 peak.
