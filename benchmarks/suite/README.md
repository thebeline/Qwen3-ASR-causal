# Benchmark suite

Reproducible benches for the published causal runtime. Everything here
imports only `qwen3_asr_causal` (never whisperlivekit) so the numbers
describe the package as users install it.

| script | measures | status |
|---|---|---|
| `run_flops.py` | measured GFLOPs per audio-second, causal vs windowed (FlopCounterMode, real tower weights) | ready |
| `run_long_session.py` | 45-min single session: memory slope, output-rate stability, bounded event history | ready |
| `run_concurrency.py` | N paced sessions on one shared ASR: backlog, decode busy ratio, output rate | ready |
| `run_wer.py` | MCIF-21 + LibriSpeech WER under the live no-rewrite contract (live vs settled, RTF, retractions) | ready |
| `run_latency.py` | per-committed-word latency p50/p90/p95 + time-to-first-token at 1.0x pace | ready |
| `fetch_data.py` | downloads LibriSpeech test-clean/test-other (sha256-pinned); prints MCIF manual steps | ready |

Latest results: `results/mps_20260707_latency/` (latency frontier,
45-min long-session verdicts, MPS concurrency).

The offline policy-sweep metrology (replay of saved decode events across a
grid of commit policies) still lives in
`experiments/qwen3-causal/scripts/commit_latency_from_events.py`; the suite's
`run_latency.py` measures the same per-word quantity live, at one config.

```bash
python benchmarks/suite/run_flops.py   # tower defaults to the published HF repo

# WER: fetch LibriSpeech once, then run (live + settled WER, RTF)
python benchmarks/suite/fetch_data.py --librispeech test-clean
python benchmarks/suite/run_wer.py \
    --librispeech-dir ~/.cache/qwen3_asr_causal/bench/LibriSpeech/test-clean \
    --limit 100 --output-json wer.json

# Latency: manifest rows {"id","audio","text"?,"word_alignments"?}, 1.0x pace
python benchmarks/suite/run_latency.py \
    --manifest-jsonl manifest.jsonl --output-json latency.json
```

Reference numbers (H100, 2026-06): causal 41.5 GFLOPs/s constant vs
windowed 125.9 avg / 172.2 peak.
