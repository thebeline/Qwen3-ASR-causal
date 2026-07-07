# Qwen3-ASR causal

Standalone runtime for the causal streaming audio tower published at
[qfuxa/qwen3-asr-0.6b-streaming](https://huggingface.co/qfuxa/qwen3-asr-0.6b-streaming).

The weights are not stored in this repository. The runtime downloads the causal
tower from Hugging Face and loads the unchanged decoder, adapter and feature
extractor from [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B).

![Offline Qwen3-ASR vs causal streaming execution](assets/offline_vs_causal.svg)

## Install

```bash
pip install -e ".[streaming]"
```

For CUDA/vLLM serving, including the causal audio tower path:

```bash
pip install -e ".[vllm]"
```

For Apple Silicon vLLM Metal serving, install vLLM with the official
vllm-metal instructions first, then install the Metal extra:

```bash
pip install -e ".[metal]"
```

## Usage

HF Transformers causal runtime:

```bash
qwen3-asr-causal transcribe audio.wav --backend hf --language en
```

vLLM CUDA runtime with the live prompt-embedding append path:

```bash
WLK_QWEN3_VLLM_LIVE_MULTIPROCESSING=1 \
qwen3-asr-causal transcribe audio.wav \
  --backend vllm \
  --decoder-backend vllm-live \
  --language en
```

The conservative vLLM fallback uses text-decoder prefix caching:

```bash
qwen3-asr-causal transcribe audio.wav \
  --backend vllm \
  --decoder-backend vllm-text \
  --language en
```

The Apple Silicon backend exposes the same runtime through
`qwen3_asr_causal.metal` for WhisperLiveKit and local benchmark integration.

## How it works

The original Qwen3-ASR audio tower is offline and fully bidirectional. This
runtime uses a fine-tuned causal tower that runs append-only:

- fixed 1.92 s audio blocks, with full attention inside each block;
- causal per-layer KV across blocks;
- bounded 15 s left attention window;
- monotonically increasing sinusoidal positions;
- no re-encoding of already finalized audio blocks.

![Decoder-side serving loop](assets/causal_streaming.svg)

The live text contract is append-only. At each non-final update the runtime
cuts the hypothesis portion aligned to the last 250 ms of audio and only
publishes text that extends the previously published prefix. End of stream
flushes the held-back tail, but still cannot revise words already published.

## Results

Long-form evaluation: 21 MCIF/ACL conference talks, about 2 h total, accented
scientific English, human references, Whisper text normalization. All three
rows run the same base model; the contracts differ.

| system | scoring contract | WER | compute per second of audio |
|---|---|---:|---|
| Offline, VAD-segmented | full file available, no latency | **7.6** | offline |
| Streaming Qwen3-ASR (windowed) | settled transcript | 8.4 | 126 GFLOPs avg, growing to 172 within each segment |
| Streaming Qwen3-ASR (windowed) | live, no rewrite, 250 ms tail cut + EOS flush | **12.6** | 126 GFLOPs avg, growing to 172 |
| Streaming Qwen3-ASR causal | live = settled (0 committed-word retractions on all 21 files) | **18.1** | **42 GFLOPs, constant** |
| Streaming Qwen3-ASR causal | committed-only, no EOS flush | 19.4 | 42 GFLOPs, constant |

Offline-done-right is the quality ceiling; the windowed backend is that
strategy made streaming; the causal tower trades ~1.4x the live windowed WER
for 3.0x (avg) to 4.1x (peak) less compute, constant in stream age
(`benchmarks/suite/run_flops.py`, FlopCounterMode on real weights).

Short-form through the same streaming stack: LibriSpeech test-clean **3.64** /
test-other **7.16** (for reference, Voxtral Mini Realtime 3B @480 ms publishes
2.1 / 5.5 with ~5x the parameters).

## Commit latency

Per-committed-word latency (word spoken → word committed, feeding at 1.0x),
measured by replaying the 21-talk events under the live contract
(`benchmarks/suite/results/mps_20260707_latency/`):

| policy | p50 | p95 | live WER (whisper-norm) |
|---|---:|---:|---:|
| hold 6 / stable 2 (old default) | 5.9 s | 13.6 s | 18.3 |
| **hold 6 / stable 1 (default)** | **4.0 s** | **7.9 s** | **18.8** |
| hold 2 / stable 1 (`--qwen3-streaming-hold-back-words 2`) | 2.1 s | 7.8 s | 18.9 |

At the old default, 93% of words were committed by the ~12-16 s punctuation
rollover rather than the stability policy — one stability iteration is enough
on the causal hypothesis stream. The 2.1 s point sits at the 1.92 s block
cadence floor. The unstable tail is additionally visible immediately through
`get_buffer()` (~2 s behind the audio head).

![Streaming RTF](assets/rtf_combined.svg)

Streaming RTF is ASR compute divided by audio duration, excluding model load.
The H100 bars are pure HF Transformers/CUDA. The A100 bars are vLLM CUDA.
The Metal bars use vLLM Metal on Apple Silicon.

Key numbers:

- H100 HF Transformers: normal windowed 0.293 RTF, causal 0.160 RTF.
- Apple M5 vLLM Metal: normal 0.262 RTF, causal 0.156 RTF.
- Apple M-series HF/MPS fp16: causal 0.116 RTF over the full 21-talk set,
  and quality parity with the H100 bf16 run (settled 18.3 vs 18.1).
- A100 vLLM CUDA: normal 0.146 RTF, causal `vllm-live` 0.099 RTF.
- The A100 `vllm-text` fallback measured 0.113 RTF on the same 22 s smoke.
- CPU fp32 (Apple M-series cores): causal 0.82 RTF on a 277 s talk —
  real-time capable, single session.

These speed smokes are not WER claims; use the long-form WER table above for
quality comparisons.

## Hardware support

| target | backend | dtype | status |
|---|---|---|---|
| CUDA (Linux) | `hf` or `vllm` (`vllm-live` / `vllm-text`) | bf16 | primary; vLLM is the multi-session path |
| Apple Silicon | `hf` (pure torch/MPS) | fp16 | measured: quality parity with CUDA bf16, RTF 0.116 |
| Apple Silicon | `qwen3_asr_causal.metal` (vLLM Metal / MLX) | fp16 | Python 3.12, darwin arm64 only |
| CPU | `hf` | fp32 | works, RTF ~0.8; no quantized path yet |

The HF backend serializes concurrent sessions behind one decode lock: treat
it as the single-session/edge path and use the vLLM CUDA backend when serving
many sessions.

## Python

```python
from qwen3_asr_causal import Qwen3StreamingASR, Qwen3StreamingOnlineProcessor

asr = Qwen3StreamingASR(
    lan="en",
    model_size="Qwen/Qwen3-ASR-0.6B",
    qwen3_streaming_audio_backend="causal",
    qwen3_streaming_tower_checkpoint="qfuxa/qwen3-asr-0.6b-streaming",
)
processor = Qwen3StreamingOnlineProcessor(asr)
```

## Development

Default tests are lightweight and do not download Qwen weights:

```bash
python -m py_compile $(find src/qwen3_asr_causal -name "*.py")
pytest -q
```

Real inference is opt-in:

```bash
QWEN3_ASR_CAUSAL_E2E=1 pytest tests/test_e2e.py
```

The ongoing research notebooks, training scripts, run logs, and WER/RTF
benchmarks live in `experiments/qwen3-causal/` and `benchmarks/`. WhisperLiveKit
imports this package instead of carrying a second copy of the Qwen runtime.

## License

Apache-2.0.
