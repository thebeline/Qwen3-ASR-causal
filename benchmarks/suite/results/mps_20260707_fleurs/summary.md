# FLEURS multilingual regression: the causal tower is English-only

FLEURS test, 80 utterances per language (≤30 s), streaming stack on MPS
fp16 (validated at quality parity with CUDA), BasicTextNormalizer, WER
(CER for zh). Runner: fleurs_regression.py (kept alongside).

| language | windowed | causal | degradation |
|---|---:|---:|---:|
| French (WER) | 7.78 | 37.85 | 4.9x |
| German (WER) | 12.57 | 49.82 | 4.0x |
| Mandarin (CER) | 11.36 | 85.65 | 7.5x |

The causal tower was self-distilled on LibriSpeech-960 (English read
speech) only; the distillation overwrote the tower's multilingual
representations. Verdict: the causal audio backend is **English-only**
the runtime warns on other languages, the windowed backend remains the
multilingual path, and recovering other languages needs multilingual
audio in a future distillation mix (MLS/CommonVoice: see the roadmap).
