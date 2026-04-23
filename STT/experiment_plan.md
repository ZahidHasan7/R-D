# Banglish STT Benchmarking — Experiment Plan

> **Goal:** Systematically improve Speech-to-Text accuracy on Bangla/code-mixed data.
> Each approach is benchmarked using `eval_framework.py` and compared to the **baseline** so progress can be shown to the team lead.

---

## How to Use This Plan

1. For each Step, implement the approaches **in order**.
2. After each approach, run `eval_framework.py` → log WER, CER, EN-WER.
3. Call `evaluator.compare(step="StepX")` to print the ranking table.
4. Mark approaches as `[x] Done` when complete.

---

## Metrics Tracked (per approach)

| Metric           | Description                                      |
| :--------------- | :----------------------------------------------- |
| **WER**          | Word Error Rate (lower = better)                 |
| **CER**          | Character Error Rate (lower = better)            |
| **EN-WER**       | WER on English tokens only (code-mixed accuracy) |
| **Sentence Acc** | % of sentences with zero errors                  |
| **Token Acc**    | % of tokens correctly recognized                 |

---

## Step 1 — Audio Preprocessing

_Clean the audio before feeding it to the ASR model_

| #   | Approach                            | Status |  WER   | Notes                           |
| :-- | :---------------------------------- | :----: | :----: | :------------------------------ |
| 1A  | Raw baseline (no processing)        |  [x]   | 93.99% | Establish the floor (n=9)       |
| 1B  | Resample to 16kHz                   |  [x]   | 94.13% | Whisper's required sample rate  |
| 1C  | Noise Reduction (noisereduce)       |  [x]   | 92.96% | Best at prop_decrease=0.75      |
| 1D  | Bandpass Filter (300–3400 Hz)       |  [x]   | 96.92% | Isolate human voice frequencies |
| 1E  | Loudness Normalization (pyloudnorm) |  [x]   | 93.84% | Normalize volume across samples |
| 1F  | VAD Segmentation (silero-vad)       |  [x]   | 75.51% | Best at aggressiveness=3        |
| 1G  | RNNoise — Neural Denoiser           |  [ ]   |   —    | Skipped (GPU dependency)        |
| 1H  | Full Stack (1B+1C+1D+1E+1F)         |  [x]   | 97.36% | Best expected accuracy          |

---

## Step 2 — ASR Model Selection

_Convert cleaned audio to text_

| #   | Approach                       | Status |  WER   | Notes                            |
| :-- | :----------------------------- | :----: | :----: | :------------------------------- |
| 2A  | Whisper Zero-Shot (tiny→large) |  [ ]   |   —    | Partial run only (n=1)           |
| 2B  | Whisper + Domain Prompt        |  [x]   | 85.04% | n=9 run completed                |
| 2C  | faster-whisper                 |  [ ]   |   —    | 4× speed, same accuracy          |
| 2D  | BanglaASR (fine-tuned Whisper) |  [ ]   |   —    | Bengali-specific model           |
| 2E  | IndicWav2Vec (AI4Bharat)       |  [x]   | 97.36% | n=9 run completed                |
| 2F  | Whisper LoRA Fine-tune         |  [ ]   |   —    | Train on your own call data      |
| 2G  | WavLM Two-Stage Fine-tune      |  [ ]   |   —    | Clean Bengali → Noisy telephonic |
| 2H  | Ensemble (2 models, pick best) |  [ ]   |   —    | Combine outputs                  |

---

## Step 3 — Text Normalization

_Fix numbers, dates, currency, abbreviations in the transcript_

| #   | Approach                                  | Status | WER | Notes                              |
| :-- | :---------------------------------------- | :----: | :-: | :--------------------------------- |
| 3A  | Existing Rule-Based (from TTS project)    |  [ ]   |  —  | **Baseline** — already implemented |
| 3B  | Extended Rules (phone, ৳, dates, OTP)     |  [ ]   |  —  | Add domain-specific rules          |
| 3C  | Dictionary Lookup (নগদ → Nagad)           |  [ ]   |  —  | Domain term matching               |
| 3D  | LLM Hybrid (rules + GPT-4o-mini fallback) |  [ ]   |  —  | Best for edge cases                |

---

## Step 4 — Language Identification (per token)

_Label each word as Bengali, English, or mixed_

| #   | Approach                              | Status | EN-WER | Notes                                     |
| :-- | :------------------------------------ | :----: | :----: | :---------------------------------------- |
| 4A  | Rule-Based Script Detection (Unicode) |  [x]   | 5.82%  | Baseline Unicode range script detection   |
| 4B  | fastText LID                          |  [x]   | 5.35%  | Facebook fastText 176-lang model          |
| 4C  | BanglaBERT Fine-Tuned (Token-Level)   |  [ ]   |   —    | Best accuracy, GPU required               |
| 4D  | Hybrid Rule + BERT                    |  [ ]   |   —    | Rules for clear cases, BERT for ambiguous |

---

## Step 5 — Post-Processing

_Final cleanup of the transcript text_

| #   | Approach                                     | Status | WER  | Notes                               |
| :-- | :------------------------------------------- | :----: | :--: | :---------------------------------- |
| 5A  | English Word Correction (অর্ডার → order)     |  [x]   | 4.27 | Initial run completed (n=9)         |
| 5B  | Dialectal Normalization (হইছে → হয়েছে)      |  [x]   | 4.27 | Initial run completed (n=9)         |
| 5C  | Punctuation — Rule-Based (।?!)               |  [x]   | 4.27 | Initial run completed (n=9)         |
| 5D  | Punctuation — ML Model                       |  [x]   | 4.26 | Initial run completed (compat mode) |
| 5E  | LLM Post-Processing (GPT-4o-mini, selective) |  [ ]   |  —   | Only for low-quality transcripts    |
| 5F  | Full Stack (5A+5B+5C then LLM if needed)     |  [ ]   |  —   | **Best expected overall accuracy**  |

---

## Progress Summary Table

> Update this after each step is fully benchmarked.

| Step                    | Best Approach         | Baseline WER | Best WER | Improvement |
| :---------------------- | :-------------------- | :----------: | :------: | :---------- |
| Step 1: Audio           | 1F (VAD level 3)      |    93.99%    |  75.51%  | 18.48%      |
| Step 2: ASR             | 2B (Whisper + Prompt) |    93.99%    |  85.04%  | 8.95%       |
| Step 3: Normalization   | —                     |      —       |    —     | —           |
| Step 4: LID             | 4B (fastText)         |   5.82%\*    | 5.35%\*  | 0.47%       |
| Step 5: Post-Processing | —                     |      —       |    —     | —           |

---

## References

- `eval_framework.py` — Evaluation harness (WER/CER/EN-WER)
- `data/metadata.csv` — Ground truth audio + transcript pairs
- `experiments/banglish_stt_v1/results.jsonl` — All logged results
