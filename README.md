# R&D

# Banglish STT & TTS Pipeline

This repository contains the ongoing Research & Development for an end-to-end code-mixed Bengali-English Speech-to-Text (STT) and Text-to-Speech (TTS) architecture.

The project is structured into two main components:
1. **[STT (Speech-to-Text)](#stt-speech-to-text-benchmarking-framework)**
2. **[TTS (Text-to-Speech)](#tts-text-to-speech-synthesis-pipeline)**

---

## 🎙️ STT: Speech-to-Text Benchmarking Framework
The `STT` directory houses a comprehensive 5-step global evaluation and benchmarking harness. Its main objective is to measure and optimize the **Word Error Rate (WER)** and **Character Error Rate (CER)** across various open-source models, processing strategies, and linguistic edge-cases (specifically code-switching and dialectal Bengali).

### The 5-Step Pipeline
The pipeline systematically cleans audio, detects language, transcribes, and normalizes the text:
1. **Audio Preprocessing (`Step 1`)**: Tests various methods (VAD, Bandpass Filters, Loudness Normalization, 16kHz resampling) to clean audio before inference.
2. **ASR Model Selection (`Step 2`)**: Evaluates code-switched ASR engines including Whisper (Tiny, Base) with zero-shot & domain prompting, faster-whisper, and IndicWav2Vec.
3. **Text Normalization (`Step 3`)**: Applies domain-specific rule-based transcript fixing for phone numbers, dates, currency, and abbreviations.
4. **Language Identification (`Step 4`)**: Tests token-level LID (Language Identification) using Unicode-range detection vs neural fastText models for mixed text.
5. **Post-Processing (`Step 5`)**: Cleans up output utilizing 4 sub-methods:
   - **5A:** English point-word code-mix correction (`অর্ডার` → `order`).
   - **5B:** Dialectal to Standard Bengali mapping (`হইছে` → `হয়েছে`).
   - **5C:** Rule-based punctuation heuristic restoration.
   - **5D:** Deep Multilingual Punctuation Restoration (Transformers token-classification backend).

### STT Usage
Navigate to the `STT` directory and run the evaluator harness:
```bash
cd STT
source venv/bin/activate

# Run evaluation on a specific step/approach
python3 run_experiment.py --approach 1A

# View the leaderboard for a specific step
python3 run_experiment.py --step 5 --compare
```
*Note: Progress and individual benchmark rankings are logged internally and documented in `STT/experiment_plan.md`.*

---

## 🔊 TTS: Text-to-Speech Synthesis Pipeline
The `TTS` directory contains a high-fidelity Bangla synthesis pipeline engineered to handle the challenges of code-switching, metric/English unit abbreviations, and prosody generation over deep neural acoustic models.

### Key Features & Architecture
- **Hybrid Text Normalization (`normalizer.py`, `ml_translator.py`)**: Standard TTS engines choke on English numbers and acronyms. This 2-layer pipeline intercepts them:
  - **Layer 1:** A deterministic regex-based ruleset expands numbers (`3.5kg` → `তিন দশমিক পাঁচ কেজি`), dates, and abbreviations (`Prof.` → `প্রফেসর`).
  - **Layer 2:** A sequence-to-sequence machine translation matrix (`csebuetnlp/banglat5_nmt_en_bn`) catches any leftover English grammatical tokens and natively translates them to Bengali syntax.
  - *Result: Improves raw baseline phonetic synthesis Word Error Rate by +95.1% relative.*

- **Acoustic Modeling (`VITS2`)**: 
  - Offloaded from cloud-dependent API latency (GTTS), the system localizes inference utilizing **VITS2** (Variational Inference for end-to-end TTS) natively on the GPU.
  - It mimics continuous human prosody smoothly across tensor boundaries without the mechanical splicing seen in legacy baseline engines.

- **Long-Form Narrative Engine (`process_story_vits.py`)**: 
  - Iterates perfectly through long-form narrative text datasets (e.g., 14,000+ word stories) without bottlenecking sequence-to-sequence memory limits.
  - Dynamically chunks paragraph matrices into strict sentences and natively scrubs unmapped English typography (like ellipsis or double quotes) to prevent acoustic dictionary dropouts.

---

## Requirements and Installation
Specific environments are maintained inside their respective folders. Generally, both pipelines rely heavily on `torch`, `transformers`, `openai-whisper`, `fasttext`, and standard audio suites (`soundfile`, `librosa`).

1. Ensure FFmpeg is installed on your OS.
2. Follow the individual setup steps located inside `STT/` and `TTS/` environments.
