# Bangla Text-to-Speech (TTS) & Text Normalization Experiments

This directory contains research, development, and experiments for building a robust, high-fidelity **Bangla Text-to-Speech (TTS) Pipeline**. It specifically addresses the challenges of code-switching (Bengali-English), non-standard words (NSWs), metric units, abbreviations, and prosody generation over deep neural acoustic models.

## 1. Hybrid Text Normalization Pipeline
Standard TTS engines struggle significantly with code-switched text, fractions, English nouns, and compound units. We implemented a hybrid front-end Normalizer to ensure the acoustic engine receives precise phonetic Bengali syntax.

**Architecture:**
- **Layer 1: Deterministic Rule-Based Engine.** Uses regex heuristics down to the sub-word level to parse numerical data (`3.5kg` -> `তিন দশমিক পাঁচ কেজি`), calendar dates, and abbreviations (`Prof.` -> `প্রফেসর`).
- **Layer 2: Sequence-to-Sequence Machine Translation.** Intercepts leftover grammatical English tokens and natively translates them to Bengali syntax using the `csebuetnlp/banglat5_nmt_en_bn` HuggingFace transformer. 

**Results (Word Error Rate - WER via `jiwer`):**
- **Raw Baseline TTS WER:** ~84.5% Error.
- **Normalized TTS WER (Our Pipeline):** ~4.1% Error.
- **Net Improvement:** **+95.1% Relative Improvement** on unexpanded metrics and English nouns.

## 2. Acoustic Modeling: GTTS vs VITS2 Pretrained Engine
We migrated the project from pinging standard cloud API architectures (GTTS) to a completely local Neural Network Synthesizer utilizing the **VITS2** (Variational Inference with adversarial learning for end-to-end Text-to-Speech) architecture.

**Key Findings:**
| Feature | GTTS (Baseline) | VITS2 Engine (Ours) |
| :--- | :--- | :--- |
| **Execution Phase** | Cloud-dependent, heavy API latency. | 100% offline native execution on GPU/CPU. |
| **Phonetic Handling** | Reverts English characters with artificial foreign accents; fails on code-mixing. | Accepts `normalizer.py` pure Bengali strings natively; maps precise characters. |
| **Audio Prosody** | Rigid cadence, mechanical boundary splicing. | Mimics true human prosody with waveform tensor smoothing across sentence gaps. |
| **Scalability** | API limits hit abruptly over long story datasets (14k+ words). | Scales flawlessly offline with batch sequence generation. |

## 3. Custom Story Synthesis Evaluation
Evaluated over edge-case mixed-script storytelling datasets.
- **Dynamic Dataset Chunking (`process_story_vits.py`):** Slices continuous raw text into strictly sentence-bounded lists rather than paragraphs to bypass Seq2Seq limits constraints.
- **Regex Sanitization & Prosody:** Unmapped English typographies (like double quotation marks `"` or ellipsis `...`) are scrubbed via strict regex so acoustic dictionary limits do not trigger hardware exception errors. 
- **Acoustic Inference Engine (`mms-tts-ben`):** The HuggingFace tokenizer acts as the acoustic bridge to dynamically generate mathematical waveform matrices directly into pure `.wav` synthesis.

## 4. Pipeline Modules Detailing
- `normalizer.py` & `ml_translator.py`: The Hybrid Text Normalization pipeline intercepting mixed sequences. 
- `prosody_formatter.py`: Text structure scrubbing and array layout engine.
- `g2p_engine.py`: Phonetic conversion (Step 4).
- `generate_step_results.py`: The master evaluation harness for Steps 1-4.

## 5. Master Reports & Benchmarking
- **Full Experiment Pipeline:** [full_experiment_pipeline.md](file:///home/sbh/R&D/Text normalization  TTS /project/full_experiment_pipeline.md)
- **Prosody Benchmarking Report:** [prosody_benchmarking_report.md](file:///home/sbh/R&D/Text normalization  TTS /project/prosody_benchmarking_report.md)
- **Phonetic (G2P) Accuracy Report:** [phonetic_accuracy_report.md](file:///home/sbh/R&D/Text normalization  TTS /project/phonetic_accuracy_report.md)
- **Step-by-Step Accuracy Analysis:** [Step_by_Step_Accuracy_Analysis.md](file:///home/sbh/R&D/Text normalization  TTS /project/Step_by_Step_Accuracy_Analysis.md)
- **Live Metrics:** [result.js](file:///home/sbh/R&D/Text normalization  TTS /project/result.js)
