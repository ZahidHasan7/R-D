# Approaches Tried for TTS Improvement

This document tracks the steps and approaches implemented to improve the quality (naturalness) and performance (latency) of the Bangla TTS pipeline.

## 1. Quality (Naturalness & Accuracy) Improvements
- **Hybrid Normalization Engine:** Custom rules to handle units, abbreviations, numbers, dates, and currency.
- **Code-Switching Contextual Refinement:** ML-based phonetic transliteration of English words to Bangla using T5.
- **Multi-layered Hybrid G2P:** A 4-layer fallback system (Dictionary -> Rules -> Neural G2P -> Fallback) for accurate pronunciation.
- **Dual-Model Hybrid Synthesis:** Using Meta MMS (VITS) for Bangla and Coqui XTTS v2 for English.
- **Audio Post-Processing:** Telephonic bandpass filter (300Hz-3400Hz) and volume normalization.
- **Emotional TTS (Experimental):** Exploring Orpheus 3B (Llama + SNAC tokens) for expressive, non-deterministic generation.

## 2. Performance (Latency) Improvements
- **Parallel Segmentation:** Using a `ThreadPoolExecutor` to dispatch and synthesize Bangla and English segments simultaneously.
- **CPU Optimization:** Thread management and precise XTTS lock mechanisms for stability in constrained environments.
- **Lazy Loading:** Loading heavy models (MMS, XTTS, Whisper) only upon first request.

---
*Note: As new experiments and approaches are tried, they will be logged below.*
