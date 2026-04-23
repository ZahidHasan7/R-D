"""
main.py — Production Bangla TTS Front-End Pipeline

Full pipeline:
  Input Text
    → TextNormalizer       (rules: abbrev, months, ordinals, currency, time, units, numbers + BanglaT5)
    → LanguageDetector     (token-level language routing)
    → NERHandler           (named entity pronunciation override)
    → G2PEngine            (grapheme-to-phoneme)
    → ProsodyFormatter     (pause & emphasis markers)
    → TTSEngineV2          (VITS → gTTS fallback)
    → Output Audio + Logs
"""

import os
import time
import sys

# Add root to sys.path to allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.normalizer import TextNormalizer
from src.language_detector import LanguageDetector
from src.ner_handler import NERHandler
from src.g2p_engine import G2PEngine
from src.prosody_formatter import ProsodyFormatter
from src.tts_engine_v2 import TTSEngineV2, TTSBackend
from src.logger import get_logger

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

OUTPUT_DIR       = os.path.join('results', 'outputs', 'demo_outputs')
TEST_CASES_FILE  = os.path.join('tests', 'cases', 'test_cases.txt')
MAX_DEMO_CASES   = None  # Set to None for all cases
AUDIO_LANG       = 'bn'
TTS_BACKEND      = TTSBackend.AUTO  # VITS → gTTS fallback


def build_pipeline():
    """Instantiates all pipeline components."""
    log = get_logger()
    log._logger.info('Building production TTS pipeline...')

    normalizer  = TextNormalizer()
    detector    = LanguageDetector()
    ner         = NERHandler()
    g2p         = G2PEngine()
    prosody     = ProsodyFormatter()
    tts         = TTSEngineV2(lang=AUDIO_LANG, backend=TTS_BACKEND)

    log._logger.info(f'TTS backend active: {tts.active_backend}')
    return normalizer, detector, ner, g2p, prosody, tts, log


def run_pipeline(raw_text: str, normalizer, detector, ner, g2p, prosody, tts, log, idx: int):
    """
    Runs a single text through the full production pipeline.

    Returns:
        (normalized_text, prosody_text, before_path, after_path)
    """
    t0 = time.perf_counter()

    # Stage 1: Text Normalization (rule-based + BanglaT5)
    normalized = normalizer.normalize(raw_text)
    log.log_pipeline_stage('TextNormalizer', raw_text, normalized)

    # Stage 2: Language Detection
    routing = detector.route(normalized)
    if routing.is_mixed:
        log._logger.info(f'[{idx}] Mixed language detected — {routing.language_summary}')
        # Translate remaining English tokens via G2P dict
        en_to_bn = {
            tok.lower(): g2p.translate_english_word(tok)
            for tok in routing.english_tokens
        }
        normalized = detector.reconstruct_with_translations(normalized, en_to_bn)
        log.log_pipeline_stage('LanguageDetector', routing.original_text, normalized)

    # Stage 3: NER — entity pronunciation override
    ner_result = ner.handle(normalized)
    if ner_result != normalized:
        log.log_pipeline_stage('NERHandler', normalized, ner_result)
    normalized = ner_result

    # Stage 4: Prosody formatting
    annotation    = prosody.annotate(normalized)
    prosody_text  = annotation.formatted_text
    log.log_pipeline_stage('ProsodyFormatter', normalized, prosody_text)

    dt = (time.perf_counter() - t0) * 1000
    log.log_normalization(raw_text, normalized, duration_ms=dt)

    # Stage 5: TTS generation (before & after audio)
    before_path = os.path.join(OUTPUT_DIR, f'audio_before_{idx}.mp3')
    after_path  = os.path.join(OUTPUT_DIR, f'audio_after_{idx}.mp3')

    print(f'    Generating Before audio...')
    tts.generate_audio(raw_text, before_path)

    print(f'    Generating After audio...')
    tts.generate_audio(prosody_text, after_path)

    return normalized, prosody_text, before_path, after_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(TEST_CASES_FILE):
        print(f"Error: {TEST_CASES_FILE} not found.")
        return

    # Build pipeline
    normalizer, detector, ner, g2p, prosody, tts, log = build_pipeline()

    # Load test cases
    with open(TEST_CASES_FILE, 'r', encoding='utf-8') as f:
        all_cases = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    demo_cases = all_cases[:MAX_DEMO_CASES] if MAX_DEMO_CASES else all_cases

    print()
    print('=' * 65)
    print('  BANGLA TTS — PRODUCTION NORMALIZATION PIPELINE (v2)')
    print(f'  TTS Backend: {tts.active_backend}')
    print('=' * 65)

    for i, raw_text in enumerate(demo_cases, start=1):
        print(f'\n[{i}/{len(demo_cases)}] Input:  {raw_text}')

        norm, prose, before_p, after_p = run_pipeline(
            raw_text, normalizer, detector, ner, g2p, prosody, tts, log, i
        )

        print(f'    Normalized:  {norm}')
        print(f'    Prosody:     {prose}')
        print(f'    Audio saved: {before_p} | {after_p}')
        print('-' * 65)

    print(f'\nAll audio saved to: {OUTPUT_DIR}/')
    log.print_summary()


if __name__ == '__main__':
    main()