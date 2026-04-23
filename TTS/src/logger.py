"""
logger.py — Structured logging for the Bangla TTS normalization pipeline.

Tracks:
  - Failed normalizations
  - Unknown words (G2P fallback)
  - ML translator invocations
  - Pipeline timing stats

Writes to logs/normalization.log with timestamps.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

_LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
_LOG_FILE = os.path.join(_LOG_DIR, 'normalization.log')


def _ensure_log_dir() -> None:
    os.makedirs(_LOG_DIR, exist_ok=True)


@dataclass
class PipelineStats:
    """Runtime statistics accumulated during a normalization session."""
    total_texts: int = 0
    failed_normalizations: int = 0
    ml_translator_calls: int = 0
    g2p_fallback_words: list[str] = field(default_factory=list)
    unknown_words: list[str] = field(default_factory=list)
    total_processing_time_ms: float = 0.0

    @property
    def avg_processing_time_ms(self) -> float:
        if self.total_texts == 0:
            return 0.0
        return self.total_processing_time_ms / self.total_texts

    def to_dict(self) -> dict:
        return {
            'total_texts': self.total_texts,
            'failed_normalizations': self.failed_normalizations,
            'ml_translator_calls': self.ml_translator_calls,
            'unique_g2p_fallback_words': list(set(self.g2p_fallback_words)),
            'unique_unknown_words': list(set(self.unknown_words)),
            'avg_processing_time_ms': round(self.avg_processing_time_ms, 2),
            'total_processing_time_ms': round(self.total_processing_time_ms, 2),
        }


class NormalizationLogger:
    """
    Structured logger for the TTS normalization pipeline.

    Usage:
        logger = NormalizationLogger()
        logger.log_normalization("3.5kg apple", "তিন দশমিক পাঁচ কেজি আপেল")
        logger.log_ml_call("3.5kg apple box")
        logger.log_g2p_fallback("apple")
        logger.log_unknown_word("xyzxyz")
        summary = logger.get_summary()
        print(summary)
    """

    _instance: Optional['NormalizationLogger'] = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern — ensures one logger per process."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_level: int = logging.INFO):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.stats = PipelineStats()
        self._start_time = time.time()

        _ensure_log_dir()

        self._logger = logging.getLogger('bangla_tts_normalizer')
        self._logger.setLevel(log_level)

        # Avoid duplicate handlers on re-instantiation
        if not self._logger.handlers:
            # File handler
            fh = logging.FileHandler(_LOG_FILE, encoding='utf-8')
            fh.setLevel(log_level)
            fmt = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)

            # Console handler (INFO and above)
            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)
            ch.setFormatter(fmt)
            self._logger.addHandler(ch)

        self._logger.info('=' * 60)
        self._logger.info('NormalizationLogger initialized')

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_normalization(
        self,
        raw: str,
        normalized: str,
        duration_ms: float = 0.0,
        success: bool = True,
    ) -> None:
        """Logs a single normalization event."""
        self.stats.total_texts += 1
        self.stats.total_processing_time_ms += duration_ms

        if success:
            self._logger.info(
                f'NORM | raw={raw!r} | out={normalized!r} | {duration_ms:.1f}ms'
            )
        else:
            self.stats.failed_normalizations += 1
            self._logger.warning(
                f'NORM_FAIL | raw={raw!r} | out={normalized!r} | {duration_ms:.1f}ms'
            )

    def log_ml_call(self, text: str) -> None:
        """Logs an ML translator invocation."""
        self.stats.ml_translator_calls += 1
        self._logger.info(f'ML_CALL | text={text!r}')

    def log_g2p_fallback(self, word: str) -> None:
        """Logs a word that fell through to the G2P neural fallback."""
        self.stats.g2p_fallback_words.append(word)
        self._logger.debug(f'G2P_FALLBACK | word={word!r}')

    def log_unknown_word(self, word: str) -> None:
        """Logs a word that no module could process."""
        self.stats.unknown_words.append(word)
        self._logger.warning(f'UNKNOWN_WORD | word={word!r}')

    def log_error(self, message: str, exc: Optional[Exception] = None) -> None:
        """Logs a pipeline error."""
        self.stats.failed_normalizations += 1
        if exc:
            self._logger.error(f'ERROR | {message} | {type(exc).__name__}: {exc}')
        else:
            self._logger.error(f'ERROR | {message}')

    def log_pipeline_stage(self, stage: str, text_before: str, text_after: str) -> None:
        """Logs the text at a specific pipeline stage for debugging."""
        if text_before != text_after:
            self._logger.debug(
                f'STAGE={stage} | before={text_before!r} | after={text_after!r}'
            )

    # ------------------------------------------------------------------
    # Summary and reporting
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Returns the accumulated pipeline statistics as a dictionary."""
        elapsed = round((time.time() - self._start_time) * 1000, 2)
        summary = self.stats.to_dict()
        summary['session_duration_ms'] = elapsed
        return summary

    def print_summary(self) -> None:
        """Prints a formatted summary report to stdout."""
        s = self.get_summary()
        print('\n' + '=' * 55)
        print('  NORMALIZATION PIPELINE — SESSION SUMMARY')
        print('=' * 55)
        print(f"  Total texts processed  : {s['total_texts']}")
        print(f"  Failed normalizations  : {s['failed_normalizations']}")
        print(f"  ML translator calls    : {s['ml_translator_calls']}")
        print(f"  G2P fallback words     : {len(s['unique_g2p_fallback_words'])}")
        print(f"  Unknown words          : {len(s['unique_unknown_words'])}")
        print(f"  Avg processing time    : {s['avg_processing_time_ms']:.1f}ms/text")
        print(f"  Session duration       : {s['session_duration_ms']:.0f}ms")
        if s['unique_unknown_words']:
            print(f"\n  Unknown words: {s['unique_unknown_words']}")
        print('=' * 55)
        self._logger.info(f'SESSION_SUMMARY | {s}')

    def reset(self) -> None:
        """Resets accumulated stats (useful for test isolation)."""
        self.stats = PipelineStats()
        self._start_time = time.time()


# Module-level convenience instance
_default_logger: Optional[NormalizationLogger] = None


def get_logger() -> NormalizationLogger:
    """Returns the shared singleton NormalizationLogger instance."""
    global _default_logger
    if _default_logger is None:
        _default_logger = NormalizationLogger()
    return _default_logger


if __name__ == '__main__':
    log = get_logger()
    log.log_normalization('৳100', 'একশ টাকা', duration_ms=1.2)
    log.log_ml_call('3.5kg apple')
    log.log_g2p_fallback('xyzword')
    log.log_unknown_word('unknownterm')
    log.print_summary()
