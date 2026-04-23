"""
tts_engine_v2.py — Upgraded TTS Engine with Strategy Pattern.

Supports multiple backends via TTSBackend enum:
  - TTSBackend.GTTS  : Google TTS (gTTS) — online, always available
  - TTSBackend.VITS  : VITS neural TTS — local, offline, higher quality
  - TTSBackend.AUTO  : Try VITS first, fall back to gTTS on failure

Maintains the same API as the original tts_engine.py so it is a
drop-in replacement.
"""

import os
import time
import warnings
from enum import Enum, auto
from typing import Optional


class TTSBackend(Enum):
    GTTS = auto()   # Google TTS (gTTS)
    VITS = auto()   # VITS neural TTS (local, offline)
    AUTO = auto()   # Auto-select: VITS → gTTS fallback


class TTSEngineV2:
    """
    Upgraded TTS Engine with pluggable backend support.

    Usage:
        # Auto mode (VITS if available, else gTTS)
        engine = TTSEngineV2(lang='bn', backend=TTSBackend.AUTO)
        engine.generate_audio("পাঁচ কেজি চাল", "output.mp3")

        # Force gTTS
        engine = TTSEngineV2(lang='bn', backend=TTSBackend.GTTS)

        # Force VITS (requires model to be set up)
        engine = TTSEngineV2(lang='bn', backend=TTSBackend.VITS, vits_model_path='/path/to/model')
    """

    def __init__(
        self,
        lang: str = 'bn',
        backend: TTSBackend = TTSBackend.AUTO,
        vits_model_path: Optional[str] = None,
        slow: bool = False,
    ):
        """
        Args:
            lang:             Language code (default 'bn' for Bangla)
            backend:          TTSBackend enum value
            vits_model_path:  Path to local VITS model directory (optional)
            slow:             Slow speech for gTTS backend (default False)
        """
        self.lang = lang
        self.backend = backend
        self.vits_model_path = vits_model_path
        self.slow = slow
        self._vits_model = None
        self._vits_available = False

        if backend in (TTSBackend.VITS, TTSBackend.AUTO):
            self._try_load_vits()

    # ------------------------------------------------------------------
    # VITS setup
    # ------------------------------------------------------------------

    def _try_load_vits(self) -> None:
        """
        Attempts to load a VITS model.
        Silently marks VITS as unavailable if setup fails.
        """
        try:
            # Attempt import — TTS (Coqui) package provides VITS models
            from TTS.api import TTS as CoquiTTS  # type: ignore
            if self.vits_model_path:
                self._vits_model = CoquiTTS(model_path=self.vits_model_path)
            else:
                # Try to load a default multilingual VITS model that includes Bangla
                # Model: "tts_models/multilingual/multi-dataset/your_tts"
                # NOTE: Replace model name once a Bangla-specific VITS model is available
                warnings.warn(
                    "VITS: No model path provided. Specify vits_model_path to use VITS. "
                    "Falling back to gTTS.",
                    RuntimeWarning,
                    stacklevel=2
                )
                self._vits_available = False
                return
            self._vits_available = True
        except (ImportError, Exception):
            # Coqui TTS not installed or model load failed → graceful fallback
            self._vits_available = False

    # ------------------------------------------------------------------
    # Audio generation
    # ------------------------------------------------------------------

    def generate_audio(
        self,
        text: str,
        output_path: str,
        backend: Optional[TTSBackend] = None,
        strip_markers: bool = True,
    ) -> bool:
        """
        Generates speech audio for the given text.

        Args:
            text:          Input text (may contain prosody markers)
            output_path:   Path to save the output audio file (.mp3 or .wav)
            backend:       Override the instance-level backend for this call
            strip_markers: Strip SSML-like prosody markers before synthesis (default True)

        Returns:
            True on success, False on failure.
        """
        if not text:
            return False

        # Strip prosody markers if the backend can't handle them
        if strip_markers:
            text = self._strip_prosody_markers(text)

        effective_backend = backend or self.backend

        if effective_backend == TTSBackend.AUTO:
            # Try VITS first
            if self._vits_available:
                success = self._generate_vits(text, output_path)
                if success:
                    return True
                warnings.warn("VITS generation failed, falling back to gTTS.", RuntimeWarning)
            return self._generate_gtts(text, output_path)

        if effective_backend == TTSBackend.VITS:
            if not self._vits_available:
                raise RuntimeError(
                    "VITS backend requested but model is not available. "
                    "Install Coqui TTS and provide a valid vits_model_path."
                )
            return self._generate_vits(text, output_path)

        # Default: gTTS
        return self._generate_gtts(text, output_path)

    def _generate_gtts(self, text: str, output_path: str) -> bool:
        """Generates audio using Google TTS (gTTS)."""
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang=self.lang, slow=self.slow)
            tts.save(output_path)
            return True
        except Exception as e:
            warnings.warn(f"gTTS generation failed: {e}", RuntimeWarning)
            return False

    def _generate_vits(self, text: str, output_path: str) -> bool:
        """Generates audio using the loaded VITS model."""
        if not self._vits_available or self._vits_model is None:
            return False
        try:
            # Coqui TTS API: model.tts_to_file(text, file_path)
            self._vits_model.tts_to_file(text=text, file_path=output_path)
            return True
        except Exception as e:
            warnings.warn(f"VITS generation failed: {e}", RuntimeWarning)
            return False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_prosody_markers(text: str) -> str:
        """Removes SSML-lite prosody markers from text."""
        import re
        text = re.sub(r'<pause:\w+>', ' ', text)
        text = re.sub(r'</?emphasis>', '', text)
        return re.sub(r'  +', ' ', text).strip()

    @property
    def active_backend(self) -> str:
        """Returns the effective backend name in use."""
        if self.backend == TTSBackend.AUTO:
            return 'VITS' if self._vits_available else 'gTTS'
        return self.backend.name

    def __repr__(self) -> str:
        return (
            f"TTSEngineV2(lang={self.lang!r}, "
            f"backend={self.backend.name}, "
            f"active={self.active_backend})"
        )


if __name__ == '__main__':
    engine = TTSEngineV2(lang='bn', backend=TTSBackend.GTTS)
    print(f"Engine: {engine}")
    success = engine.generate_audio("পাঁচ কেজি চাল", "/tmp/test_v2.mp3")
    print(f"Generated: {success}")
