import os
import tempfile
from typing import Optional

def transcribe(file_path: str) -> str:
    """Simple wrapper that tries to use existing STT scripts.
    Falls back to returning an error string if inference isn't available.
    """
    # Try to call a known STT entry point
    try:
        # prefer baseline_1A if it exposes a transcribe function
        import STT.baseline_1A as baseline
        if hasattr(baseline, "transcribe"):
            return baseline.transcribe(file_path)
    except Exception:
        pass

    try:
        # try approach_2A_large
        import STT.approach_2A_large as a2
        if hasattr(a2, "transcribe"):
            return a2.transcribe(file_path)
    except Exception:
        pass

    # Last-resort: attempt a very small builtin fallback using whisper (if installed)
    try:
        from whisper import load_model
        model = load_model("small")
        result = model.transcribe(file_path)
        return result.get("text", "")
    except Exception:
        pass

    return "[ERROR] No usable STT inference available in repository."
