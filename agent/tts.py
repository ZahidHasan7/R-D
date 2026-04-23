import os
from io import BytesIO
from typing import Optional

def synthesize(text: str) -> bytes:
    """Simple TTS wrapper that tries existing repo scripts.
    Returns WAV bytes or raises RuntimeError on failure.
    """
    # Prefer using the repository's infer_vits2.synthesize function
    try:
        repo_root = os.path.dirname(os.path.dirname(__file__))
        sys_path_added = False
        if repo_root not in os.sys.path:
            os.sys.path.insert(0, repo_root)
            sys_path_added = True
        from TTS.scripts.infer_vits2 import synthesize as vits_synth
        import tempfile
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out.close()
        vits_synth(text, out.name)
        with open(out.name, "rb") as f:
            data = f.read()
        try:
            os.unlink(out.name)
        except Exception:
            pass
        if sys_path_added:
            try:
                os.sys.path.remove(repo_root)
            except Exception:
                pass
        return data
    except Exception as e:
        raise RuntimeError(f"No usable TTS inference available: {e}")
