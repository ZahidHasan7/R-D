import os
from io import BytesIO
from typing import Optional

def synthesize(text: str) -> bytes:
    """Simple TTS wrapper that tries existing repo scripts.
    Returns WAV bytes or raises RuntimeError on failure.
    """
    # Try tts_engine_v2 if it exposes a synthesize or infer function
    try:
        from TTS.src import tts_engine_v2 as engine
        if hasattr(engine, "synthesize"):
            wav = engine.synthesize(text)
            # expect wav bytes or numpy array; try to convert
            if isinstance(wav, bytes):
                return wav
            try:
                import soundfile as sf
                import numpy as np
                bio = BytesIO()
                sf.write(bio, np.asarray(wav), 22050, format="WAV")
                return bio.getvalue()
            except Exception:
                pass
    except Exception:
        pass

    # Try scripts/infer_vits2.py as a CLI invocation
    try:
        repo_root = os.path.dirname(os.path.dirname(__file__))
        script = os.path.join(repo_root, "TTS", "scripts", "infer_vits2.py")
        if os.path.exists(script):
            # call script producing a file and read it
            import subprocess, tempfile
            out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            out.close()
            cmd = ["python", script, "--text", text, "--out", out.name]
            subprocess.check_call(cmd)
            with open(out.name, "rb") as f:
                data = f.read()
            os.unlink(out.name)
            return data
    except Exception:
        pass

    raise RuntimeError("No usable TTS inference available in repository.")
