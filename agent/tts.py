import os
import sys
import torch
import numpy as np
import wave
import subprocess
import tempfile
import librosa
import scipy.signal
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import threading

# Global instances for lazy loading and performance
# Note: These are initialized in _init_vits()
_mms_bn = {'model': None, 'tokenizer': None}
_xtts_en = None
_normalizer = None
_detector = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Locks for thread safety
_xtts_lock = threading.Lock()

# Naturalness parameters
_segment_pause_ms = 150 # Pause between Bangla/English segments

def _init_vits():
    """Lazily initialize the VITS model, XTTS, and normalizers."""
    global _mms_bn, _xtts_en, _normalizer, _detector
    if _mms_bn['model'] is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        tts_root = os.path.join(project_root, 'TTS')
        if tts_root not in sys.path:
            sys.path.append(tts_root)
        
        try:
            from transformers import VitsModel, AutoTokenizer
            from src.normalizer import TextNormalizer
            from src.language_detector import LanguageDetector
            from TTS.api import TTS as CoquiTTS
            
            print(f"[*] Loading Bangla VITS (facebook/mms-tts-ben)...")
            _mms_bn['tokenizer'] = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
            _mms_bn['model'] = VitsModel.from_pretrained("facebook/mms-tts-ben").to(_device)
            
            print("[*] Loading English XTTS v2 (Cloning Engine)...")
            _xtts_en = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(_device)
            
            print("[*] Initializing Language Detector and Normalizer...")
            _normalizer = TextNormalizer(use_ml=True)
            _detector = LanguageDetector()
            
            print("[+] Hybrid TTS pipeline initialized successfully.")
        except Exception as e:
            import traceback
            print(f"[!] Error initializing TTS pipeline: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"Could not load neural TTS: {e}")

def _synthesize_worker(args):
    """Worker function to synthesize a single segment."""
    lang, chunk, ref_wav = args
    if not chunk.strip():
        return None
        
    print(f"[*] Processing {lang} segment in parallel: {chunk[:20]}...")
    
    try:
        if lang == 'bangla':
            normalized = _normalizer.normalize(chunk)
            inputs = _mms_bn['tokenizer'](normalized, return_tensors="pt").to(_device)
            with torch.no_grad():
                output = _mms_bn['model'](**inputs).waveform
            
            wav = output[0].cpu().numpy()
            sr = _mms_bn['model'].config.sampling_rate
            
            if sr != 24000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
            return wav
            
        else: # english
            temp_out = tempfile.mktemp(suffix=".wav")
            try:
                # XTTS v2 can be unstable on CPU with high concurrency, use a lock
                with _xtts_lock:
                    _xtts_en.tts_to_file(
                        text=chunk,
                        speaker_wav=ref_wav,
                        language="en",
                        file_path=temp_out
                    )
                wav, _ = librosa.load(temp_out, sr=24000)
                return wav
            finally:
                if os.path.exists(temp_out):
                    os.remove(temp_out)
    except Exception as e:
        print(f"[!] Worker failed for {lang} segment: {e}")
        return None

def synthesize(text: str, apply_filter: bool = True) -> bytes:
    """
    True Hybrid TTS via Parallel Language Segmentation.
    """
    if not text:
        return b''
        
    _init_vits()
    
    from src.utils.segmenter import segment_text
    segments = segment_text(text, _detector)
    
    # Filter empty chunks
    active_segments = [(lang, chunk) for lang, chunk in segments if chunk.strip()]
    if not active_segments:
        return b''
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ref_wav = os.path.join(current_dir, '..', 'ref_speaker.wav')
    sample_rate = 24000

    # Parallelize synthesis across segments
    worker_args = [(lang, chunk, ref_wav) for lang, chunk in active_segments]
    
    # Use a lock for XTTS as it's not fully thread-safe on CPU
    print(f"[*] Dispatching {len(worker_args)} segments to ThreadPoolExecutor...")
    with ThreadPoolExecutor(max_workers=min(len(worker_args), 4)) as executor:
        segment_wavs = list(executor.map(_synthesize_worker, worker_args))

    # Combine with naturalness pauses
    combined_wavs = []
    for i, wav in enumerate(segment_wavs):
        if wav is None:
            continue
            
        # Add pause between segments (except before the first one)
        if combined_wavs and _segment_pause_ms > 0:
            pause_samples = int(sample_rate * (_segment_pause_ms / 1000.0))
            combined_wavs.append(np.zeros(pause_samples))
            
        combined_wavs.append(wav)

    if not combined_wavs:
        return b''
        
    full_audio = np.concatenate(combined_wavs)

    # Audio Post-Processing (Optional for benchmarking)
    if apply_filter:
        try:
            def bandpass_filter(data, lowcut, highcut, fs, order=3):
                nyq = 0.5 * fs
                low, high = lowcut / nyq, highcut / nyq
                b, a = scipy.signal.butter(order, [low, high], btype='band')
                return scipy.signal.lfilter(b, a, data)
            full_audio = bandpass_filter(full_audio, 300.0, 3400.0, sample_rate)
        except Exception as e:
            print(f"[!] Post-processing failed: {e}")

    # Normalize volume
    max_val = np.max(np.abs(full_audio))
    if max_val > 0:
        full_audio = full_audio / max_val
        
    pcm_data = (full_audio * 32767).astype(np.int16)
    
    bio = BytesIO()
    with wave.open(bio, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data.tobytes())
    
    return bio.getvalue()
