import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

def load_audio_mono_16k(input_audio) -> np.ndarray:
    """Load audio from various input sources, return mono float32 16kHz array.
    """
    if input_audio is None:
        return np.array([], dtype=np.float32)
        
    if isinstance(input_audio, dict):
        path = input_audio.get("path") or input_audio.get("name")
        if path:
            input_audio = path
        else:
            raise ValueError(f"Invalid dictionary format for audio input: {input_audio}")
            
    if isinstance(input_audio, str):
        # Filepath input
        audio_arr, sr = sf.read(input_audio, dtype="float32")
    elif isinstance(input_audio, tuple) and len(input_audio) == 2:
        # Tuple input: (sr, np.ndarray)
        sr, audio_arr = input_audio
    else:
        # Direct numpy array (fallback assume 16kHz)
        if isinstance(input_audio, np.ndarray):
            audio_arr = input_audio
            sr = 16000
        else:
            raise ValueError(f"Unsupported audio input type: {type(input_audio)}")
            
    # Ensure it is a numpy array
    audio_arr = np.ascontiguousarray(audio_arr)
    
    # Downmix if multi-channel
    if audio_arr.ndim > 1:
        if audio_arr.shape[0] < audio_arr.shape[1]:
            # shape (num_channels, num_samples)
            audio_arr = audio_arr.mean(axis=0)
        else:
            # shape (num_samples, num_channels)
            audio_arr = audio_arr.mean(axis=1)
            
    # Convert integer PCM to float32
    if np.issubdtype(audio_arr.dtype, np.integer):
        max_val = float(np.iinfo(audio_arr.dtype).max)
        audio_arr = audio_arr.astype(np.float32) / max_val
    else:
        audio_arr = audio_arr.astype(np.float32)
        
    # Resample to 16 kHz if needed
    if sr != 16000:
        g = gcd(16000, sr)
        audio_arr = resample_poly(audio_arr, 16000 // g, sr // g).astype(np.float32)
        
    # Flatten/ravel to 1D
    audio_arr = audio_arr.ravel()
    
    # Sanitize NaN/Inf
    audio_arr = np.nan_to_num(audio_arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    return audio_arr
