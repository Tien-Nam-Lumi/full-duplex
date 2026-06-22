import logging
import numpy as np

logger = logging.getLogger(__name__)

GLOBAL_SILERO_MODEL = None

class SileroVAD:
    """Stateful wrapper around Silero VAD for chunk-by-chunk real-time processing."""

    def __init__(self, threshold: float = 0.40, hangover_frames: int = 10) -> None:
        self.threshold = threshold
        self.hangover_frames = hangover_frames
        self._silero_model = None
        self._vad_sample_buffer = np.zeros(0, dtype=np.float32)
        self._silero_speech_state = False
        self._silero_hangover_frames = 0

    def _load_silero_vad(self):
        """Lazy-load Silero VAD model."""
        global GLOBAL_SILERO_MODEL
        if self._silero_model is None:
            if GLOBAL_SILERO_MODEL is not None:
                self._silero_model = GLOBAL_SILERO_MODEL
                return self._silero_model
                
            logger.info("Loading Silero VAD for pipeline...")
            import torch
            from pathlib import Path
            
            # Try loading from local hub cache to avoid GitHub network checks
            local_cache_path = Path.home() / ".cache/torch/hub/snakers4_silero-vad_master"
            if local_cache_path.exists():
                try:
                    logger.info(f"Loading Silero VAD from local cache: {local_cache_path}")
                    model, _ = torch.hub.load(
                        str(local_cache_path), 'silero_vad',
                        source='local'
                    )
                    GLOBAL_SILERO_MODEL = model
                    self._silero_model = GLOBAL_SILERO_MODEL
                    logger.info("✓ Silero VAD ready for pipeline (loaded from local cache)")
                    return self._silero_model
                except Exception as e:
                    logger.warning(f"Failed to load from local cache: {e}. Falling back to github...")
                    
            # Fallback to github download
            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', model='silero_vad',
                trust_repo=True,
            )
            GLOBAL_SILERO_MODEL = model
            self._silero_model = GLOBAL_SILERO_MODEL
            logger.info("✓ Silero VAD ready for pipeline")
        return self._silero_model

    def process_chunk(self, chunk: np.ndarray, threshold: float | None = None) -> bool:
        """Run stateful Silero VAD on 1280-sample chunk by slicing into 512-sample frames."""
        import torch
        model = self._load_silero_vad()
        
        current_threshold = threshold if threshold is not None else self.threshold
        
        self._vad_sample_buffer = np.concatenate([self._vad_sample_buffer, chunk])
        
        speech_detected = False
        
        while len(self._vad_sample_buffer) >= 512:
            frame = self._vad_sample_buffer[:512]
            self._vad_sample_buffer = self._vad_sample_buffer[512:]
            
            tensor = torch.from_numpy(frame).float()
            with torch.no_grad():
                prob = model(tensor, 16000).item()
            
            if prob >= current_threshold:
                self._silero_speech_state = True
                self._silero_hangover_frames = self.hangover_frames
            else:
                if self._silero_hangover_frames > 0:
                    self._silero_hangover_frames -= 1
                else:
                    self._silero_speech_state = False
            
            if self._silero_speech_state:
                speech_detected = True
                
        return speech_detected

    def reset(self) -> None:
        """Reset stateful VAD states."""
        self._silero_speech_state = False
        self._silero_hangover_frames = 0
        self._vad_sample_buffer = np.zeros(0, dtype=np.float32)

    @property
    def speech_state(self) -> bool:
        return self._silero_speech_state
