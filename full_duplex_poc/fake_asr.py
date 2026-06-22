import time
import logging

logger = logging.getLogger(__name__)

class FakeASR:
    """Mock ASR that simply sleeps and returns a predefined command."""
    
    @staticmethod
    def transcribe(audio_buffer):
        logger.info(f"[FakeASR] Transcribing audio buffer of {len(audio_buffer)} samples...")
        time.sleep(0.5)  # Simulate API/Inference latency
        
        command = "bật đèn phòng khách"
        logger.info(f"[FakeASR] Result: {command}")
        return command
