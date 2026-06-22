import numpy as np
import logging
from .audio_types import CHUNK_MS

logger = logging.getLogger(__name__)

MIN_UTTERANCE_MS = 500
SILENCE_BEFORE_SMARTTURN_MS = 300
MAX_SILENCE_FALLBACK_MS = 900

class PostInterruptTurnCollector:
    def __init__(self, vad, turn_detector):
        self.vad = vad
        self.turn_detector = turn_detector
        self.reset()

    def reset(self):
        self.audio_chunks = []
        self.speech_ms = 0
        self.silence_ms = 0
        self.started = False

    def start(self, initial_audio: np.ndarray = None):
        self.reset()
        self.started = True
        
        if initial_audio is not None and len(initial_audio) > 0:
            self.audio_chunks.append(initial_audio)
            # Rough estimate of initial speech ms
            self.speech_ms += (len(initial_audio) / 16000) * 1000

    def process_chunk(self, chunk: np.ndarray):
        if not self.started:
            return {"turn_finalized": False}

        is_speech = self.vad.process_chunk(chunk)
        self.audio_chunks.append(chunk.copy())

        if is_speech:
            self.speech_ms += CHUNK_MS
            self.silence_ms = 0
        else:
            self.silence_ms += CHUNK_MS

        # Log periodic updates for debugging
        if self.silence_ms % 100 == 0 and self.silence_ms > 0:
            logger.debug(f"[Collector] silence_ms={self.silence_ms}, speech_ms={self.speech_ms:.1f}")

        if self.speech_ms < MIN_UTTERANCE_MS:
            return {"turn_finalized": False}

        if self.silence_ms >= SILENCE_BEFORE_SMARTTURN_MS:
            audio = np.concatenate(self.audio_chunks)
            pred = self.turn_detector.predict(audio)

            if pred["finished"]:
                return {
                    "turn_finalized": True,
                    "reason": "smartturn",
                    "audio": audio,
                    "smartturn_score": pred.get("score"),
                }

        if self.silence_ms >= MAX_SILENCE_FALLBACK_MS:
            return {
                "turn_finalized": True,
                "reason": "silence_fallback",
                "audio": np.concatenate(self.audio_chunks),
                "smartturn_score": None,
            }

        return {"turn_finalized": False}
