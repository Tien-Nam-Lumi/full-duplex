import logging
import time
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from .audio_types import CHUNK_MS, CHUNK_SIZE
from .vad_backend import SileroVAD
from .target_speaker_backend import extract_embedding

logger = logging.getLogger(__name__)

class InterruptionDetector:
    def __init__(self, target_embedding: np.ndarray, sv_threshold: float = 0.5):
        self.target_embedding = target_embedding
        self.sv_threshold = sv_threshold
        
        self.vad = SileroVAD()
        
        # Interruption config (Lower Latency for PoC)
        self.INTERRUPT_CONFIRM_MS = 400
        self.MIN_SV_CONTEXT_CHUNKS = 5
        self.SV_CHECK_EVERY_CHUNKS = 2
        self.MAX_ROLLING_CHUNKS = 30
        
        # Threading for non-blocking SV
        self.sv_executor = ThreadPoolExecutor(max_workers=1)
        self.sv_future = None
        self.current_buffer_id = 0
        
        # State tracking
        self.state = "IDLE"  # IDLE, LISTENING, SPEAKING, INTERRUPTED
        self.target_positive_ms = 0
        self.rolling_speech_buffer = deque(maxlen=self.MAX_ROLLING_CHUNKS)
        self.speech_chunks_since_sv = 0
        self.last_sv_score = 0.0
        self.is_target_verified = False
        self.speech_start_time = None
        self.silence_ms = 0
        
    def clear_interruption_state(self):
        self.target_positive_ms = 0
        self.rolling_speech_buffer.clear()
        self.speech_chunks_since_sv = 0
        self.last_sv_score = 0.0
        self.is_target_verified = False
        self.current_buffer_id += 1
        self.sv_future = None
        self.speech_start_time = None
        self.silence_ms = 0

    def reset(self):
        self.state = "IDLE"
        self.clear_interruption_state()
        self.vad.reset()

    def set_state(self, new_state: str):
        self.state = new_state
        logger.info(f"[STATE] {new_state}")

    def _run_sv_extraction(self, audio_data: np.ndarray, buffer_id: int):
        t0 = time.time()
        emb = extract_embedding(audio_data)
        elapsed = time.time() - t0
        return emb, buffer_id, elapsed

    def process_chunk(self, chunk: np.ndarray, tts_player) -> dict:
        """
        Process a 1280-sample chunk.
        If state is SPEAKING, evaluate interruption.
        Returns a dict with state and evaluation results.
        """
        is_speech = self.vad.process_chunk(chunk)
        
        interrupted = False
        stop_latency_ms = None
        
        # Check if SV future is done
        if self.sv_future is not None and self.sv_future.done():
            try:
                emb, buffer_id, elapsed = self.sv_future.result()
                if buffer_id == self.current_buffer_id:
                    if emb is not None:
                        self.last_sv_score = float(np.dot(emb, self.target_embedding))
                        self.is_target_verified = self.last_sv_score >= self.sv_threshold
                        logger.info(f"[SV] score={self.last_sv_score:.4f} (Target? {self.is_target_verified}) - inference took {elapsed*1000:.1f}ms")
                    else:
                        self.last_sv_score = 0.0
                        self.is_target_verified = False
            except Exception as e:
                logger.error(f"Error in SV extraction thread: {e}")
            self.sv_future = None

        # Capture the score before it gets wiped by clear_interruption_state
        event_sv_score = self.last_sv_score
        
        if self.state == "SPEAKING":
            if is_speech:
                self.silence_ms = 0
                if self.speech_start_time is None:
                    self.speech_start_time = time.time()
                    
                self.rolling_speech_buffer.append(chunk.copy())
                self.speech_chunks_since_sv += 1
                
                # Initiate SV score periodically if not already running
                if self.speech_chunks_since_sv >= self.SV_CHECK_EVERY_CHUNKS and len(self.rolling_speech_buffer) >= self.MIN_SV_CONTEXT_CHUNKS:
                    self.speech_chunks_since_sv = 0
                    if self.sv_future is None:
                        rolling_speech_flat = np.concatenate(list(self.rolling_speech_buffer))
                        self.sv_future = self.sv_executor.submit(self._run_sv_extraction, rolling_speech_flat, self.current_buffer_id)
            else:
                self.silence_ms += CHUNK_MS
                if self.silence_ms >= 320:
                    self.clear_interruption_state()
                
            # If currently actively recognized as speech and target, increase positive Ms
            if is_speech and self.is_target_verified:
                self.target_positive_ms += CHUNK_MS
            else:
                self.target_positive_ms = max(0, self.target_positive_ms - CHUNK_MS)
                
            if is_speech and self.is_target_verified:
                logger.info(f"[INTERRUPT] candidate_ms={self.target_positive_ms}")
                
            if self.target_positive_ms >= self.INTERRUPT_CONFIRM_MS:
                interrupted = True
                stop_latency_ms = (time.time() - self.speech_start_time) * 1000 if self.speech_start_time else self.target_positive_ms
                logger.info(f"[INTERRUPT] confirmed. Stop Latency (from speech start): {stop_latency_ms:.1f}ms")
                tts_player.stop_tts()
                tts_player.clear_tts_queue()
                self.set_state("INTERRUPTED")
                logger.info("TARGET INTERRUPTION DETECTED")
                self.set_state("LISTENING_AFTER_INTERRUPT")
                trigger_audio = np.concatenate(list(self.rolling_speech_buffer))
                self.clear_interruption_state()
                
        return {
            "state": self.state,
            "interrupted": interrupted,
            "stop_latency_ms": stop_latency_ms,
            "sv_score": event_sv_score,
            "is_target": self.is_target_verified,
            "is_speech": is_speech,
            "trigger_audio": trigger_audio if interrupted else None
        }
