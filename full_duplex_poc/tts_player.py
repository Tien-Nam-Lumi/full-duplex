import os
import time
import threading
import logging
from pathlib import Path
import soundfile as sf

logger = logging.getLogger(__name__)

class BrowserAudioTTSPlayer:
    """
    SLURM/headless-safe TTS player.

    Does not play audio on the server.
    It only tracks logical playback state.
    The browser plays the WAV via Gradio Audio output.
    """

    def __init__(self, wav_filename: str = "assistant_long_response.wav"):
        self.wav_path = str(Path(__file__).parent / wav_filename)
        self._playing = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._playback_id = 0

    def start_tts(self, text: str | None = None) -> str | None:
        wav_file = "assistant_long_response.wav"
        if text == "Đã bật đèn phòng khách.":
            wav_file = "assistant_short_response.wav"
            
        self.wav_path = str(Path(__file__).parent / wav_file)

        if not os.path.exists(self.wav_path):
            logger.error(f"Assistant WAV not found: {self.wav_path}")
            return None

        if text:
            logger.info(f"[BROWSER TTS STARTED] {text}")
            
        try:
            duration = sf.info(self.wav_path).duration
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            duration = 5.0 # fallback

        with self._lock:
            self._playback_id += 1
            playback_id = self._playback_id
            self._playing = True
            self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._finish_after_duration,
            args=(duration, playback_id),
            daemon=True,
        )
        self._thread.start()

        return self.wav_path

    def _finish_after_duration(self, duration: float, playback_id: int):
        start = time.time()
        while time.time() - start < duration:
            if self._stop_event.is_set():
                return
            time.sleep(0.05)

        with self._lock:
            if playback_id == self._playback_id:
                self._playing = False
        logger.info(f"[BROWSER TTS] Finished logically after {duration:.1f}s")

    def stop_tts(self):
        with self._lock:
            self._playback_id += 1
            self._playing = False
            self._stop_event.set()
        logger.info("[BROWSER TTS] stopped logically")

    def clear_tts_queue(self):
        pass

    def is_tts_playing(self) -> bool:
        with self._lock:
            return self._playing

    def get_audio_path(self) -> str:
        return self.wav_path

class MockTTSPlayer:
    """
    Silent timing-only mode.
    Useful for SLURM offline tests where we only care about detector logic.
    """

    def __init__(self):
        self._playing = False
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def start_tts(self, text: str | None = None) -> bool:
        with self._lock:
            self._playing = True
            self._stop_event.clear()

        self._thread = threading.Thread(target=self._mock_loop, daemon=True)
        self._thread.start()
        return True

    def _mock_loop(self):
        for _ in range(1200):  # roughly 60 seconds (1200 * 0.05) long enough for demo
            if self._stop_event.is_set():
                break
            time.sleep(0.05)

        with self._lock:
            self._playing = False

    def stop_tts(self):
        with self._lock:
            self._playing = False
            self._stop_event.set()

    def clear_tts_queue(self):
        pass

    def is_tts_playing(self) -> bool:
        with self._lock:
            return self._playing

class LocalAudioTTSPlayer:
    """
    Local laptop only. Do not use on SLURM.
    """

    def __init__(self, wav_filename: str = "assistant_long_response.wav"):
        import soundfile as sf

        self._playing = False
        self._wav_path = str(Path(__file__).parent / wav_filename)
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._audio_data = None
        self._sr = None

        if os.path.exists(self._wav_path):
            self._audio_data, self._sr = sf.read(self._wav_path)
            if self._audio_data.ndim > 1:
                self._audio_data = self._audio_data.mean(axis=1)
        else:
            logger.warning(f"Audio file not found: {self._wav_path}")

    def start_tts(self, text: str | None = None) -> bool:
        if self._audio_data is None:
            return False

        with self._lock:
            if not self._playing:
                self._playing = True
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._play_loop, daemon=True)
                self._thread.start()

        return True

    def _play_loop(self):
        import numpy as np
        import sounddevice as sd

        try:
            chunk_size = int(self._sr * 0.1)
            with sd.OutputStream(samplerate=self._sr, channels=1) as stream:
                offset = 0
                while offset < len(self._audio_data):
                    if self._stop_event.is_set():
                        break

                    end = min(offset + chunk_size, len(self._audio_data))
                    chunk = self._audio_data[offset:end]
                    stream.write(np.expand_dims(chunk, axis=1).astype(np.float32))
                    offset = end
        finally:
            with self._lock:
                self._playing = False

    def stop_tts(self):
        self._stop_event.set()

    def clear_tts_queue(self):
        pass

    def is_tts_playing(self) -> bool:
        with self._lock:
            return self._playing

def create_tts_player():
    mode = os.getenv("TTS_MODE", "browser").lower()

    if mode == "local":
        return LocalAudioTTSPlayer(), mode

    if mode == "mock":
        return MockTTSPlayer(), mode

    return BrowserAudioTTSPlayer(), "browser"
