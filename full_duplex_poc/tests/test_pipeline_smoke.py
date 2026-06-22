import numpy as np

import full_duplex_poc.pipeline as pipeline_module
from full_duplex_poc.config import FullDuplexConfig
from full_duplex_poc.pipeline import FullDuplexPipeline
from full_duplex_poc.state_machine import FullDuplexState


class FakeTTS:
    def __init__(self):
        self.playing = False
        self.stopped = False

    def start_tts(self, text=None):
        self.playing = True
        return "/tmp/fake_tts.wav"

    def stop_tts(self):
        self.playing = False
        self.stopped = True

    def clear_tts_queue(self):
        pass

    def is_tts_playing(self):
        return self.playing

    def get_audio_path(self):
        return "/tmp/fake_tts.wav"


class FakeDetector:
    def __init__(self):
        self.state = FullDuplexState.SPEAKING.value
        self.target_positive_ms = 0
        self.last_sv_score = 0.91
        self.is_target_verified = True
        self._calls = 0

    def set_state(self, state):
        self.state = state

    def reset(self):
        self.state = FullDuplexState.IDLE.value

    def clear_interruption_state(self):
        self.target_positive_ms = 0
        self.last_sv_score = 0.0
        self.is_target_verified = False

    def process_chunk(self, chunk, tts):
        self._calls += 1
        if self._calls == 1:
            return {
                "state": self.state,
                "interrupted": True,
                "stop_latency_ms": 120.0,
                "sv_score": 0.91,
                "is_target": True,
                "is_speech": True,
                "trigger_audio": chunk.copy(),
            }
        return {
            "state": self.state,
            "interrupted": False,
            "stop_latency_ms": None,
            "sv_score": 0.91,
            "is_target": True,
            "is_speech": True,
            "trigger_audio": None,
        }


class FakeTurnCollector:
    def __init__(self):
        self.started = False
        self.audio = []
        self.calls = 0

    def reset(self):
        self.started = False
        self.audio = []
        self.calls = 0

    def start(self, initial_audio=None):
        self.started = True
        self.audio = []
        if initial_audio is not None:
            self.audio.append(initial_audio.copy())

    def process_chunk(self, chunk):
        self.calls += 1
        self.audio.append(chunk.copy())
        if self.calls >= 2:
            return {
                "turn_finalized": True,
                "reason": "silence_fallback",
                "audio": np.concatenate(self.audio),
                "smartturn_score": None,
            }
        return {"turn_finalized": False}


def test_pipeline_interrupt_then_response(monkeypatch):
    monkeypatch.setattr(pipeline_module, "enroll_speaker", lambda inputs: ("ok", np.ones(192, dtype=np.float32)))
    monkeypatch.setattr(pipeline_module.FakeASR, "transcribe", staticmethod(lambda audio: "bật đèn phòng khách"))

    pipeline = FullDuplexPipeline(FullDuplexConfig())
    pipeline.tts = FakeTTS()
    pipeline.detector = FakeDetector()
    pipeline.turn_collector = FakeTurnCollector()
    pipeline.state_machine.set_state(FullDuplexState.SPEAKING)

    interrupt_chunk = np.ones(1280, dtype=np.float32)
    first = pipeline.process_audio(interrupt_chunk)
    assert first.turn_reason == "Listening for command..."
    assert pipeline.detector.state == FullDuplexState.INTERRUPTED_LISTENING.value

    followup_audio = np.ones(2560, dtype=np.float32)
    second = pipeline.process_audio(followup_audio)
    assert second.fake_asr_text == "bật đèn phòng khách"
    assert second.assistant_audio_path == "/tmp/fake_tts.wav"
    assert second.turn_reason.startswith("Finalized by:")
    assert pipeline.detector.state == FullDuplexState.SPEAKING.value

