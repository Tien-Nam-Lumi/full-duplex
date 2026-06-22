#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from full_duplex_poc.config import FullDuplexConfig
from full_duplex_poc.pipeline import FullDuplexPipeline
from full_duplex_poc.state_machine import FullDuplexState


class DemoTTS:
    def __init__(self):
        self.playing = False

    def start_tts(self, text=None):
        print(f"TTS start: {text}")
        self.playing = True
        return "/tmp/demo_tts.wav"

    def stop_tts(self):
        print("TTS stop")
        self.playing = False

    def clear_tts_queue(self):
        pass

    def is_tts_playing(self):
        return self.playing

    def get_audio_path(self):
        return "/tmp/demo_tts.wav"


class DemoDetector:
    def __init__(self):
        self.state = FullDuplexState.SPEAKING.value
        self.target_positive_ms = 0
        self.last_sv_score = 0.0
        self.is_target_verified = True
        self._calls = 0

    def set_state(self, state):
        self.state = state
        print(f"STATE -> {state}")

    def reset(self):
        self.state = FullDuplexState.IDLE.value

    def clear_interruption_state(self):
        self.target_positive_ms = 0

    def process_chunk(self, chunk, tts):
        self._calls += 1
        if self._calls == 1:
            return {
                "state": self.state,
                "interrupted": True,
                "stop_latency_ms": 90.0,
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


class DemoTurnCollector:
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
        print("Turn collector started")
        self.audio = []
        if initial_audio is not None:
            self.audio.append(initial_audio.copy())

    def process_chunk(self, chunk):
        self.calls += 1
        self.audio.append(chunk.copy())
        print(f"Turn collector chunk {self.calls}")
        if self.calls >= 2:
            return {
                "turn_finalized": True,
                "reason": "silence_fallback",
                "audio": np.concatenate(self.audio),
                "smartturn_score": None,
            }
        return {"turn_finalized": False}


def main():
    pipeline = FullDuplexPipeline(FullDuplexConfig())
    pipeline.tts = DemoTTS()
    pipeline.detector = DemoDetector()
    pipeline.turn_collector = DemoTurnCollector()

    pipeline.detector.set_state(FullDuplexState.SPEAKING.value)
    print("Assistant speaking...")

    interrupt_chunk = np.ones(1280, dtype=np.float32)
    output1 = pipeline.process_audio(interrupt_chunk)
    print(output1.state_text)
    print(output1.turn_reason)

    followup_audio = np.ones(2560, dtype=np.float32)
    output2 = pipeline.process_audio(followup_audio)
    print(output2.state_text)
    print(output2.fake_asr_text)
    print(output2.turn_reason)


if __name__ == "__main__":
    main()
