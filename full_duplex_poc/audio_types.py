from dataclasses import dataclass
from enum import Enum

import numpy as np


SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
CHUNK_MS = 80


class PipelineState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    LISTENING_AFTER_INTERRUPT = "LISTENING_AFTER_INTERRUPT"
    FAKE_ASR_PROCESSING = "FAKE_ASR_PROCESSING"
    RESPONDING = "RESPONDING"


@dataclass(slots=True)
class AudioChunk:
    samples: np.ndarray
    sample_rate: int = SAMPLE_RATE
    source: str = "mic"


@dataclass(slots=True)
class DetectorScores:
    vad_speech: bool = False
    sv_score: float = 0.0
    target_verified: bool = False
    stop_latency_ms: float | None = None


@dataclass(slots=True)
class PipelineOutput:
    state_text: str
    tts_status: str
    assistant_audio_path: str | None = None
    fake_asr_text: str = ""
    turn_reason: str = ""
