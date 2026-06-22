from __future__ import annotations

import logging

import numpy as np

from .audio_types import CHUNK_SIZE, PipelineOutput
from .audio_utils import load_audio_mono_16k
from .buffers import RollingAudioBuffer
from .config import FullDuplexConfig
from .fake_asr import FakeASR
from .interruption_detector import InterruptionDetector
from .llm import FakeLLM
from .smart_turn import get_turn_detector
from .state_machine import FullDuplexState, FullDuplexStateMachine
from .target_speaker_backend import enroll_speaker
from .tts_player import create_tts_player
from .turn_collector import PostInterruptTurnCollector
from .vad_backend import SileroVAD

logger = logging.getLogger(__name__)


class FullDuplexPipeline:
    def __init__(self, config: FullDuplexConfig | None = None) -> None:
        self.config = config or FullDuplexConfig()
        self.tts, self.tts_mode = create_tts_player()
        self.turn_detector = get_turn_detector("silence")
        self.llm = FakeLLM()
        self.state_machine = FullDuplexStateMachine()
        self.detector: InterruptionDetector | None = None
        self.turn_collector: PostInterruptTurnCollector | None = None
        self.leftover_audio = RollingAudioBuffer(max_samples=self.config.chunk_size * 4)
        self.latest_interruption_info = ""
        self.last_cumulative_samples = 0

    def enroll(self, audio_inputs: list) -> str:
        msg, emb = enroll_speaker(audio_inputs)
        if emb is None:
            return msg

        self.detector = InterruptionDetector(target_embedding=emb, sv_threshold=self.config.sv_threshold)
        self.turn_collector = PostInterruptTurnCollector(
            SileroVAD(threshold=self.config.vad_threshold, hangover_frames=self.config.vad_hangover_frames),
            self.turn_detector,
        )
        self.leftover_audio.clear()
        self.last_cumulative_samples = 0
        self.latest_interruption_info = ""
        self.state_machine.set_state(FullDuplexState.LISTENING)
        self.detector.set_state(FullDuplexState.LISTENING.value)
        return msg

    def start_assistant(self) -> tuple[str, str | None]:
        self.latest_interruption_info = ""

        if self.detector is None:
            return "❌ Please enroll target speaker first.", None

        success = self.tts.start_tts(self.config.assistant_intro_text)
        if not success:
            return "❌ Error: Cannot start assistant audio.", None

        self.state_machine.set_state(FullDuplexState.SPEAKING)
        self.detector.set_state(FullDuplexState.SPEAKING.value)

        if hasattr(self.tts, "get_audio_path"):
            return f"Assistant is speaking... mode={self.tts_mode}", self.tts.get_audio_path()
        return f"Assistant is speaking... mode={self.tts_mode}", None

    def reset(self) -> str:
        if self.detector:
            self.detector.reset()
            self.detector.set_state(FullDuplexState.LISTENING.value)
        if self.turn_collector:
            self.turn_collector.reset()
        self.leftover_audio.clear()
        self.last_cumulative_samples = 0
        self.latest_interruption_info = ""
        self.state_machine.set_state(FullDuplexState.IDLE)
        self.tts.stop_tts()
        self.tts.clear_tts_queue()
        return "State Reset."

    def _display_state(self, base_state: str) -> str:
        if self.latest_interruption_info:
            return f"{self.latest_interruption_info}\n\n{base_state}"
        return base_state

    def process_audio(self, input_audio, gradio_audio_mode: str = "incremental") -> PipelineOutput:
        if self.detector is None:
            return PipelineOutput(
                state_text="Please enroll first.",
                tts_status=f"TTS Playing: {self.tts.is_tts_playing()}",
            )

        if self.detector.state == FullDuplexState.SPEAKING.value and not self.tts.is_tts_playing():
            logger.info("[PIPELINE] Natural TTS finish detected. Resetting state to LISTENING.")
            self.detector.set_state(FullDuplexState.LISTENING.value)
            self.detector.clear_interruption_state()
            self.state_machine.set_state(FullDuplexState.LISTENING)

        if input_audio is None:
            base_state = (
                f"State: {self.detector.state} | Candidate MS: {self.detector.target_positive_ms} "
                f"| SV Score: {self.detector.last_sv_score:.4f} (Target={self.detector.is_target_verified})"
            )
            return PipelineOutput(
                state_text=self._display_state(base_state),
                tts_status=f"TTS Playing: {self.tts.is_tts_playing()}",
            )

        try:
            audio = load_audio_mono_16k(input_audio)
            raw_len = len(audio)
            if gradio_audio_mode == "cumulative":
                if raw_len > self.last_cumulative_samples:
                    audio = audio[self.last_cumulative_samples :]
                    self.last_cumulative_samples = raw_len
                else:
                    audio = np.array([], dtype=np.float32)
            else:
                self.last_cumulative_samples = 0
        except Exception as e:
            logger.error("Error loading audio: %s", e)
            return PipelineOutput(
                state_text="Error loading audio",
                tts_status=f"TTS Playing: {self.tts.is_tts_playing()}",
            )

        incoming = self.leftover_audio.to_numpy()
        if incoming.size:
            audio = np.concatenate([incoming, audio])
        self.leftover_audio.clear()

        offset = 0
        while offset + CHUNK_SIZE <= len(audio):
            chunk = audio[offset : offset + CHUNK_SIZE]
            offset += CHUNK_SIZE

            if self.detector.state == FullDuplexState.INTERRUPTED_LISTENING.value:
                assert self.turn_collector is not None
                tc_result = self.turn_collector.process_chunk(chunk)

                if tc_result.get("turn_finalized"):
                    logger.info("Turn finalized! Reason: %s", tc_result["reason"])
                    self.detector.set_state(FullDuplexState.FAKE_ASR_PROCESSING.value)
                    self.state_machine.set_state(FullDuplexState.FAKE_ASR_PROCESSING)

                    fake_text = FakeASR.transcribe(tc_result["audio"])
                    response_text = self.llm.generate_response(fake_text)

                    self.detector.set_state(FullDuplexState.RESPONDING.value)
                    self.state_machine.set_state(FullDuplexState.RESPONDING)
                    response_audio_path = self.tts.start_tts(response_text)
                    if response_audio_path is None:
                        self.detector.set_state(FullDuplexState.LISTENING.value)
                        self.state_machine.set_state(FullDuplexState.LISTENING)
                        return PipelineOutput(
                            state_text=self._display_state("❌ Missing response audio file: assistant_short_response.wav"),
                            tts_status=f"TTS Playing: {self.tts.is_tts_playing()}",
                            fake_asr_text=fake_text,
                            turn_reason="Response audio missing",
                        )

                    self.detector.set_state(FullDuplexState.SPEAKING.value)
                    self.state_machine.set_state(FullDuplexState.SPEAKING)

                    reason_str = f"Finalized by: {tc_result['reason']} (score: {tc_result.get('smartturn_score')})"
                    return PipelineOutput(
                        state_text=self._display_state(f"State: {self.detector.state} | SV Score: {self.detector.last_sv_score:.4f}"),
                        tts_status=f"TTS Playing: {self.tts.is_tts_playing()}",
                        assistant_audio_path=response_audio_path,
                        fake_asr_text=fake_text,
                        turn_reason=reason_str,
                    )
                continue

            result = self.detector.process_chunk(chunk, self.tts)
            if result["interrupted"]:
                self.latest_interruption_info = (
                    "🚨 TARGET INTERRUPTION DETECTED 🚨\n"
                    f"Stop Latency: {result['stop_latency_ms']:.1f}ms\n"
                    f"State: {result['state']} | SV Score: {result['sv_score']:.4f}"
                )
                assert self.turn_collector is not None
                self.turn_collector.start(initial_audio=result["trigger_audio"])
                self.leftover_audio.append(audio[offset:])

                self.detector.set_state(FullDuplexState.INTERRUPTED_LISTENING.value)
                self.state_machine.set_state(FullDuplexState.INTERRUPTED_LISTENING)
                return PipelineOutput(
                    state_text=self._display_state(f"State: {result['state']} | SV Score: {result['sv_score']:.4f}"),
                    tts_status=f"TTS Playing: {self.tts.is_tts_playing()}",
                    assistant_audio_path=None,
                    fake_asr_text="",
                    turn_reason="Listening for command...",
                )

        self.leftover_audio.append(audio[offset:])
        base_state = (
            f"State: {self.detector.state} | Candidate MS: {self.detector.target_positive_ms} "
            f"| SV Score: {self.detector.last_sv_score:.4f} (Target={self.detector.is_target_verified})"
        )
        return PipelineOutput(
            state_text=self._display_state(base_state),
            tts_status=f"TTS Playing: {self.tts.is_tts_playing()}",
        )

