from enum import Enum


class FullDuplexState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    INTERRUPTED_LISTENING = "LISTENING_AFTER_INTERRUPT"
    FAKE_ASR_PROCESSING = "FAKE_ASR_PROCESSING"
    RESPONDING = "RESPONDING"


class FullDuplexStateMachine:
    def __init__(self, initial_state: FullDuplexState = FullDuplexState.IDLE) -> None:
        self.state = initial_state

    def set_state(self, new_state: FullDuplexState) -> None:
        self.state = new_state

    @property
    def is_speaking(self) -> bool:
        return self.state == FullDuplexState.SPEAKING

    @property
    def is_listening_after_interrupt(self) -> bool:
        return self.state == FullDuplexState.INTERRUPTED_LISTENING

