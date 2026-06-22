# AGENTS.md — Full-Duplex Voice Assistant Implementation Guide


## Project Mission

Build a full-duplex voice assistant harness where the assistant can speak while continuously listening for the enrolled target user. The system must support target-aware barge-in: when the target user interrupts, TTS stops quickly, the user's speech is buffered, SmartTurn decides end-of-turn, then the assistant responds again.

This is not a simple VAD-while-TTS system. It must distinguish:

- assistant echo from real user speech
- target user from imposter speaker
- background TV/music/noise from valid barge-in
- partial target speech from complete user turns

## Core Architecture

```text
Mic audio stream
  -> rolling audio buffer
  -> VAD / PVAD / speaker verification / optional DOA
  -> echo-aware barge-in detector
  -> target turn buffer
  -> SmartTurn end-of-turn detector
  -> fake or real ASR
  -> LLM response stub
  -> TTS playback
  -> while TTS is playing, mic continues listening
```

## Required State Machine

Implement or preserve this high-level state machine:

```text
IDLE
  -> LOCKING_TARGET
  -> ACTIVE_USER_TURN
  -> PROCESSING
  -> ASSISTANT_SPEAKING
  -> INTERRUPTED_LISTENING
  -> PROCESSING
  -> ASSISTANT_SPEAKING
```

### State Definitions

#### IDLE
No active user turn. Mic is streaming. Assistant may be silent.

#### LOCKING_TARGET
Speech activity exists, but target identity is not yet confirmed. Accumulate short chunks and run target checks.

#### ACTIVE_USER_TURN
Target speaker is confirmed. Buffer speech until SmartTurn or silence finalizes the turn.

#### PROCESSING
User turn is finalized. Run fake ASR, fake LLM, and prepare response.

#### ASSISTANT_SPEAKING
TTS is playing, but mic remains active. Barge-in detection runs with stricter thresholds.

#### INTERRUPTED_LISTENING
Target barge-in was confirmed during assistant speech. TTS must stop or duck. Continue buffering target speech until end-of-turn.

## Implementation Rules

### General Code Rules

- Keep modules small and testable.
- Prefer deterministic tests with synthetic audio fixtures.
- Avoid global mutable state except for clearly documented runtime config.
- Add type hints to all new Python functions.
- Do not silently swallow exceptions in audio pipeline code.
- Use clear log events for state transitions, threshold decisions, and interruption decisions.
- Do not introduce heavy dependencies unless necessary.
- CPU smoke tests must run without GPU.

### Audio Rules

- Default sample rate: 16 kHz unless existing project config says otherwise.
- Use chunked streaming, ideally 20 ms, 30 ms, or 40 ms chunks.
- Maintain at least three buffers:
  - `rolling_buffer`: recent audio, about 2 seconds
  - `pending_buffer`: possible speech before target confirmation
  - `target_turn_buffer`: confirmed target-user speech
- When target barge-in is confirmed, prepend relevant rolling/pending audio so the beginning of the interrupt is not cut.

### Barge-In Rules

Never interrupt TTS using VAD alone.

A valid target interrupt requires evidence from several signals:

```python
valid_interrupt = (
    assistant_is_speaking
    and vad_score >= vad_threshold
    and target_score >= target_threshold
    and echo_risk <= echo_risk_threshold
    and consecutive_target_chunks >= required_chunks
)
```

When assistant is speaking, thresholds must be stricter than when the assistant is silent.

Suggested starting thresholds:

```python
NORMAL_MODE = {
    "vad_threshold": 0.50,
    "target_threshold": 0.80,
    "sv_threshold": 0.35,
    "required_chunks": 2,
}

ASSISTANT_SPEAKING_MODE = {
    "vad_threshold": 0.60,
    "target_threshold": 0.90,
    "sv_threshold": 0.42,
    "required_chunks": 4,
}
```

These are starting points only. Keep them configurable.

### Echo-Aware Rules

Milestone 1 may use a heuristic echo-risk estimator. Real AEC is not required at the beginning.

Minimum echo-aware inputs:

- `assistant_is_speaking`
- current TTS/playback PCM chunk if available
- mic energy
- playback energy
- optional correlation between mic chunk and playback reference

Echo risk should increase when:

- assistant is speaking
- mic energy strongly follows playback energy
- mic chunk correlates with playback reference
- no independent target-speaker evidence exists

Echo risk should decrease when:

- target speaker score is strong
- speech energy starts before/after playback energy pattern
- mic signal contains speech-like components not explained by playback

### TTS Rules

Implement a `TTSPlayer` interface.

Required methods:

```python
class TTSPlayer:
    def play(self, text: str) -> None: ...
    def stop(self) -> None: ...
    def is_playing(self) -> bool: ...
    def current_playback_chunk(self) -> bytes | None: ...
```

For early milestones, use `MockTTSPlayer` with generated sine/noise/speech-like PCM. Real TTS can be added later.

### ASR / LLM Rules

Do not block full-duplex harness development on real ASR or real LLM.

Milestone 1 and 2 can use:

- `FakeASR`: returns fixed text such as `"turn on the light"`
- `FakeLLM`: returns fixed assistant response
- `MockTTSPlayer`: simulates speaking

Real ASR/LLM/TTS should be integration layers, not hard dependencies for core state-machine tests.

## Recommended File Structure

Codex should create or adapt files like this:

```text
full_duplex_assistant/
  __init__.py
  config.py
  audio_types.py
  buffers.py
  state_machine.py
  vad.py
  speaker_gate.py
  echo_gate.py
  smart_turn.py
  tts.py
  asr.py
  llm.py
  pipeline.py
  logging_utils.py

tests/
  test_buffers.py
  test_state_machine.py
  test_echo_gate.py
  test_barge_in.py
  test_pipeline_smoke.py

scripts/
  run_full_duplex_demo.py
  run_synthetic_bargein_eval.py
```

If the existing repository already has equivalent files, modify the existing structure instead of duplicating functionality.

## Main Interfaces

### AudioChunk

Use a simple typed structure:

```python
@dataclass
class AudioChunk:
    pcm: np.ndarray
    sample_rate: int
    timestamp_sec: float
    duration_sec: float
```

### DetectorScores

```python
@dataclass
class DetectorScores:
    vad_score: float
    target_score: float
    sv_score: float | None
    echo_risk: float
    is_speech: bool
    is_target: bool
    should_interrupt: bool
```

### PipelineOutput

```python
@dataclass
class PipelineOutput:
    state: str
    event: str | None
    transcript: str | None
    assistant_text: str | None
    debug: dict[str, Any]
```

## Required Events

Log these events clearly:

```text
speech_started
target_lock_started
target_confirmed
target_rejected
user_turn_finalized
assistant_response_started
assistant_response_stopped
target_barge_in_confirmed
echo_rejected
imposter_rejected
state_changed
```

## Test Requirements

Codex must implement synthetic tests for these cases:

1. Assistant silent, target speaks -> user turn starts and finalizes.
2. Assistant silent, imposter speaks -> no target turn.
3. Assistant speaking, no user -> no interruption.
4. Assistant speaking, echo only -> no interruption.
5. Assistant speaking, target interrupts -> TTS stops and state becomes `INTERRUPTED_LISTENING`.
6. Assistant speaking, imposter interrupts -> TTS continues.
7. Target begins during assistant speech -> start of target speech is preserved from rolling buffer.
8. SmartTurn finalizes interrupted user turn -> system enters `PROCESSING`.
9. Fake ASR/LLM/TTS loop returns to `ASSISTANT_SPEAKING`.
10. Full CPU smoke test completes without external models.

## Metrics

The synthetic eval script should report:

```text
barge_in_recall
false_barge_in_rate
imposter_interrupt_rate
echo_interrupt_rate
average_interrupt_latency_ms
turn_start_cut_ms
turn_end_delay_ms
```

## Acceptance Criteria

The implementation is acceptable when:

- CPU smoke tests pass.
- The pipeline runs without real ASR, real LLM, real TTS, or GPU.
- Target barge-in stops mock TTS.
- Echo-only audio does not stop mock TTS.
- Imposter speech does not stop mock TTS.
- The interrupted target speech buffer includes pre-roll audio.
- The state machine transitions are covered by tests.
- Config thresholds can be changed without editing pipeline logic.

## Do Not Do Yet

Do not implement these until the harness is stable:

- production Android playback capture
- real multi-mic beamforming
- heavy model training
- complicated AEC neural network
- cloud ASR dependency in unit tests
- real device UI

## Development Order

1. Create config, types, buffers.
2. Create state machine.
3. Create mock VAD, mock target speaker gate, mock echo gate.
4. Create mock TTS, fake ASR, fake LLM.
5. Create `FullDuplexPipeline.step(chunk)`.
6. Add tests for state transitions.
7. Add synthetic barge-in demo.
8. Add metrics script.
9. Only then connect real VAD/PVAD/SV/SmartTurn modules.

## Final Reminder

The goal is target-aware full duplex. Do not optimize for generic speech interruption. Optimize for:

```text
Target user interrupts -> stop assistant.
Echo/imposter/noise interrupts -> assistant continues.
```
