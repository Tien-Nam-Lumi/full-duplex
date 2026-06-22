# PROGESS.md — Full-Duplex Voice Assistant Progress Tracker

> Filename follows the user's requested spelling: `PROGESS.md`. If desired, also copy this to `PROGRESS.md`.

## Current Status

Status: Not implemented yet.

Primary objective: build a CPU-runnable full-duplex voice assistant harness before integrating real ASR, real TTS, real AEC, PVAD2, SpeechBrain, or SmartTurn.

## Progress Checklist

### Phase 0 — Repo Setup

- [ ] Decide package location.
- [ ] Create `full_duplex_assistant/` package or adapt existing package.
- [ ] Add `scripts/` directory if missing.
- [ ] Add `tests/` directory if missing.
- [ ] Ensure `pytest` can run.
- [ ] Ensure CPU-only environment works.

Notes:

```text
Keep this phase small. Do not add real model dependencies yet.
```

---

### Phase 1 — Core Types and Config

- [ ] Create config object for thresholds and sample rate.
- [ ] Create `AudioChunk` dataclass.
- [ ] Create `DetectorScores` dataclass.
- [ ] Create `PipelineOutput` dataclass.
- [ ] Add state enum or constants.

Expected files:

```text
full_duplex_assistant/config.py
full_duplex_assistant/audio_types.py
```

Acceptance:

- [ ] Types import correctly.
- [ ] Config can change thresholds without modifying pipeline logic.

---

### Phase 2 — Audio Buffers

- [ ] Implement rolling buffer.
- [ ] Implement pending buffer.
- [ ] Implement target turn buffer.
- [ ] Support append by chunk.
- [ ] Support export as one waveform.
- [ ] Support clear/reset.
- [ ] Support pre-roll extraction.

Expected files:

```text
full_duplex_assistant/buffers.py
tests/test_buffers.py
```

Acceptance:

- [ ] Buffer duration limit works.
- [ ] Pre-roll is preserved.
- [ ] Buffer test passes.

---

### Phase 3 — Mock Components

- [ ] Implement mock VAD.
- [ ] Implement mock target speaker gate.
- [ ] Implement mock echo gate.
- [ ] Implement fake ASR.
- [ ] Implement fake LLM.
- [ ] Implement mock TTS player.

Expected files:

```text
full_duplex_assistant/vad.py
full_duplex_assistant/speaker_gate.py
full_duplex_assistant/echo_gate.py
full_duplex_assistant/asr.py
full_duplex_assistant/llm.py
full_duplex_assistant/tts.py
```

Acceptance:

- [ ] Mock TTS can play/stop/report `is_playing`.
- [ ] Fake ASR returns deterministic text.
- [ ] Fake LLM returns deterministic response.
- [ ] Mock gates can be controlled in tests.

---

### Phase 4 — State Machine

- [ ] Implement states:
  - [ ] `IDLE`
  - [ ] `LOCKING_TARGET`
  - [ ] `ACTIVE_USER_TURN`
  - [ ] `PROCESSING`
  - [ ] `ASSISTANT_SPEAKING`
  - [ ] `INTERRUPTED_LISTENING`
- [ ] Log state transitions.
- [ ] Add unit tests for state transitions.

Expected files:

```text
full_duplex_assistant/state_machine.py
tests/test_state_machine.py
```

Acceptance:

- [ ] Target speech from idle enters active turn.
- [ ] Imposter speech does not enter active turn.
- [ ] Processing starts assistant response.
- [ ] Assistant speaking can transition to interrupted listening.

---

### Phase 5 — FullDuplexPipeline

- [ ] Implement `FullDuplexPipeline.step(audio_chunk)`.
- [ ] Connect buffers.
- [ ] Connect VAD, target gate, echo gate.
- [ ] Connect state machine.
- [ ] Connect fake ASR/LLM/TTS.
- [ ] Return `PipelineOutput` each step.

Expected files:

```text
full_duplex_assistant/pipeline.py
tests/test_pipeline_smoke.py
```

Acceptance:

- [ ] Pipeline can process chunks continuously.
- [ ] Pipeline does not require GPU.
- [ ] Pipeline does not require network.
- [ ] CPU smoke test passes.

---

### Phase 6 — Barge-In Logic

- [ ] Add stricter thresholds during assistant speaking.
- [ ] Require consecutive target chunks.
- [ ] Reject echo-only input.
- [ ] Reject imposter input.
- [ ] Stop TTS only on confirmed target barge-in.
- [ ] Enter `INTERRUPTED_LISTENING` after valid target interrupt.
- [ ] Add pre-roll audio to target turn buffer.

Expected files:

```text
tests/test_barge_in.py
tests/test_echo_gate.py
```

Acceptance:

- [ ] Assistant speaking + echo only -> no stop.
- [ ] Assistant speaking + imposter -> no stop.
- [ ] Assistant speaking + target -> TTS stop.
- [ ] State becomes `INTERRUPTED_LISTENING`.
- [ ] Target turn buffer includes pre-roll.

---

### Phase 7 — Turn Finalization

- [ ] Add simple silence-based end-of-turn fallback.
- [ ] Add SmartTurn abstraction.
- [ ] Keep fake SmartTurn for tests.
- [ ] Finalize target turn after interrupt.
- [ ] Run fake ASR.
- [ ] Run fake LLM.
- [ ] Start TTS again.

Expected files:

```text
full_duplex_assistant/smart_turn.py
tests/test_pipeline_smoke.py
```

Acceptance:

- [ ] Interrupted target turn finalizes.
- [ ] Fake ASR receives buffered target audio.
- [ ] Assistant responds again.

---

### Phase 8 — Demo Script

- [ ] Create a synthetic full-duplex demo.
- [ ] Simulate assistant speaking.
- [ ] Simulate echo-only chunks.
- [ ] Simulate imposter chunks.
- [ ] Simulate target barge-in chunks.
- [ ] Print timeline of events.

Expected file:

```text
scripts/run_full_duplex_demo.py
```

Acceptance:

- [ ] Demo prints clear state transitions.
- [ ] Demo shows echo rejected.
- [ ] Demo shows imposter rejected.
- [ ] Demo shows target barge-in accepted.
- [ ] Demo shows TTS stopped and restarted.

---

### Phase 9 — Synthetic Evaluation

- [ ] Create eval scenarios.
- [ ] Compute barge-in recall.
- [ ] Compute false barge-in rate.
- [ ] Compute imposter interrupt rate.
- [ ] Compute echo interrupt rate.
- [ ] Compute interrupt latency.
- [ ] Compute turn start cut.

Expected file:

```text
scripts/run_synthetic_bargein_eval.py
```

Acceptance:

- [ ] Eval script runs on CPU.
- [ ] Metrics are printed in table form.
- [ ] Failures identify scenario names.

---

### Phase 10 — Real Module Integration Later

Do not start this phase until Phase 1–9 pass.

- [ ] Add real VAD adapter.
- [ ] Add SpeechBrain ECAPA adapter.
- [ ] Add PVAD2 adapter.
- [ ] Add SmartTurn ONNX adapter.
- [ ] Add real TTS adapter.
- [ ] Add real ASR adapter.
- [ ] Add optional playback reference / AEC adapter.

Acceptance:

- [ ] Mock mode remains default for tests.
- [ ] Real mode is optional.
- [ ] Unit tests do not require external services.

## Known Risks

### Risk 1 — False barge-in from assistant echo

Mitigation:

- do not interrupt using VAD alone
- use echo risk
- require target-speaker confidence
- use stricter assistant-speaking thresholds

### Risk 2 — Missing the first word of target interrupt

Mitigation:

- rolling buffer
- pending buffer
- pre-roll copy when target is confirmed

### Risk 3 — Imposter stops assistant

Mitigation:

- target speaker gate
- consecutive target chunks
- higher threshold during assistant speech

### Risk 4 — SmartTurn finalizes too early

Mitigation:

- use silence fallback first
- make thresholds configurable
- keep interrupted listening state separate from normal active turn

## Commands to Maintain

Codex should make these work:

```bash
pytest -q
python scripts/run_full_duplex_demo.py
python scripts/run_synthetic_bargein_eval.py
```

## Latest Progress Log

### Entry 001 — Initial Planning

Date: TBD

Summary:

```text
Created implementation documents for Codex. No code has been implemented yet.
```

Next action:

```text
Start Phase 1: create config, dataclasses, state constants, and CPU smoke test skeleton.
```
