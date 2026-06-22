# GOAL.md — Full-Duplex Voice Assistant Harness

## One-Line Goal

Build a target-aware full-duplex voice assistant harness that can keep listening while speaking, stop TTS only when the enrolled target user interrupts, then capture the full user turn and respond again.

## Why This Project Exists

A normal assistant is usually half-duplex: it speaks, then listens. A better home voice assistant must be full-duplex: it should allow the real owner to interrupt naturally while the assistant is speaking.

However, full-duplex is dangerous if implemented with only VAD. The assistant may stop itself because of its own echo, TV/music, background noise, or another person. This project solves that by combining target-speaker verification, echo-aware logic, buffering, and turn-taking.

## Product Behavior

The desired behavior:

```text
Assistant: "I will turn on the living room lights and then..."
Target user: "No, turn on the bedroom light."
Assistant immediately stops speaking.
Assistant listens until the target user finishes.
Assistant responds to the new command.
```

The undesired behavior:

```text
Assistant speaks -> its own echo triggers VAD -> assistant stops itself.
Imposter speaks -> assistant stops even though it is not the enrolled user.
TV/music plays -> assistant stops randomly.
Target interrupts -> assistant misses the first word.
Target interrupts -> assistant stops but finalizes too early.
```

## Scope

### In Scope

- Full-duplex state machine
- Streaming audio pipeline
- Rolling/pending/target buffers
- Mock TTS playback
- Fake ASR and fake LLM for harness testing
- Echo-aware barge-in heuristic
- Target-speaker gate abstraction
- SmartTurn or simple EoT abstraction
- Synthetic tests and metrics
- CPU-only smoke test

### Out of Scope for First Implementation

- Real Android playback capture
- Production AEC
- Real cloud ASR
- Real LLM service
- Real production TTS
- Beamforming and DOA
- GPU-only model inference
- UI integration

These can be added after the harness is stable.

## Target Architecture

```text
Audio input
  -> FullDuplexPipeline.step(audio_chunk)
  -> VAD score
  -> Target speaker score
  -> Echo risk score
  -> State machine decision
  -> TTS control
  -> Turn buffer
  -> SmartTurn finalization
  -> ASR
  -> LLM
  -> TTS
```

## Key Design Principle

Do not treat all speech as interruption.

Only this should interrupt the assistant:

```text
confirmed target-user speech during assistant playback
```

These should not interrupt:

```text
assistant echo
imposter speech
background music
TV speech
short noise burst
low-confidence target speech
```

## Core Milestones

### Milestone 1 — Harness Skeleton

Create the core project structure, config, types, buffers, state machine, and mock components.

Deliverables:

- `FullDuplexPipeline`
- `MockTTSPlayer`
- `FakeASR`
- `FakeLLM`
- `RollingAudioBuffer`
- basic tests

Success criteria:

- CPU tests pass
- assistant can go from user turn -> processing -> speaking
- no real ML models required

### Milestone 2 — Target-Aware Barge-In

Add interruption detection while TTS is speaking.

Deliverables:

- assistant-speaking mode
- stricter thresholds during TTS
- target interrupt confirmation
- TTS stop on target barge-in
- no stop on echo-only or imposter-only input

Success criteria:

- target interrupt stops TTS
- echo does not stop TTS
- imposter does not stop TTS

### Milestone 3 — Turn Capture After Interrupt

After barge-in, continue listening until the user finishes.

Deliverables:

- `INTERRUPTED_LISTENING` state
- target turn buffer with pre-roll audio
- SmartTurn or silence-based finalization
- fake ASR/LLM response loop

Success criteria:

- beginning of interrupted speech is preserved
- user turn finalizes cleanly
- assistant responds again

### Milestone 4 — Synthetic Evaluation

Build deterministic synthetic tests and metrics.

Deliverables:

- synthetic echo-only scenario
- synthetic target-interrupt scenario
- synthetic imposter-interrupt scenario
- synthetic background-noise scenario
- metrics script

Success criteria:

- report barge-in recall
- report false barge-in rate
- report interrupt latency
- report buffer cut amount

### Milestone 5 — Real Module Integration

Only after the harness passes synthetic tests, connect real modules.

Possible modules:

- Silero VAD
- SpeechBrain ECAPA fine-tuned model
- PVAD2
- SmartTurn ONNX
- real TTS service
- real ASR service

Success criteria:

- same tests still pass with mock mode
- real mode can be run separately
- no unit test requires GPU or network

## Target Metrics

Initial synthetic goals:

```text
Target barge-in recall:       >= 95%
Echo false interruption:      <= 1%
Imposter false interruption:  <= 5%
Average interrupt latency:    <= 500 ms
Turn start cut:               <= 200 ms
CPU smoke test:               pass
```

Real-device goals can be stricter later after AEC and speaker models improve.

## Minimum Demo

A successful demo should show:

```text
1. Assistant starts speaking.
2. Mic stream continues.
3. Echo-only chunks are ignored.
4. Imposter speech is ignored.
5. Target user interrupts.
6. TTS stops.
7. Target speech is buffered.
8. SmartTurn finalizes the user turn.
9. Fake ASR returns text.
10. Fake LLM returns response.
11. Assistant speaks again.
```

## Final Definition of Done

The first implementation is done when Codex can run:

```bash
pytest -q
python scripts/run_full_duplex_demo.py
python scripts/run_synthetic_bargein_eval.py
```

and the output proves:

- target barge-in works
- echo rejection works
- imposter rejection works
- interrupted speech is not cut
- the assistant can respond again after interruption
