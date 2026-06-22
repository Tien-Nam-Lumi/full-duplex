# Full-Duplex Target-Speaker PoC — Minimal Plan for Claude

## Purpose

Build a minimal full-duplex PoC for a smart-home voice assistant.

The most important thing to prove is:

```text
Assistant is speaking
+
Microphone is still listening
+
Target speaker starts talking
↓
System detects target speaker
↓
Assistant stops TTS immediately
```

Do **not** focus on ASR, LLM, or complex SmartTurn in the first milestone.

The first PoC should prove the core full-duplex behavior:

```text
TTS playback does not block microphone listening.
Target speaker can interrupt assistant speech.
Non-target speaker should not stop assistant.
```

---

## Existing Components

Assume we already have:

- Realtime mic chunk input
- Streaming TTS playback
- TTS stop / queue control, or ability to add it
- VAD backend
- Target speaker backend:
  - WavLM fine-tuned model if available
  - otherwise current SpeechBrain / WavLM speaker verification model
- Optional SmartTurn backend
- Optional fake ASR / fake command result

---

## Milestone 1 — Core Full-Duplex Interruption

### Goal

While assistant is speaking, mic must remain active and detect target speaker interruption.

### Required behavior

```text
Case A: Assistant speaking, no user speech
→ TTS continues

Case B: Assistant speaking, imposter speaks
→ TTS continues

Case C: Assistant speaking, target speaker speaks
→ TTS stops
→ print/log: TARGET INTERRUPTION DETECTED
```

### No ASR required

Do not implement real ASR in Milestone 1.

After target interruption is detected, just log the event:

```text
[TTS] stopped
[INTERRUPT] target speaker detected
[STATE] LISTENING
```

---

## Milestone 1 Architecture

```text
TTS Player
  ↓
Speaker Output

Mic Input
  ↓
Realtime Chunk Buffer
  ↓
VAD
  ↓
Target Speaker Score
  ↓
Interruption Detector
  ↓
Stop TTS
```

Important:

```text
TTS playback and mic processing must run concurrently.
```

---

## State Machine

Use a minimal state machine:

```text
IDLE
LISTENING
SPEAKING
INTERRUPTED
```

### Main flow

```text
IDLE / LISTENING
  ↓ assistant starts TTS
SPEAKING
  ↓ target speaker interruption confirmed
INTERRUPTED
  ↓ stop TTS
LISTENING
```

### Required state behavior

When state is `SPEAKING`:

```text
- keep reading mic chunks
- run VAD
- run target speaker scoring
- update interruption detector
- stop TTS if interruption confirmed
```

---

## Interruption Detector Logic

Do not trigger from a single chunk.

Use duration confirmation.

Recommended parameters:

```python
CHUNK_MS = 80
INTERRUPT_CONFIRM_MS = 600
VAD_THRESHOLD = 0.5
TARGET_SPEAKER_THRESHOLD = current_best_threshold
```

Core logic:

```python
if state == "SPEAKING":
    mic_chunk = read_mic_chunk()

    vad_score = vad(mic_chunk)
    sv_score = target_speaker_score(rolling_sv_buffer)

    is_speech = vad_score >= VAD_THRESHOLD
    is_target = sv_score >= TARGET_SPEAKER_THRESHOLD

    if is_speech and is_target:
        target_positive_ms += CHUNK_MS
    else:
        # decay instead of hard reset
        target_positive_ms = max(0, target_positive_ms - CHUNK_MS)

    if target_positive_ms >= INTERRUPT_CONFIRM_MS:
        stop_tts()
        clear_tts_queue()
        state = "INTERRUPTED"
        log("TARGET INTERRUPTION DETECTED")
```

Target speaker score should use a rolling buffer, not only one 80 ms chunk.

Recommended:

```text
Rolling SV buffer: 0.8s to 1.5s
```

Example:

```python
rolling_sv_buffer.push(mic_chunk)
sv_audio = rolling_sv_buffer.last(seconds=1.2)
sv_score = target_speaker_score(sv_audio)
```

---

## TTS Requirements

TTS player must expose:

```python
start_tts(text: str)
stop_tts()
clear_tts_queue()
is_tts_playing()
```

For Milestone 1, TTS can simply play a long fixed sentence.

Example:

```text
"Xin chào, tôi đang giới thiệu về hệ thống nhà thông minh Lumi..."
```

During this speech, user will try to interrupt.

---

## Mic Requirements

The microphone must not pause while TTS is playing.

This is the key requirement.

Bad behavior:

```text
TTS playing → mic input paused
```

Correct behavior:

```text
TTS playing → mic input continues → VAD/SV still running
```

---

## Milestone 1 Demo

Demo script should do:

```text
1. Start assistant TTS with a long response.
2. Keep mic stream active.
3. User speaks while assistant is talking.
4. If user is target speaker:
   - stop TTS
   - print TARGET INTERRUPTION DETECTED
5. If user is imposter:
   - do not stop TTS
```

Required logs:

```text
[STATE] SPEAKING
[VAD] score=...
[SV] score=...
[INTERRUPT] candidate_ms=...
[INTERRUPT] confirmed
[TTS] stopped
[STATE] LISTENING
```

---

## Milestone 1 Evaluation

Create or record these 5 tests first:

```text
01_no_interruption
02_target_interrupt_near
03_target_interrupt_far
04_imposter_interrupt_near
05_assistant_echo_only
```

Metrics:

```text
1. Target interrupt detected? yes/no
2. Stop latency from target speech start
3. Imposter false stop? yes/no
4. Assistant self-echo false stop? yes/no
```

Initial success criteria:

```text
Target interruption recall >= 90%
Stop latency <= 800 ms
Imposter false stop <= 10%
Assistant echo false stop near 0
```

---

## Milestone 2 — Add Fake ASR and Response Loop

Only after Milestone 1 works.

After target interruption:

```text
Stop TTS
↓
Enter LISTENING
↓
Collect user speech
↓
End by SmartTurn or silence fallback
↓
Fake ASR returns predefined command
↓
Generate fixed response
↓
Start new TTS
```

No real ASR needed yet.

Fake ASR example:

```python
class FakeASR:
    def transcribe(self, audio):
        return "bật đèn phòng khách"
```

Fixed response example:

```python
response = "Đã bật đèn phòng khách."
start_tts(response)
```

---

## Milestone 2 End-of-Command Logic

Use SmartTurn as primary, silence as fallback.

Recommended:

```python
MIN_UTTERANCE_MS = 500
SILENCE_BEFORE_SMARTTURN_MS = 300
MAX_SILENCE_FALLBACK_MS = 900
```

Logic:

```python
if state == "LISTENING_AFTER_INTERRUPT":
    collect_audio(mic_chunk)

    if target_speech_seen_ms >= MIN_UTTERANCE_MS:
        if silence_ms >= SILENCE_BEFORE_SMARTTURN_MS:
            smartturn_score = smartturn(user_audio_buffer)

            if smartturn_score >= SMARTTURN_END_THRESHOLD:
                finalize_user_command()

        if silence_ms >= MAX_SILENCE_FALLBACK_MS:
            finalize_user_command()
```

This prevents the demo from hanging if SmartTurn fails.

---

## Milestone 3 — Add Echo-Aware Handling

Only after Milestone 1 and 2 work.

The first implementation can use raw mic.

Then add simple echo guard.

### TTS reference buffer

Every TTS audio chunk sent to speaker should also be stored in a timestamped reference buffer.

```python
tts_reference_buffer.push(tts_audio_chunk)
```

### Echo correlation guard

Compare mic chunk with aligned TTS reference chunk.

```python
echo_corr = normalized_cross_correlation(mic_chunk, tts_ref_chunk)
```

Guard logic:

```python
if echo_corr > ECHO_CORR_THRESHOLD and sv_score < STRONG_TARGET_THRESHOLD:
    suppress_interruption_candidate()
```

Important:

```text
Do not block real target interruption only because echo correlation is high.
Target speaker score should be allowed to override echo guard.
```

---

## Milestone 4 — Optional Real ASR

Only after full-duplex interruption is stable.

Replace FakeASR with real ASR.

Flow:

```text
Target interruption
↓
Stop TTS
↓
Collect utterance
↓
SmartTurn/silence finalization
↓
Real ASR
↓
LLM/fixed smart-home action
↓
TTS response
```

---

## What Not To Do Initially

Do not implement these in the first milestone:

```text
- Real ASR
- Real LLM
- MOSS/Moshi
- Neural AEC
- Complex multi-turn dialogue
- Complex SmartTurn tuning
```

These are not needed to prove the core full-duplex behavior.

---

## Suggested File Structure

```text
full_duplex_poc/
  audio_types.py
  state_machine.py
  tts_player.py
  vad_backend.py
  target_speaker_backend.py
  interruption_detector.py
  fake_asr.py
  smartturn_backend.py
  echo_guard.py
  run_milestone1_demo.py
  run_milestone2_fake_asr_demo.py
  run_offline_eval.py
```

---

## Required Deliverable

Build Milestone 1 first:

```text
run_milestone1_demo.py
```

It should:

```text
1. Start long TTS response.
2. Keep mic open.
3. Continuously print VAD/SV/interruption status.
4. Stop TTS if target speaker interrupts.
5. Ignore imposter speech.
```

After this works, build:

```text
run_milestone2_fake_asr_demo.py
```

It should:

```text
1. Start long TTS.
2. Detect target interruption.
3. Stop TTS.
4. Collect command.
5. Use SmartTurn or silence fallback to finalize.
6. Fake ASR returns predefined command.
7. Generate new TTS response.
```

---

## Final Definition of Success

The PoC succeeds when this behavior is demonstrated:

```text
Assistant is talking.
Target user speaks over it.
Assistant hears the target user while still talking.
Assistant stops itself quickly.
Assistant ignores non-target speakers.
```

This is the core full-duplex capability.
