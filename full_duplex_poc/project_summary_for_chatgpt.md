# Full-Duplex Target-Speaker PoC Summary

This document summarizes the work completed for Milestone 1 and outlines the next steps for Milestones 2 and 3. You can use this document to share the current project state and future plans with ChatGPT or other assistants.

## 1. What Has Been Implemented (Milestone 1)

We have successfully built the **Core Full-Duplex Interruption** behavior. The system allows an assistant (TTS) to speak while the microphone remains active. If the enrolled target speaker interrupts the assistant, the TTS playback is stopped immediately. The system ignores imposter speech.

### Key Components Built
All files are located in `full_duplex_poc/`:

1. **`audio_types.py` & `audio_utils.py`**:
   - Centralized audio constants (`CHUNK_SIZE = 1280`, `SAMPLE_RATE = 16000`).
   - Reused robust audio loading and resampling (`load_audio_mono_16k`).

2. **`vad_backend.py`**:
   - A stateful chunk-by-chunk wrapper around `Silero VAD`. It tracks active speech status over consecutive audio chunks.

3. **`target_speaker_backend.py`**:
   - Uses `speechbrain_ft` (Fine-tuned SpeechBrain ECAPA-TDNN) for speaker verification.
   - **Enrollment Process**: Before starting the real-time demo, the user records or uploads up to 3 voice clips (minimum 0.5s each) via the Gradio UI. The backend uses `enroll_speaker` to extract a 1D L2-normalized embedding for each clip and averages them. This creates the canonical "target speaker embedding" that is saved in memory.
   - Contains `extract_embedding` to compute similar 1D vectors on the fly for incoming real-time chunks.

4. **`interruption_detector.py`**:
   - Houses the core logic. Evaluates chunks in real-time.
   - Maintains a rolling buffer (`MAX_ROLLING_CHUNKS = 30`).
   - Every `SV_CHECK_EVERY_CHUNKS` (2 chunks), it asynchronously calculates the cosine similarity against the target speaker embedding using a `ThreadPoolExecutor` to avoid blocking the mic stream.
   - Requires `400ms` of consecutive target speech (`INTERRUPT_CONFIRM_MS`) to formally trigger an "INTERRUPTED" state, which stops the TTS.

5. **`tts_player.py`**:
   - Contains `LocalAudioTTSPlayer` which uses `sounddevice` to play a local fixed WAV file (`assistant_long_response.wav`) out of the local speaker. It correctly tracks playback state and stops immediately when requested.
   - Also contains `MockTTSPlayer` for offline or silent evaluations.

6. **`app.py`**:
   - The interactive Gradio application that brings it all together.
   - **Step 1: Enrollment**: The UI provides a dedicated section with 3 `gr.Audio` microphone components where the user records their baseline voice. Clicking "Enroll" locks in their voice profile.
   - **Step 2: Real-time Interruption**: The UI then uses `gr.Audio(streaming=True)` to stream real-time chunks from the browser microphone. As the user talks, it compares their incoming chunks against the enrolled voice profile to test the full-duplex interruption.

---

## 2. Next Steps: Milestone 2 — Fake ASR and Response Loop

Once Milestone 1 is verified to be robust, we will move to Milestone 2 to handle what happens *after* the user interrupts.

### Goal
Implement the end-of-turn detection and simulated back-and-forth conversation.

### Architecture Plan
1. **Post-Interruption State Machine**:
   - When `INTERRUPTED` triggers, transition to `LISTENING_AFTER_INTERRUPT`.
2. **Collect Utterance**:
   - Buffer the user's speech until a turn-completion is detected.
3. **Turn Finalization**:
   - End the turn using a SmartTurn VAD model, or fallback to simple silence detection.
   - Constants needed: `MIN_UTTERANCE_MS = 500`, `SILENCE_BEFORE_SMARTTURN_MS = 300`, `MAX_SILENCE_FALLBACK_MS = 900`.
4. **Fake ASR & Response**:
   - Once the user stops speaking, pass the buffered audio to a `FakeASR` class that simply returns a predefined command (e.g., "bật đèn phòng khách").
   - Feed the fake command to the `TTSPlayer` with a predefined response (e.g., "Đã bật đèn phòng khách").

---

## 3. Next Steps: Milestone 3 — Echo-Aware Handling

Currently, the microphone may pick up the assistant's own voice (echo) from the speakers. While SV naturally rejects the assistant's voice if it doesn't sound like the target, we need a robust echo-guard to prevent false positives or edge cases.

### Goal
Prevent the assistant's own TTS output from triggering the interruption detector or poisoning the VAD.

### Architecture Plan
1. **TTS Reference Buffer**:
   - Whenever the `TTSPlayer` generates audio, store a copy of the PCM chunks in a timestamped reference buffer.
2. **Echo Correlation Guard**:
   - Compare incoming microphone chunks with the aligned TTS reference chunks using Normalized Cross-Correlation.
   - `echo_corr = normalized_cross_correlation(mic_chunk, tts_ref_chunk)`
3. **Suppression Logic**:
   - If `echo_corr > ECHO_CORR_THRESHOLD`, we strongly penalize or suppress the interruption candidate score.
   - *Exception*: If the `sv_score` is overwhelmingly high (e.g., `STRONG_TARGET_THRESHOLD`), we allow the interruption to proceed, meaning the user is talking loudly *over* the assistant.
