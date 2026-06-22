import time
import numpy as np
import logging
import argparse
import os
import soundfile as sf
from full_duplex_poc.target_speaker_backend import enroll_speaker
from full_duplex_poc.interruption_detector import InterruptionDetector
from full_duplex_poc.tts_player import MockTTSPlayer
from full_duplex_poc.audio_utils import load_audio_mono_16k
from full_duplex_poc.audio_types import CHUNK_SIZE

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def generate_dummy_audio(duration_sec=3.0, freq=440.0):
    t = np.linspace(0, duration_sec, int(16000 * duration_sec), False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    return audio.astype(np.float32)

def run_test(name, detector, audio_data, should_stop):
    logger.info(f"\n--- Running Test: {name} ---")
    tts = MockTTSPlayer()
    detector.reset()
    detector.set_state("SPEAKING")
    tts.start_tts("Testing offline playback...")
    
    stop_latency = None
    interrupted = False
    
    offset = 0
    t0 = time.time()
    
    while offset + CHUNK_SIZE <= len(audio_data):
        chunk = audio_data[offset:offset+CHUNK_SIZE]
        offset += CHUNK_SIZE
        
        result = detector.process_chunk(chunk, tts)
        state = result["state"]
        
        if result["interrupted"] and not interrupted:
            interrupted = True
            stop_latency = result["stop_latency_ms"]
                
        # Simulate realtime delay
        time.sleep(CHUNK_SIZE / 16000.0)
        
    # Wait for any pending SV threads to close
    if detector.sv_future is not None:
        detector.sv_future.result()

    success = (interrupted == should_stop)
    logger.info(f"Result: {'PASS' if success else 'FAIL'} | Interrupted: {interrupted} (Expected: {should_stop}) | Latency: {stop_latency if stop_latency else 'N/A'}")
    return success, interrupted, stop_latency

def main():
    parser = argparse.ArgumentParser(description="Offline evaluation for Milestone 1 Full-Duplex PoC.")
    parser.add_argument("--enroll", type=str, help="Path to enrollment WAV file")
    parser.add_argument("--target", type=str, help="Path to target test WAV file (should interrupt)")
    parser.add_argument("--imposter", type=str, help="Path to imposter test WAV file (should NOT interrupt)")
    args = parser.parse_args()

    logger.info("Initializing Offline Eval...")
    
    if args.enroll and os.path.exists(args.enroll):
        logger.info(f"Enrolling from {args.enroll}")
        dummy_enroll = load_audio_mono_16k(args.enroll)
        real_mode = True
    else:
        logger.warning("No real enrollment file provided. Using DUMMY sine waves. This is a skeleton test only!")
        dummy_enroll = generate_dummy_audio(2.0, 440.0)
        real_mode = False

    msg, emb = enroll_speaker([dummy_enroll])
    
    if emb is None:
        logger.error("Enrollment failed.")
        return
        
    detector = InterruptionDetector(target_embedding=emb, sv_threshold=0.29)
    
    if real_mode:
        no_interrupt_audio = generate_dummy_audio(2.0, 0.0) # Silence
        
        target_audio = load_audio_mono_16k(args.target) if args.target and os.path.exists(args.target) else generate_dummy_audio(2.0, 440.0)
        imposter_audio = load_audio_mono_16k(args.imposter) if args.imposter and os.path.exists(args.imposter) else generate_dummy_audio(2.0, 880.0)
    else:
        no_interrupt_audio = generate_dummy_audio(2.0, 0.0) # Silence
        target_audio = generate_dummy_audio(2.0, 440.0)     # Matches target freq
        imposter_audio = generate_dummy_audio(2.0, 880.0)   # Different freq
        logger.warning("NOTE: Testing with dummy sine waves. Real VAD/SV models may ignore this. Provide --enroll, --target, --imposter for real eval.")
    
    run_test("No Interruption (Silence)", detector, no_interrupt_audio, False)
    run_test("Target Interruption", detector, target_audio, True)
    run_test("Imposter Interruption", detector, imposter_audio, False)

if __name__ == "__main__":
    main()
