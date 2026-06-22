import gradio as gr
import numpy as np
import logging
import os

from .target_speaker_backend import enroll_speaker
from .interruption_detector import InterruptionDetector
from .tts_player import create_tts_player, BrowserAudioTTSPlayer
from .audio_utils import load_audio_mono_16k
from .audio_types import CHUNK_SIZE
from .smart_turn import get_turn_detector
from .turn_collector import PostInterruptTurnCollector
from .fake_asr import FakeASR
from .vad_backend import SileroVAD

GRADIO_AUDIO_MODE = os.getenv("GRADIO_AUDIO_MODE", "incremental")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global instances for a single-user demo
tts, TTS_MODE = create_tts_player()
detector = None
turn_detector = get_turn_detector("silence")  # Default to silence for M2 demo safety
turn_collector = None
leftover_audio = np.array([], dtype=np.float32)
last_cumulative_samples = 0
latest_interruption_info = ""

def handle_enroll(audio1, audio2, audio3):
    global detector, leftover_audio, last_cumulative_samples, turn_collector
    msg, emb = enroll_speaker([audio1, audio2, audio3])
    if emb is not None:
        # Default start threshold for speechbrain_ft without znorm is roughly 0.29
        detector = InterruptionDetector(target_embedding=emb, sv_threshold=0.29)
        turn_collector = PostInterruptTurnCollector(SileroVAD(threshold=0.40, hangover_frames=5), turn_detector)
        leftover_audio = np.array([], dtype=np.float32)
        last_cumulative_samples = 0
        detector.set_state("LISTENING")
    return msg

def start_assistant():
    global detector, tts, latest_interruption_info
    latest_interruption_info = ""
    
    if detector is None:
        return "❌ Please enroll target speaker first.", None
    
    text = "Xin chào, tôi đang giới thiệu về hệ thống nhà thông minh Lumi. Bạn có thể ngắt lời tôi bất cứ lúc nào nếu bạn là người dùng đích, vì hệ thống này hỗ trợ nhận diện người dùng thông minh, chỉ ngắt khi đúng giọng của bạn thôi nhé."
    
    success = tts.start_tts(text)
    if not success:
        return "❌ Error: Cannot start assistant audio.", None

    detector.set_state("SPEAKING")

    if hasattr(tts, "get_audio_path"):
        return f"Assistant is speaking... mode={TTS_MODE}", tts.get_audio_path()

    return f"Assistant is speaking... mode={TTS_MODE}", None

def reset_state():
    global detector, leftover_audio, last_cumulative_samples, latest_interruption_info, turn_collector
    if detector:
        detector.reset()
        detector.set_state("LISTENING")
    if turn_collector:
        turn_collector.reset()
    leftover_audio = np.array([], dtype=np.float32)
    last_cumulative_samples = 0
    latest_interruption_info = ""
    tts.stop_tts()
    tts.clear_tts_queue()
    return "State Reset."

def process_audio(input_audio):
    global detector, tts, leftover_audio, last_cumulative_samples, latest_interruption_info, turn_collector
    
    if detector is None:
        return "Please enroll first.", f"TTS Playing: {tts.is_tts_playing()}", gr.update(), "", ""
        
    if detector.state == "SPEAKING" and not tts.is_tts_playing():
        logger.info("[APP] Natural TTS finish detected. Resetting state to LISTENING.")
        detector.set_state("LISTENING")
        detector.clear_interruption_state()
        
    def get_display_str(base_str):
        if latest_interruption_info:
            return f"{latest_interruption_info}\n\n{base_str}"
        return base_str

    if input_audio is None:
        base_state = f"State: {detector.state} | Candidate MS: {detector.target_positive_ms} | SV Score: {detector.last_sv_score:.4f} (Target={detector.is_target_verified})"
        return get_display_str(base_state), f"TTS Playing: {tts.is_tts_playing()}", gr.update(), gr.update(), gr.update()
        
    try:
        audio = load_audio_mono_16k(input_audio)
        logger.info(f"[APP] Received audio chunk. Shape: {audio.shape}")
        
        raw_len = len(audio)
        if GRADIO_AUDIO_MODE == "cumulative":
            if raw_len > last_cumulative_samples:
                audio = audio[last_cumulative_samples:]
                last_cumulative_samples = raw_len
            else:
                audio = np.array([], dtype=np.float32)
        else:
            last_cumulative_samples = 0
            
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        return f"Error loading audio", f"TTS Playing: {tts.is_tts_playing()}", gr.update(), gr.update(), gr.update()
        
    audio = np.concatenate([leftover_audio, audio])
    
    offset = 0
    while offset + CHUNK_SIZE <= len(audio):
        chunk = audio[offset:offset+CHUNK_SIZE]
        offset += CHUNK_SIZE
        # If we are in LISTENING_AFTER_INTERRUPT, feed chunks to turn collector instead
        if detector.state == "LISTENING_AFTER_INTERRUPT":
            tc_result = turn_collector.process_chunk(chunk)
            
            if tc_result.get("turn_finalized"):
                logger.info(f"Turn finalized! Reason: {tc_result['reason']}")
                detector.set_state("FAKE_ASR_PROCESSING")
                
                fake_text = FakeASR.transcribe(tc_result["audio"])
                response_text = "Đã bật đèn phòng khách." if "bật đèn" in fake_text else "Xin lỗi, tôi không hiểu."
                
                detector.set_state("RESPONDING")
                response_audio_path = tts.start_tts(response_text)
                if response_audio_path is None:
                    detector.set_state("LISTENING")
                    return (
                        get_display_str("❌ Missing response audio file: assistant_short_response.wav"),
                        f"TTS Playing: {tts.is_tts_playing()}",
                        None,
                        fake_text,
                        "Response audio missing"
                    )
                
                detector.set_state("SPEAKING")
                
                reason_str = f"Finalized by: {tc_result['reason']} (score: {tc_result.get('smartturn_score')})"
                
                return (
                    get_display_str(f"State: {detector.state} | SV Score: {detector.last_sv_score:.4f}"), 
                    f"TTS Playing: {tts.is_tts_playing()}",
                    response_audio_path,
                    fake_text,
                    reason_str
                )
            continue # skip interruption detection for this chunk
            
        # Normal process_chunk for interruption detection
        result = detector.process_chunk(chunk, tts)
        
        # Display logic
        if result["interrupted"]:
            latest_interruption_info = f"🚨 TARGET INTERRUPTION DETECTED 🚨\nStop Latency: {result['stop_latency_ms']:.1f}ms\nState: {result['state']} | SV Score: {result['sv_score']:.4f}"
            
            # Start turn collector with trigger audio
            turn_collector.start(initial_audio=result["trigger_audio"])
            
            # Preserve any unprocessed samples from this callback
            leftover_audio = audio[offset:].copy()
            
            return (
                get_display_str(f"State: {result['state']} | SV Score: {result['sv_score']:.4f}"), 
                f"TTS Playing: {tts.is_tts_playing()}",
                None, # clear browser audio
                "", # clear fake ASR text
                "Listening for command..." # Turn reason clear
            )
            
    leftover_audio = audio[offset:]
    
    base_state = f"State: {detector.state} | Candidate MS: {detector.target_positive_ms} | SV Score: {detector.last_sv_score:.4f} (Target={detector.is_target_verified})"
    return get_display_str(base_state), f"TTS Playing: {tts.is_tts_playing()}", gr.update(), gr.update(), gr.update()

with gr.Blocks(title="Full-Duplex Target-Speaker PoC") as demo:
    gr.HTML("<h1 style='text-align: center;'>🎙️ Full-Duplex Target-Speaker PoC (Milestone 2: Fake ASR Loop)</h1>")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Target Speaker Enrollment")
            gr.Markdown("Upload or record up to 3 voice clips of the target user.")
            audio1 = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Enrollment 1")
            audio2 = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Enrollment 2")
            audio3 = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Enrollment 3")
            
            enroll_btn = gr.Button("🔑 Enroll Target Voice", variant="primary")
            enroll_out = gr.HTML()
            enroll_btn.click(handle_enroll, inputs=[audio1, audio2, audio3], outputs=[enroll_out])
            
        with gr.Column(scale=1):
            gr.Markdown("### 2. Demo Interruption")
            gr.Markdown("Click 'Start Assistant Speaking'. While it is speaking, talk into the microphone. Only the enrolled target speaker should be able to interrupt and stop the TTS.")
            
            start_tts_btn = gr.Button("🗣️ Start Assistant Speaking")
            tts_status = gr.Textbox(label="TTS Start Action")
            
            assistant_audio = gr.Audio(
                label="Assistant Audio",
                type="filepath",
                autoplay=True,
                interactive=False,
            )
            
            start_tts_btn.click(start_assistant, outputs=[tts_status, assistant_audio])
            
            gr.Markdown("### 3. Microphone Stream")
            mic_input = gr.Audio(sources=["microphone"], type="numpy", streaming=True, label="Speak into Microphone")
            
            with gr.Row():
                fake_asr_out = gr.Textbox(label="Fake ASR Result (User Command)")
                turn_reason_out = gr.Textbox(label="Turn Finalization Reason")

            state_out = gr.Textbox(label="Interruption Detector Status")
            tts_is_playing_out = gr.Textbox(label="TTS Real-time Status")
            
            reset_btn = gr.Button("🔄 Reset Pipeline")
            reset_btn.click(reset_state, outputs=[tts_status])
            
            mic_input.stream(process_audio, inputs=[mic_input], outputs=[state_out, tts_is_playing_out, assistant_audio, fake_asr_out, turn_reason_out])

if __name__ == '__main__':
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
