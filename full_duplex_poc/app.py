import gradio as gr
import logging
import os

from .pipeline import FullDuplexPipeline

GRADIO_AUDIO_MODE = os.getenv("GRADIO_AUDIO_MODE", "incremental")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

pipeline = FullDuplexPipeline()

def handle_enroll(audio1, audio2, audio3):
    return pipeline.enroll([audio1, audio2, audio3])

def start_assistant():
    return pipeline.start_assistant()

def reset_state():
    return pipeline.reset()

def process_audio(input_audio):
    output = pipeline.process_audio(input_audio, GRADIO_AUDIO_MODE)
    return (
        output.state_text,
        output.tts_status,
        output.assistant_audio_path if output.assistant_audio_path is not None else gr.update(),
        output.fake_asr_text if output.fake_asr_text else gr.update(),
        output.turn_reason if output.turn_reason else gr.update(),
    )

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
