import torch
import soundfile as sf
import numpy as np
from transformers import VitsModel, AutoTokenizer
import os

print("Loading MMS-TTS-VIE model...")
model_id = "facebook/mms-tts-vie"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = VitsModel.from_pretrained(model_id)

text = "Đã bật đèn phòng khách."

inputs = tokenizer(text, return_tensors="pt")
print(f"Generating TTS for: {text}")

with torch.no_grad():
    output = model(**inputs).waveform
    
audio = output.squeeze().numpy()

# Normalize audio
audio = audio / np.max(np.abs(audio))

out_path = "full_duplex_poc/assistant_short_response.wav"
sf.write(out_path, audio, model.config.sampling_rate)
print(f"Saved short response to {out_path}")
