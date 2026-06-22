import torch
import torchaudio
from gtts import gTTS
import os

text = "Xin chào, tôi đang giới thiệu về hệ thống nhà thông minh Lumi. Bạn có thể ngắt lời tôi bất cứ lúc nào nếu bạn là người dùng đích, vì hệ thống này hỗ trợ nhận diện người dùng thông minh, chỉ ngắt khi đúng giọng của bạn thôi nhé."

print("Generating TTS...")
tts = gTTS(text=text, lang="vi")
tts.save("temp.mp3")

print("Converting to 16kHz mono WAV...")
waveform, sample_rate = torchaudio.load("temp.mp3")
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

torchaudio.save("assistant_long_response.wav", waveform, 16000)
os.remove("temp.mp3")
print("Done! Saved assistant_long_response.wav")
