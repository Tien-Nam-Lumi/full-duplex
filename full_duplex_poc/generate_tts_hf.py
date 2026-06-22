import torch
import scipy.io.wavfile
from transformers import VitsModel, AutoTokenizer

text = "Xin chào, tôi đang giới thiệu về hệ thống nhà thông minh Lumi. Bạn có thể ngắt lời tôi bất cứ lúc nào nếu bạn là người dùng đích, vì hệ thống này hỗ trợ nhận diện người dùng thông minh, chỉ ngắt khi đúng giọng của bạn thôi nhé."
print("Loading model...")
model = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

print("Generating audio...")
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    output = model(**inputs).waveform

scipy.io.wavfile.write("assistant_long_response.wav", rate=model.config.sampling_rate, data=output[0].cpu().numpy())
print("Done!")
