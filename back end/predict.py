import sounddevice as sd
import numpy as np
import joblib
import tempfile
import soundfile as sf
from features import extract_features

model = joblib.load("../models/stress_model.pkl")

def record_audio(duration=5, fs=22050):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Recording complete.")
    return audio, fs

audio, fs = record_audio()

with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
    sf.write(tmpfile.name, audio, fs)
    features = extract_features(tmpfile.name)

prediction = model.predict([features])
print("Prediction:", prediction[0])
