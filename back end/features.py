import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=5, offset=0.5)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0])

    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    features = np.hstack([pitch, rms, zcr, mfcc])
    return features
