import soundfile as sf
import numpy as np
import librosa
from pydub.utils import mediainfo
import io

def validate_audio(file_bytes):
    info = mediainfo(io.BytesIO(file_bytes))
    duration = float(info['duration'])
    if duration < 5 or duration > 10:
        raise ValueError("Audio duration should be between 5 and 10 seconds.")
    return True

def preprocess_audio(file):
    y, sr = librosa.load(file, sr=16000)
    if y.shape[0] > 160000:
        y = y[:160000]
    elif y.shape[0] < 160000:
        y = np.pad(y, (0, 160000 - y.shape[0]))
    return y.astype(np.float32)
