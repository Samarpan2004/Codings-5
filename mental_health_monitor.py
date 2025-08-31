# mental_health_monitor.py
# Usage: analyze text and a recorded audio file (wav)
# python mental_health_monitor.py "I feel sad today" audio.wav

import sys, numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import librosa

def text_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)

def audio_features(path):
    y, sr = librosa.load(path, sr=16000)
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    pitch = np.mean(librosa.yin(y, fmin=75, fmax=600))
    return {"rms": float(rms), "zcr": float(zcr), "pitch": float(pitch)}

def main():
    if len(sys.argv)<3:
        print("Usage: python mental_health_monitor.py \"your text\" audio.wav"); return
    text = sys.argv[1]
    audio = sys.argv[2]
    print("Text sentiment:", text_sentiment(text))
    print("Audio features:", audio_features(audio))
    print("Note: This is not clinical. Use as a tracker not diagnosis.")

if __name__=="__main__":
    main()
