import librosa
import numpy as np


def extract_features(audio_path):
    """
    Extract acoustic features from a voice recording.
    """

    audio, sample_rate = librosa.load(audio_path, sr=None)

    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_scaled = np.mean(mfccs.T, axis=0)

    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    )

    zero_crossing_rate = np.mean(
        librosa.feature.zero_crossing_rate(audio)
    )

    features = np.hstack([
        mfccs_scaled,
        spectral_centroid,
        zero_crossing_rate
    ])

    return features


if __name__ == "__main__":
    sample_audio = "data/sample_voice.wav"
    features = extract_features(sample_audio)

    print("Extracted Features:")
    print(features)
