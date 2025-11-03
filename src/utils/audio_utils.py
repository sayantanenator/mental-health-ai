# src/utils/audio_utils.py
import librosa
import numpy as np
import torch


def extract_audio_features(audio_path, sample_rate=16000, n_mfcc=13, max_duration=30):
    """Extract audio features from file"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sample_rate)

        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        # Ensure maximum duration
        max_samples = int(max_duration * sr)
        if len(y_trimmed) > max_samples:
            y_trimmed = y_trimmed[:max_samples]

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=y_trimmed,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=2048,
            hop_length=512
        )

        # Normalize
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / \
            np.std(mfcc, axis=1, keepdims=True)

        # Pad/truncate to fixed length
        target_length = 100
        if mfcc.shape[1] < target_length:
            pad_width = target_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :target_length]

        return torch.FloatTensor(mfcc.T)

    except Exception as e:
        # Return zero features on error
        return torch.zeros(100, n_mfcc)
