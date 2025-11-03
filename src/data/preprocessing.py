# src/data/preprocessing.py
import re
import torch
import librosa
import numpy as np
from transformers import AutoTokenizer


class TextProcessor:
    def __init__(self, model_name="bert-base-uncased", max_length=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def preprocess_text(self, text):
        """Clean and preprocess text input"""
        if not text or not isinstance(text, str):
            return ""

        # Basic cleaning
        text = text.lower().strip()
        text = re.sub(r'[^\w\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text)

        return text

    def tokenize_text(self, text):
        """Tokenize text for model input"""
        cleaned_text = self.preprocess_text(text)

        encoding = self.tokenizer(
            cleaned_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {k: v.squeeze(0) for k, v in encoding.items()}


class AudioProcessor:
    def __init__(self, sample_rate=16000, n_mfcc=13, max_duration=30):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_duration = max_duration
        self.target_length = 100

    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)

            # Ensure maximum duration
            max_samples = int(self.max_duration * sr)
            if len(y_trimmed) > max_samples:
                y_trimmed = y_trimmed[:max_samples]

            return y_trimmed, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file: {str(e)}")

    def extract_features(self, audio_path):
        """Extract MFCC features from audio"""
        try:
            y, sr = self.load_audio(audio_path)

            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=2048,
                hop_length=512
            )

            # Normalize features
            mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / \
                np.std(mfcc, axis=1, keepdims=True)

            # Pad or truncate to fixed length
            if mfcc.shape[1] < self.target_length:
                pad_width = self.target_length - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :self.target_length]

            return torch.FloatTensor(mfcc.T)
        except:
            # Return dummy features if audio processing fails
            return torch.zeros(self.target_length, self.n_mfcc)
