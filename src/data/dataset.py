# src/data/dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import os


class MentalHealthDataset(Dataset):
    def __init__(self, texts, audio_paths, labels, text_processor, audio_processor, config):
        self.texts = texts
        self.audio_paths = audio_paths
        self.labels = labels
        self.text_processor = text_processor
        self.audio_processor = audio_processor
        self.config = config

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            text = str(self.texts[idx])
            audio_path = self.audio_paths[idx]
            label = int(self.labels[idx])

            # Process text
            text_input = self.text_processor.tokenize_text(text)

            # Process audio
            audio_features = self.audio_processor.extract_features(audio_path)

            return {
                'text_input': text_input,
                'audio_features': audio_features,
                'labels': torch.tensor(label, dtype=torch.long)
            }

        except Exception as e:
            # Return dummy sample on error
            return self._create_dummy_sample()

    def _create_dummy_sample(self):
        text_input = self.text_processor.tokenize_text("sample text")
        audio_features = torch.zeros(100, self.config.model.audio_features)

        return {
            'text_input': text_input,
            'audio_features': audio_features,
            'labels': torch.tensor(0, dtype=torch.long)
        }


class DataManager:
    def __init__(self, config):
        self.config = config
        self.text_processor = None
        self.audio_processor = None

    def initialize_processors(self):
        from src.data.preprocessing import TextProcessor, AudioProcessor

        self.text_processor = TextProcessor(
            model_name=self.config.model.text_model,
            max_length=self.config.data.max_text_length
        )
        self.audio_processor = AudioProcessor(
            sample_rate=self.config.data.audio_sample_rate,
            n_mfcc=self.config.model.audio_features,
            max_duration=self.config.data.max_audio_duration
        )

    def load_dataset(self, csv_path):
        """Load dataset from CSV file"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Validate required columns
        required_columns = ['text', 'audio_path', 'label']
        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if self.text_processor is None or self.audio_processor is None:
            self.initialize_processors()

        return MentalHealthDataset(
            texts=df['text'].tolist(),
            audio_paths=df['audio_path'].tolist(),
            labels=df['label'].tolist(),
            text_processor=self.text_processor,
            audio_processor=self.audio_processor,
            config=self.config
        )
