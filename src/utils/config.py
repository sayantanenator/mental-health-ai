# src/utils/config.py
import yaml
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    text_model: str
    audio_features: int
    hidden_size: int
    num_classes: int
    dropout: float


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    early_stopping_patience: int


@dataclass
class DataConfig:
    max_text_length: int
    audio_sample_rate: int
    max_audio_duration: int


class Config:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = "config/default.yaml"

        self.config_path = config_path
        self.load_config()

    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        except FileNotFoundError:
            # Create default config
            config_data = self.get_default_config()
            os.makedirs("config", exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(config_data, f)

        # Create configuration objects
        self.model = ModelConfig(**config_data['model'])
        self.training = TrainingConfig(**config_data['training'])
        self.data = DataConfig(**config_data['data'])

        # Store raw config
        self.data = config_data

    def get_default_config(self):
        return {
            'model': {
                'text_model': 'distilbert-base-uncased',
                'audio_features': 13,
                'hidden_size': 128,
                'num_classes': 3,
                'dropout': 0.3
            },
            'training': {
                'batch_size': 8,
                'learning_rate': 2e-5,
                'epochs': 10,
                'early_stopping_patience': 3
            },
            'data': {
                'max_text_length': 256,
                'audio_sample_rate': 16000,
                'max_audio_duration': 30
            }
        }
