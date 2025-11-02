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
    validation_split: float


@dataclass
class APIConfig:
    host: str
    port: int
    debug: bool
    workers: int


class Config:
    def __init__(self, config_path="config/default.yaml"):
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Create configuration objects
        self.model = ModelConfig(**config_data['model'])
        self.training = TrainingConfig(**config_data['training'])
        self.api = APIConfig(**config_data['api'])

        # Store raw config
        self.data = config_data

    def save_config(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.data, f)
