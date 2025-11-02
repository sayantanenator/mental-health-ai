# src/model/architecture.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import torchaudio


class MultiModalMentalHealthModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Text encoder
        text_config = AutoConfig.from_pretrained(config.model.text_model)
        self.text_encoder = AutoModel.from_pretrained(
            config.model.text_model,
            config=text_config
        )

        # Text classifier
        self.text_classifier = nn.Sequential(
            nn.Dropout(config.model.dropout),
            nn.Linear(text_config.hidden_size, config.model.hidden_size),
            nn.ReLU(),
            nn.Linear(config.model.hidden_size, 128)
        )

        # Audio encoder
        self.audio_lstm = nn.LSTM(
            input_size=config.model.audio_features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=config.model.dropout,
            bidirectional=True
        )

        # Audio classifier
        self.audio_classifier = nn.Sequential(
            nn.Linear(256, 128),  # 2 * hidden_size for bidirectional
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(128, 128)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.model.num_classes)
        )

        # Risk level mappings
        self.risk_levels = {
            0: "Low Risk - Normal emotional variation",
            1: "Moderate Risk - Suggest professional consultation",
            2: "High Risk - Urgent professional consultation recommended"
        }

    def forward(self, text_input, audio_features):
        # Text forward pass
        text_outputs = self.text_encoder(**text_input)
        text_embedding = text_outputs.last_hidden_state[:, 0, :]  # CLS token
        text_features = self.text_classifier(text_embedding)

        # Audio forward pass
        audio_output, (hidden, _) = self.audio_lstm(audio_features)
        # Concatenate forward and backward hidden states
        audio_features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        audio_features = self.audio_classifier(audio_features)

        # Feature fusion
        combined = torch.cat([text_features, audio_features], dim=1)
        output = self.fusion(combined)

        return output

    def get_risk_level(self, prediction_idx):
        return self.risk_levels.get(prediction_idx, "Unknown risk level")
