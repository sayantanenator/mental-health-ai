# src/models/model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MultiModalMentalHealthModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Text encoder
        text_config = AutoConfig.from_pretrained(config.model.text_model)
        self.text_encoder = AutoModel.from_pretrained(config.model.text_model)

        # Freeze early layers for efficiency
        for param in list(self.text_encoder.parameters())[:-4]:
            param.requires_grad = False

        # Text classifier
        self.text_classifier = nn.Sequential(
            nn.Dropout(config.model.dropout),
            nn.Linear(text_config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Audio encoder
        self.audio_lstm = nn.LSTM(
            input_size=config.model.audio_features,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=config.model.dropout
        )

        # Audio classifier
        self.audio_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(128, config.model.num_classes)
        )

        self.risk_levels = {
            0: "Low Risk - Normal emotional variation",
            1: "Moderate Risk - Suggest professional consultation",
            2: "High Risk - Urgent professional consultation recommended"
        }

    def forward(self, text_input, audio_features):
        # Text forward pass
        text_outputs = self.text_encoder(**text_input)
        text_embedding = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_classifier(text_embedding)

        # Audio forward pass
        audio_output, (hidden, _) = self.audio_lstm(audio_features)
        audio_features = self.audio_classifier(hidden[-1])

        # Feature fusion
        combined = torch.cat([text_features, audio_features], dim=1)
        output = self.fusion(combined)

        return output

    def get_risk_level(self, prediction_idx):
        return self.risk_levels.get(prediction_idx, "Unknown risk level")
