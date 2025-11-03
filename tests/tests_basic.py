# tests/test_basic.py
import pytest
import torch
from src.models.architecture import MultiModalMentalHealthModel
from src.utils.config import Config


def test_model_initialization():
    """Test that model initializes correctly"""
    config = Config()
    model = MultiModalMentalHealthModel(config)

    assert model is not None
    assert hasattr(model, 'text_encoder')
    assert hasattr(model, 'audio_lstm')


def test_model_forward_pass():
    """Test model forward pass with dummy data"""
    config = Config()
    model = MultiModalMentalHealthModel(config)

    # Create dummy input
    batch_size = 2
    text_input = {
        'input_ids': torch.randint(0, 1000, (batch_size, 512)),
        'attention_mask': torch.ones((batch_size, 512))
    }
    audio_features = torch.randn(batch_size, 100, config.model.audio_features)

    # Forward pass
    output = model(text_input, audio_features)

    assert output.shape == (batch_size, config.model.num_classes)


def test_text_processor():
    """Test text preprocessing"""
    from src.data.text_processor import TextProcessor

    processor = TextProcessor()
    text = "Hello, world! This is a test."

    processed = processor.tokenize_text(text)

    assert 'input_ids' in processed
    assert 'attention_mask' in processed
