# models/inference/predictor.py
import torch
import numpy as np
from typing import Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)


class MentalHealthPredictor:
    def __init__(self, config, model_path: Optional[str] = None):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize processors
        from src.data.text_processor import TextProcessor
        from src.data.audio_processor import AudioProcessor

        self.text_processor = TextProcessor(
            model_name=config.model.text_model,
            max_length=config.data.max_text_length
        )
        self.audio_processor = AudioProcessor(
            sample_rate=config.data.audio_sample_rate,
            n_mfcc=config.model.audio_features,
            max_duration=config.data.max_audio_duration
        )

        # Initialize model
        from src.models.architecture import MultiModalMentalHealthModel
        self.model = MultiModalMentalHealthModel(config)

        # Load trained weights if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.warning(
                "No model weights loaded. Using randomly initialized model.")

        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path: str):
        """Load trained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            logger.info(f"Model loaded successfully from {model_path}")

        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise

    def predict(self, text: str, audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Make prediction on text and optional audio"""
        try:
            # Validate input
            if not text or not text.strip():
                raise ValueError("Text input cannot be empty")

            # Process text
            text_input = self.text_processor.tokenize_text(text)
            text_input = {k: v.unsqueeze(0).to(self.device)
                          for k, v in text_input.items()}

            # Process audio
            if audio_path and os.path.exists(audio_path):
                audio_features = self.audio_processor.extract_features(
                    audio_path)
            else:
                # Create dummy audio features if no audio provided
                audio_features = torch.zeros(
                    100, self.config.model.audio_features)
                logger.info("No audio provided, using dummy features")

            audio_features = audio_features.unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(text_input, audio_features)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, 1)

                risk_level = prediction.item()
                confidence_score = confidence.item()
                all_probs = probabilities.squeeze().cpu().numpy().tolist()

            # Get risk description
            risk_description = self.model.get_risk_level(risk_level)

            # Prepare response
            result = {
                'risk_level': risk_level,
                'risk_description': risk_description,
                'confidence': confidence_score,
                'all_probabilities': all_probs,
                # Placeholder for actual timestamp
                'timestamp': torch.tensor(len(text)).numpy(),
                'model_version': self.config.project.version
            }

            logger.info(
                f"Prediction completed - Risk Level: {risk_level}, Confidence: {confidence_score:.3f}")

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def batch_predict(self, texts: list[str], audio_paths: Optional[list[str]] = None) -> list[Dict[str, Any]]:
        """Make predictions on multiple samples"""
        if audio_paths is None:
            audio_paths = [None] * len(texts)

        if len(texts) != len(audio_paths):
            raise ValueError("Texts and audio_paths must have the same length")

        results = []
        for text, audio_path in zip(texts, audio_paths):
            try:
                result = self.predict(text, audio_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction failed for sample: {e}")
                # Add error result
                results.append({
                    'risk_level': -1,
                    'risk_description': 'Prediction failed',
                    'confidence': 0.0,
                    'all_probabilities': [0.0, 0.0, 0.0],
                    'error': str(e)
                })

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        return {
            'model_name': self.config.model.text_model,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'audio_features': self.config.model.audio_features,
            'num_classes': self.config.model.num_classes
        }
