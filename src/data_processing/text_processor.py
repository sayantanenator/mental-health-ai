# src/data_processing/text_processor.py
import re
import torch
from transformers import AutoTokenizer


class TextProcessor:
    def __init__(self, model_name="bert-base-uncased", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def preprocess_text(self, text):
        """Clean and preprocess text input"""
        if not text or not isinstance(text, str):
            return ""

        # Basic cleaning
        text = text.lower().strip()
        text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove special chars
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace

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
