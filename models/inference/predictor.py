from .preprocessor import preprocess_input


class Predictor:
    """Simple inference pipeline stub."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        # TODO: Load model from model_path

    def predict(self, data):
        x = preprocess_input(data)
        # TODO: Run model inference
        return {"prediction": "neutral", "confidence": 0.5}
