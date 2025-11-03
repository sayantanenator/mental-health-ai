# src/api/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import aiofiles
import os
import uuid
from datetime import datetime

from src.utils.config import Config
from src.models.model import MultiModalMentalHealthModel
from src.data.preprocessing import TextProcessor
from src.utils.audio_utils import extract_audio_features


class AnalysisRequest(BaseModel):
    text: str
    consent_given: bool = True


class AnalysisResponse(BaseModel):
    risk_level: int
    risk_description: str
    confidence: float
    timestamp: str


app = FastAPI(title="Mental Health Analysis API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
config = Config()
model = None
text_processor = None


@app.on_event("startup")
async def startup_event():
    global model, text_processor

    try:
        # Initialize text processor
        text_processor = TextProcessor(config.model.text_model)

        # Load model
        model = MultiModalMentalHealthModel(config)

        # Load trained weights if available
        if os.path.exists("models/best_model.pth"):
            checkpoint = torch.load(
                "models/best_model.pth", map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        print("✅ Model loaded successfully")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None


@app.get("/")
async def root():
    return {
        "message": "Mental Health Analysis API",
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_mental_health(
    text: str,
    consent_given: bool = False,
    audio_file: UploadFile = File(None)
):
    if not consent_given:
        raise HTTPException(status_code=400, detail="Consent required")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Process text
        text_input = text_processor.tokenize_text(text)
        text_input = {k: v.unsqueeze(0) for k, v in text_input.items()}

        # Process audio if provided
        if audio_file:
            # Save uploaded file temporarily
            temp_path = f"/tmp/{uuid.uuid4()}_{audio_file.filename}"
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await audio_file.read()
                await f.write(content)

            audio_features = extract_audio_features(
                temp_path,
                sample_rate=config.data.audio_sample_rate,
                n_mfcc=config.model.audio_features,
                max_duration=config.data.max_audio_duration
            )

            # Clean up
            os.remove(temp_path)
        else:
            audio_features = torch.zeros(1, 100, config.model.audio_features)

        audio_features = audio_features.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            outputs = model(text_input, audio_features)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)

        risk_level = prediction.item()
        confidence_score = confidence.item()

        return AnalysisResponse(
            risk_level=risk_level,
            risk_description=model.get_risk_level(risk_level),
            confidence=confidence_score,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
