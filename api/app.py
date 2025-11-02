# api/app.py
from src.utils.config import Config
from src.data_processing.audio_processor import AudioProcessor
from src.data_processing.text_processor import TextProcessor
from src.model.architecture import MultiModalMentalHealthModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import aiofiles
import os
from typing import Optional
import uuid
import sys
from pathlib import Path
import torch

# Ensure project root is on sys.path so `src` can be imported when running this file
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Pydantic models for request/response


class AnalysisRequest(BaseModel):
    text: str
    consent_given: bool = False


class AnalysisResponse(BaseModel):
    risk_level: int
    risk_description: str
    confidence: float
    all_probabilities: list[float]
    disclaimer: str


# Initialize app
app = FastAPI(
    title="Mental Health Analysis API",
    description="AI-powered mental health risk assessment using multimodal analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables (would use dependency injection in production)
config = Config()
text_processor = TextProcessor()
audio_processor = AudioProcessor()
model = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    try:
        model = MultiModalMentalHealthModel(config)
        # Load your trained weights here
        # model.load_state_dict(torch.load("path/to/model.pth"))
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/")
async def root():
    return {"message": "Mental Health Analysis API", "status": "healthy"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_mental_health(
    text: str,
    consent_given: bool = False,
    audio_file: Optional[UploadFile] = File(None)
):
    """
    Analyze text and optional audio for mental health risk assessment

    - **text**: User's text input (required)
    - **consent_given**: User consent flag (required)
    - **audio_file**: Audio recording (optional)
    """

    # Validate consent
    if not consent_given:
        raise HTTPException(
            status_code=400,
            detail="User consent is required for analysis"
        )

    try:
        # Process text
        text_input = text_processor.tokenize_text(text)

        # Process audio if provided
        audio_features = None
        if audio_file:
            # Save uploaded file temporarily
            temp_path = f"/tmp/{uuid.uuid4()}_{audio_file.filename}"
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await audio_file.read()
                await f.write(content)

            # Extract audio features
            audio_features = audio_processor.extract_features(temp_path)

            # Clean up
            os.remove(temp_path)
        else:
            # Create dummy audio features if no audio provided
            audio_features = torch.zeros(100, config.model.audio_features)

        # Run inference (placeholder - you'd add your actual inference logic)
        # This is where you'd call your model
        risk_level = 0  # Placeholder
        confidence = 0.8  # Placeholder

        return AnalysisResponse(
            risk_level=risk_level,
            risk_description=model.get_risk_level(risk_level),
            confidence=confidence,
            all_probabilities=[0.7, 0.2, 0.1],  # Placeholder
            disclaimer="This analysis is for informational purposes only and not a medical diagnosis. Please consult healthcare professionals for medical advice."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze-text-only")
async def analyze_text_only(request: AnalysisRequest):
    """Analyze text only without audio"""
    return await analyze_mental_health(
        text=request.text,
        consent_given=request.consent_given,
        audio_file=None
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug
    )
