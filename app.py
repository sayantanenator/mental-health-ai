# api/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import aiofiles
import os
import uuid
import logging
from datetime import datetime

from src.utils.config import Config
from models.inference.predictor import MentalHealthPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models


class AnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      description="Text to analyze")
    consent_given: bool = Field(True, description="User consent flag")


class BatchAnalysisRequest(BaseModel):
    samples: List[AnalysisRequest]
    audio_files: Optional[List[str]] = None


class AnalysisResponse(BaseModel):
    risk_level: int
    risk_description: str
    confidence: float
    all_probabilities: List[float]
    model_version: str
    timestamp: str
    disclaimer: str = "This analysis is for informational purposes only and not a medical diagnosis. Please consult healthcare professionals for medical advice."


class ModelInfoResponse(BaseModel):
    model_name: str
    total_parameters: int
    trainable_parameters: int
    device: str
    audio_features: int
    num_classes: int


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
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
config = Config()
predictor = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global predictor
    try:
        model_path = config.model.model_path
        predictor = MentalHealthPredictor(config, model_path)
        logger.info("Mental Health Predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        predictor = None


async def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file and return path"""
    file_extension = os.path.splitext(upload_file.filename)[1]
    temp_filename = f"/tmp/{uuid.uuid4()}{file_extension}"

    async with aiofiles.open(temp_filename, 'wb') as f:
        content = await upload_file.read()
        await f.write(content)

    return temp_filename


async def cleanup_file(file_path: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup file {file_path}: {e}")


@app.get("/")
async def root():
    return {
        "message": "Mental Health Analysis API",
        "status": "healthy",
        "version": config.project.version,
        "model_loaded": predictor is not None
    }


@app.get("/health")
async def health_check():
    model_status = "loaded" if predictor is not None else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        model_info = predictor.get_model_info()
        return model_info
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get model information")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_mental_health(
    background_tasks: BackgroundTasks,
    text: str,
    consent_given: bool = False,
    audio_file: Optional[UploadFile] = File(None)
):
    """
    Analyze text and optional audio for mental health risk assessment
    """
    # Validate consent
    if not consent_given:
        raise HTTPException(
            status_code=400,
            detail="User consent is required for analysis"
        )

    # Validate predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable. Model not loaded."
        )

    audio_path = None

    try:
        # Save audio file if provided
        if audio_file:
            if audio_file.content_type not in ['audio/wav', 'audio/mpeg', 'audio/flac']:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported audio format. Use WAV, MP3, or FLAC."
                )

            audio_path = await save_upload_file(audio_file)
            background_tasks.add_task(cleanup_file, audio_path)

        # Run prediction
        result = predictor.predict(text, audio_path)

        # Prepare response
        response = AnalysisResponse(
            risk_level=result['risk_level'],
            risk_description=result['risk_description'],
            confidence=result['confidence'],
            all_probabilities=result['all_probabilities'],
            model_version=result['model_version'],
            timestamp=datetime.now().isoformat()
        )

        logger.info(f"Analysis completed for text length: {len(text)}")
        return response

    except Exception as e:
        # Clean up audio file in case of error
        if audio_path:
            background_tasks.add_task(cleanup_file, audio_path)

        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/batch")
async def batch_analyze_mental_health(
    request: BatchAnalysisRequest
):
    """
    Analyze multiple samples in batch
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable"
        )

    try:
        texts = [sample.text for sample in request.samples]
        audio_paths = request.audio_files if request.audio_files else [
            None] * len(texts)

        results = predictor.batch_predict(texts, audio_paths)

        return {
            "results": results,
            "total_samples": len(texts),
            "successful_predictions": len([r for r in results if r.get('risk_level', -1) >= 0]),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch analysis failed: {str(e)}")


@app.post("/analyze/text-only", response_model=AnalysisResponse)
async def analyze_text_only(request: AnalysisRequest):
    """Analyze text only without audio"""
    return await analyze_mental_health(
        background_tasks=BackgroundTasks(),
        text=request.text,
        consent_given=request.consent_given,
        audio_file=None
    )

# Error handlers


@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


@app.exception_handler(413)
async def request_too_large_handler(request, exc):
    return JSONResponse(
        status_code=413,
        content={"detail": "File too large. Maximum size is 10MB."}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        workers=config.api.workers
    )
