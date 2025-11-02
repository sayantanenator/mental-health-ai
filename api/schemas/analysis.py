from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    text: str | None = None
    audio_path: str | None = None


class AnalysisResponse(BaseModel):
    result: str
    confidence: float
