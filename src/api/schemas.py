# src/api/schemas.py
from pydantic import BaseModel
from typing import List, Optional


class AnalysisRequest(BaseModel):
    text: str
    consent_given: bool = True


class AnalysisResponse(BaseModel):
    risk_level: int
    risk_description: str
    confidence: float
    timestamp: str
    disclaimer: str = "This analysis is for informational purposes only and not a medical diagnosis."


class BatchAnalysisRequest(BaseModel):
    samples: List[AnalysisRequest]


class BatchAnalysisResponse(BaseModel):
    results: List[AnalysisResponse]
    total_processed: int
