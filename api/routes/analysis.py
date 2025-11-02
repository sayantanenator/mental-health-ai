from fastapi import APIRouter
from ..schemas.analysis import AnalysisRequest, AnalysisResponse

router = APIRouter()


@router.post("/analysis", response_model=AnalysisResponse)
async def analyze(payload: AnalysisRequest):
    # TODO: call models/inference Predictor
    return AnalysisResponse(result="neutral", confidence=0.5)
