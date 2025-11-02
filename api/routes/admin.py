from fastapi import APIRouter

router = APIRouter()


@router.get("/admin/health")
async def health():
    return {"status": "ok"}
