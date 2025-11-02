from fastapi import APIRouter

router = APIRouter()


@router.post("/batch")
async def process_batch(items: list[dict]):
    # TODO: implement batch processing
    return {"processed": len(items)}
