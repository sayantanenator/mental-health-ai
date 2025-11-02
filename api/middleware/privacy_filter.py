from fastapi import Request, Response


async def privacy_filter_middleware(request: Request, call_next):
    # TODO: Inspect and redact PII if needed.
    response: Response = await call_next(request)
    return response
