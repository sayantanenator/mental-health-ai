from fastapi import Request, Response


async def rate_limiter_middleware(request: Request, call_next):
    # TODO: Add real rate limiting. For now, pass-through.
    response: Response = await call_next(request)
    return response
