from fastapi import Request, Response


async def auth_middleware(request: Request, call_next):
    # TODO: Implement real auth. For now, pass-through.
    response: Response = await call_next(request)
    return response
