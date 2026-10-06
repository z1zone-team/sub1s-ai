import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.errors.exceptions import BaseAiException

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(BaseAiException)
    async def base_ai_exception_handler(_request: Request, exc: BaseAiException):
        logger.error(f"BaseAiException: {exc.message} (status: {exc.status_code})")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.__class__.__name__,
                    "message": exc.message,
                },
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(_request: Request, exc: Exception):
        logger.exception(f"Unhandled Exception: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "AI 워커 내부 오류가 발생했습니다.",
                },
            },
        )
