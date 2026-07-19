from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import demo, experiment, metrics, recommendations, users
from app.config import get_settings
from app.services.agentic_service import AgenticServiceError
from app.services.data_service import DataValidationError


settings = get_settings()
app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(DataValidationError)
@app.exception_handler(AgenticServiceError)
@app.exception_handler(FileNotFoundError)
async def handle_known_errors(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(experiment.router)
app.include_router(users.router)
app.include_router(recommendations.router)
app.include_router(metrics.router)
app.include_router(demo.router)
