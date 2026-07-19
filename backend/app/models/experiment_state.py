from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ExperimentState(BaseModel):
    status: str
    dataset: str
    sample_size: int = 0
    train_size: int = 0
    test_size: int = 0
    evaluated_users: int = 0
    models: list[str] = []
    metrics_ready: bool = False
    split_boundary_date: str | None = None
    artifacts: dict[str, str] = {}
    updated_at: datetime

