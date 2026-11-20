
from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    artifacts: dict[str, bool]


class ForecastRecord(BaseModel):
    stock_code: str
    week: str
    predicted_demand: float


class ForecastResponse(BaseModel):
    count: int
    source: str
    records: list[ForecastRecord]


class APIMessage(BaseModel):
    message: str
    data: dict[str, Any] = {}
