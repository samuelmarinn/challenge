from enum import Enum
from pydantic import BaseModel, Field
from typing import List


class FlightTypes(str, Enum):
    international = "I"
    national = "N"

class Flights(BaseModel):
    OPERA: str
    TIPOVUELO: FlightTypes
    MES: int = Field(..., ge=1, le=12)

class PredictionRequest(BaseModel):
    flights: List[Flights]
