"""Schemas for FastAPI api implementation"""
from enum import Enum
from pydantic import BaseModel, Field
from typing import List


class FlightTypes(str, Enum):
    """
    Types of flights that exists
    """
    international = "I"
    national = "N"

class Flights(BaseModel):
    """
    Flight class with it's correspondant properties
    """
    OPERA: str
    TIPOVUELO: FlightTypes
    MES: int = Field(..., ge=1, le=12)

class PredictionRequest(BaseModel):
    """
    Defines PredictionRequest as a list of Flights
    """
    flights: List[Flights]
