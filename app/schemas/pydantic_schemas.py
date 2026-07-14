import pydantic 
from pydantic import BaseModel, Field
from enum import Enum as PyEnum
from typing import Optional

class TicketType(PyEnum):
    Technical = "Technical"
    Billing = "Billing"
    General_Support = "General Support"

class ClassificationResult(BaseModel):
    category: TicketType = Field(..., description="The category of the ticket")
    confidence: float = Field(..., ge=0.0, le=1.0, description="The confidence score of the classification")

class CriticResult(BaseModel):
    resolved: bool
    rejection_reason: str = Field( default=None)
