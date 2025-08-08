from sqlmodel import SQLModel, Field
from typing import Optional

# SQL Table
class Sentiment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    text: str
    prediction: str

# Pred Input
class PredictionInput(SQLModel):
    text: str

# Pred Output
class PredictionOutput(SQLModel):
    prediction: str