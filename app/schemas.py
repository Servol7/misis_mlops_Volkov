from pydantic import BaseModel
from typing import List


class HealthResponse(BaseModel):
    status: str


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: str
    toxicity_score: float
    is_toxic: bool


class BatchPredictionRequest(BaseModel):
    texts: List[str]


class BatchPredictionItem(BaseModel):
    text: str
    toxicity_score: float
    is_toxic: bool


class BatchPredictionResponse(BaseModel):
    predictions: List[BatchPredictionItem]


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    max_length: int
    device: str
